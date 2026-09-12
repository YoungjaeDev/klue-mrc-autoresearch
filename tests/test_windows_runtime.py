import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from autoresearch_lab import run as control
from autoresearch_lab.windows_runtime import (
    build_child_environment,
    prepare_manifest,
    probe_python,
)


class WindowsRuntimeManifestTests(unittest.TestCase):
    def _fixture(self, root):
        project = root / "project"
        runtime = project / "runtime/windows_store"
        runtime.mkdir(parents=True)
        for name in ("sitecustomize.py", "launch.py", "dispatch.py"):
            (runtime / name).write_text(name, encoding="utf-8")
        for relative in ("autoresearch_lab/run.py", "autoresearch_lab/bootstrap.py",
                         "autoresearch_lab/windows_runtime.py"):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
        fixed = project / "fixed.json"
        fixed.write_text("fixed", encoding="utf-8")
        source = project / "data/manifest.json"
        source.parent.mkdir(parents=True)
        import hashlib
        source.write_text(json.dumps({"frozen_files": {
            str(fixed.resolve()): hashlib.sha256(fixed.read_bytes()).hexdigest(),
        }}), encoding="utf-8")
        launcher = root / "venv/Scripts/python.exe"
        launcher.parent.mkdir(parents=True)
        launcher.write_bytes(b"launcher")
        actual = root / "python/python.exe"
        dll = root / "python/python311.dll"
        actual.parent.mkdir(parents=True)
        actual.write_bytes(b"python")
        dll.write_bytes(b"dll")
        (root / "venv/pyvenv.cfg").write_text("home = elsewhere", encoding="utf-8")
        overlay = root / "overlay"
        overlay.mkdir()
        profile = {"actual_python": str(actual), "sys_executable": str(launcher),
                   "prefix": str(root / "venv"), "python_dll": str(dll),
                   "python_version": [3, 11, 9]}
        return project, runtime, source, launcher, overlay, profile

    def test_manifest_binds_direct_entry_venv_prefix_dll_and_frozen_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, runtime, source, launcher, overlay, profile = self._fixture(root)
            output = root / "runtime-manifest.json"
            record = prepare_manifest(launcher, overlay, source, output,
                                      project_root=project, runtime_dir=runtime,
                                      profile=profile)
            self.assertEqual(record["actual_python"], str(Path(profile["actual_python"]).resolve()))
            self.assertEqual(record["venv_prefix"], str(launcher.parent.parent.resolve()))
            self.assertEqual(record["python_dll"], str(Path(profile["python_dll"]).resolve()))
            self.assertEqual(record["status"], "cpu_prepared_gpu_unverified")
            self.assertIn(str(source.resolve()), record["frozen_files"])
            self.assertIn(str((runtime / "sitecustomize.py").resolve()), record["frozen_files"])
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                prepare_manifest(launcher, overlay, source, output,
                                 project_root=project, runtime_dir=runtime,
                                 profile=profile)

    def test_manifest_rejects_probe_from_another_prefix_or_dll_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, runtime, source, launcher, overlay, profile = self._fixture(root)
            wrong_prefix = root / "wrong-prefix"
            wrong_prefix.mkdir()
            bad_prefix = {**profile, "prefix": str(wrong_prefix)}
            with self.assertRaisesRegex(ValueError, "prefix"):
                prepare_manifest(launcher, overlay, source, root / "bad-prefix.json",
                                 project_root=project, runtime_dir=runtime,
                                 profile=bad_prefix)
            other = root / "other/python311.dll"
            other.parent.mkdir(parents=True)
            other.write_bytes(b"dll")
            bad_dll = {**profile, "python_dll": str(other)}
            with self.assertRaisesRegex(ValueError, "DLL"):
                prepare_manifest(launcher, overlay, source, root / "bad-dll.json",
                                 project_root=project, runtime_dir=runtime,
                                 profile=bad_dll)

    def test_child_environment_is_scoped_and_keeps_runtime_first(self):
        contract = {"actual_python": "C:/Python/python.exe", "venv_launcher": "C:/venv/Scripts/python.exe",
                    "python_dll": "C:/Python/python311.dll", "python_dll_sha256": "a" * 64,
                    "python_version": [3, 11, 9], "runtime_dir": "C:/repo/runtime/windows_store"}
        env = build_child_environment(contract, Path("C:/overlay"), Path("C:/repo"),
                                      {"PATH": "kept", "PYTHONPATH": "old", "__PYVENV_LAUNCHER__": "bad"})
        self.assertEqual(env["AUTORESEARCH_DIRECT_PYTHON"], contract["actual_python"])
        self.assertEqual(env["__PYVENV_LAUNCHER__"], contract["venv_launcher"])
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], contract["runtime_dir"])
        self.assertNotIn("old", env["PYTHONPATH"])
        self.assertEqual(env["PATH"], "kept")


@unittest.skipUnless(os.name == "nt", "Windows Job Object contract")
class WindowsDirectProcessTests(unittest.TestCase):
    def _open_and_check_job(self, pid, job):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                          ctypes.POINTER(wintypes.BOOL)]
        kernel.IsProcessInJob.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000 | 0x001000, False, pid)  # synchronize + query
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        member = wintypes.BOOL()
        if not kernel.IsProcessInJob(handle, job, ctypes.byref(member)):
            kernel.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        return kernel, handle, bool(member.value)

    def _assert_timeout_owns_grandchild(self, launcher):
        launcher = Path(launcher).resolve()
        profile = probe_python(launcher)
        project = Path(__file__).resolve().parents[1]
        runtime = project / "runtime/windows_store"
        contract = {**profile, "venv_launcher": str(launcher), "runtime_dir": str(runtime),
                    "python_dll_sha256": __import__("hashlib").sha256(
                        Path(profile["python_dll"]).read_bytes()).hexdigest()}
        env = build_child_environment(contract, project, project, os.environ)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "ready.json"
            marker = root / "escaped.txt"
            child = ("import ctypes,json,os,pathlib,sys,time;from ctypes import wintypes;"
                     "kernel=ctypes.WinDLL('kernel32',use_last_error=True);"
                     "kernel.GetCurrentProcess.restype=ctypes.c_void_p;"
                     "kernel.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(wintypes.BOOL)];"
                     "kernel.IsProcessInJob.restype=wintypes.BOOL;inside=wintypes.BOOL();"
                     "ok=kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside));"
                     "pathlib.Path(%r).write_text(json.dumps({'pid':os.getpid(),'prefix':sys.prefix,'version':list(sys.version_info[:3]),'job_member':bool(ok and inside.value)}));"
                     "time.sleep(30);pathlib.Path(%r).write_text('escaped')") % (str(ready), str(marker))
            parent = ("import pathlib,subprocess,sys,time;"
                      "p=subprocess.Popen([sys.executable,'-B','-c',%r]);"
                      "deadline=time.time()+10;ready=pathlib.Path(%r);"
                      "\nwhile not ready.exists() and p.poll() is None and time.time()<deadline: time.sleep(.01)"
                      "\nif not ready.exists(): raise RuntimeError('grandchild did not become ready')"
                      "\ntime.sleep(30)") % (child, str(ready))
            observed, owned_ready, finished = {}, threading.Event(), threading.Event()

            class ObservedOwnedProcess(control.OwnedProcess):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    observed["owned"] = self
                    owned_ready.set()

            def supervise():
                try:
                    observed["result"] = control.run_command(
                        [profile["actual_python"], "-B", "-c", parent],
                        root / "run", root / "state", 2, env=env, cwd=project)
                except BaseException as error:
                    observed["error"] = error
                finally:
                    finished.set()

            with mock.patch.object(control, "OwnedProcess", ObservedOwnedProcess):
                worker = threading.Thread(target=supervise)
                worker.start()
                owned_started = owned_ready.wait(1)
                deadline = time.monotonic() + 1.5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                ready_recorded = ready.is_file()
                record = json.loads(ready.read_text(encoding="utf-8")) if ready_recorded else None
                kernel = child_handle = exact_member = query_error = None
                if owned_started and ready_recorded:
                    try:
                        kernel, child_handle, exact_member = self._open_and_check_job(
                            record["pid"], observed["owned"].job)
                    except BaseException as error:
                        query_error = error
                supervisor_finished = finished.wait(5)
                worker.join()
            child_exit = kernel.WaitForSingleObject(child_handle, 1000) if child_handle else None
            if child_handle:
                kernel.CloseHandle(child_handle)
            self.assertTrue(owned_started, "supervised direct process did not start")
            self.assertTrue(ready_recorded, "grandchild did not record readiness before timeout")
            if query_error is not None:
                raise query_error
            if "error" in observed:
                raise observed["error"]
            result = observed["result"]
            self.assertEqual(Path(record["prefix"]).resolve(), Path(profile["prefix"]).resolve())
            self.assertEqual(record["version"], profile["python_version"])
            self.assertTrue(record["job_member"])
            self.assertTrue(exact_member, "grandchild is outside the supervisor Job Object")
            self.assertTrue(supervisor_finished, "timeout supervisor did not return")
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(child_exit, 0, "grandchild remains alive after timeout")
            self.assertFalse(marker.exists())

    def test_direct_runtime_keeps_grandchild_in_owned_job(self):
        self._assert_timeout_owns_grandchild(sys.executable)

    def test_installed_studio_store_runtime_keeps_grandchild_in_owned_job(self):
        configured = os.environ.get("AUTORESEARCH_TEST_STUDIO_PYTHON")
        launcher = Path(configured) if configured else (
            Path.home() / ".unsloth/studio/unsloth_studio/Scripts/python.exe")
        if not launcher.is_file():
            self.skipTest("No installed Studio Python was selected")
        self._assert_timeout_owns_grandchild(launcher)


if __name__ == "__main__":
    unittest.main()
