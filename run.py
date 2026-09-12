"""Serial subprocess supervisor; no candidate grid, model import or package install."""
from __future__ import annotations

import argparse
import codecs
import contextlib
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time


GPU_FAILURE = re.compile(r"CUDA (?:error|out of memory)|out of memory|device.side assert|illegal memory access|"
    r"(?:Triton|kernel|cuDNN|cuBLAS).*(?:Error|error|failed|failure)|"
    r"(?:Error|error).*\b(?:Triton|kernel|cuDNN|cuBLAS)\b|No CUDA GPUs", re.IGNORECASE)


def write_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def verify_files(hashes):
    for filename, digest in hashes.items():
        path = Path(filename)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Frozen file must exist at an absolute path")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Frozen input changed: {path.name}")


class GpuLock:
    """OS lock held for the whole owned train/eval process tree."""
    def __init__(self, state_dir):
        self.path = Path(state_dir) / "gpu.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise RuntimeError("Another process holds the campaign GPU lock") from None
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


class OwnedProcess:
    """Windows Job Object owns descendants; child is assigned before it starts.

    On POSIX the newly created session is the process group we own. Never enumerate
    or terminate arbitrary GPU/user processes.
    """
    def __init__(self, argv, env, cwd):
        self.job = None
        self.process = None
        if os.name == "nt":
            self._create_windows_job()
            try:
                self.process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, cwd=cwd, creationflags=0x00000004 | 0x08000000)  # suspended, hidden
                if not self.kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
                # Popen closes the initial thread handle. Resume only this suspended process.
                ntdll = self.ctypes.WinDLL("ntdll")
                resume = ntdll.NtResumeProcess
                resume.argtypes = [self.ctypes.c_void_p]
                resume.restype = self.ctypes.c_long
                if resume(int(self.process._handle)) != 0:
                    raise RuntimeError("Could not resume supervised child")
            except BaseException:
                if self.process is not None:
                    self.process.kill()
                    self.process.wait(timeout=5)
                    self.process.stdout.close()
                self.close()
                raise
        else:
            self.process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, cwd=cwd, start_new_session=True)

    def _create_windows_job(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes

        class BasicLimit(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        self.job = kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def terminate(self):
        if self.process is None:
            return
        if self.job is not None:
            self.kernel.TerminateJobObject(self.job, 1)
        elif os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait(timeout=10)

    def close(self):
        if self.job is not None:
            self.kernel.CloseHandle(self.job)
            self.job = None


def run_command(argv, run_dir, state_dir, timeout_seconds, *, gpu=False, deadline=None,
                frozen_files=None, env=None, cwd=None, secrets=None):
    if not argv or timeout_seconds <= 0:
        raise ValueError("A command and positive timeout are required")
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Campaign deadline reached")
    run_dir, state_dir = Path(run_dir), Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    hashes = dict(frozen_files or {})
    verify_files(hashes)
    stop = state_dir / "STOP_GPU.json"
    lock = GpuLock(state_dir) if gpu else contextlib.nullcontext()
    with lock:
        if gpu and stop.exists():
            raise RuntimeError("GPU campaign stopped; inspect STOP_GPU.json before any new run")
        run_dir.mkdir(parents=True, exist_ok=False)
        started_at = time.time()
        started = time.monotonic()
        expires = started + min(timeout_seconds, max(0, deadline - started_at) if deadline is not None else timeout_seconds)
        child_env = dict(os.environ if env is None else env)
        if gpu:
            child_env["CUDA_VISIBLE_DEVICES"] = "0"
        child_env["PYTHONUNBUFFERED"] = "1"
        status, fatal = "completed", None
        owned = None
        stream_thread = None
        credential_names = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
        credentials = [value for key, value in child_env.items()
            if value and (key in credential_names or key.endswith("_API_KEY") or key.endswith("_ACCESS_TOKEN"))]
        secret_values = sorted({*credentials, *(value for value in secrets or [] if value)}, key=len, reverse=True)

        def redact(text):
            for value in secret_values:
                text = text.replace(value, "[REDACTED]")
            return text

        def latch(reason):
            if not stop.exists():
                try:
                    write_json(stop, {"reason": reason, "run_dir": str(run_dir.resolve()), "at": time.time()})
                except FileExistsError:
                    pass

        try:
            owned = OwnedProcess(list(argv), child_env, cwd)
            process = owned.process
            chunks = queue.Queue()

            def read_output():
                try:
                    while True:
                        chunk = process.stdout.read1(4096)
                        if not chunk:
                            break
                        chunks.put(chunk)
                finally:
                    chunks.put(None)

            stream_thread = threading.Thread(target=read_output, daemon=True)
            stream_thread.start()
            ended = False
            tail = ""
            pending = ""
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            with (run_dir / "process.log").open("x", encoding="utf-8") as logfile:
                while not ended or process.poll() is None:
                    if time.monotonic() >= expires and status == "completed":
                        status = "timeout"
                        owned.terminate()
                        if gpu:
                            latch("timeout")
                    if process.poll() is not None:
                        # Also close descendants that inherited pipes after the parent exited.
                        owned.terminate()
                    try:
                        chunk = chunks.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if chunk is None:
                        ended = True
                        pending += decoder.decode(b"", final=True)
                        if pending:
                            logfile.write(redact(pending))
                            logfile.flush()
                        continue
                    decoded = decoder.decode(chunk)
                    tail = (tail + decoded)[-8192:]
                    # Retain enough pending text to redact a secret split across read chunks.
                    pending += decoded
                    retain = max([len(value) for value in secret_values] + [0])
                    cut = pending.rfind("\n", 0, max(0, len(pending) - retain)) + 1
                    if cut:
                        logfile.write(redact(pending[:cut]))
                        logfile.flush()
                        pending = pending[cut:]
                    if gpu and GPU_FAILURE.search(tail) and status == "completed":
                        status, fatal = "gpu_error", "GPU/CUDA/kernel failure detected in process output"
                        latch("gpu_error")
                        owned.terminate()
                if pending and not ended:
                    logfile.write(redact(pending))
            returncode = process.wait()
            if status == "completed" and returncode:
                if gpu and (returncode < 0 or returncode >= 0x80000000):
                    status = "native_crash"
                    latch("native_crash")
                else:
                    status = "failed"
            try:
                verify_files(hashes)
            except ValueError:
                status = "frozen_input_changed"
                if gpu:
                    latch(status)
            result = {"status": status, "returncode": returncode, "gpu": gpu,
                "started_at": started_at, "wall_seconds": time.monotonic() - started,
                "argv": [redact(str(arg)) for arg in argv], "fatal": fatal,
                "frozen_files": hashes}
            write_json(run_dir / "process.json", result)
            return result
        except BaseException as error:
            if gpu:
                latch("supervisor_exception")
            if not (run_dir / "process.json").exists():
                write_json(run_dir / "process.json", {"status": "supervisor_exception",
                    "error_type": type(error).__name__, "wall_seconds": time.monotonic() - started})
            raise
        finally:
            if owned is not None:
                owned.terminate()
                owned.close()
                if stream_thread:
                    stream_thread.join(timeout=5)
                if owned.process.stdout:
                    owned.process.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--deadline", help="Absolute ISO timestamp with timezone")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    deadline = None
    if args.deadline:
        parsed = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parser.error("--deadline must have a timezone")
        deadline = parsed.timestamp()
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8")) if args.frozen_manifest else {}
    result = run_command(command, args.run_dir, args.state_dir, args.timeout_seconds,
        gpu=args.gpu, deadline=deadline, frozen_files=frozen.get("frozen_files", frozen))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
