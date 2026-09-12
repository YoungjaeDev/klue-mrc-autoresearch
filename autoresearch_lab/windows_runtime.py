"""Prepare a CPU-verified manifest for a direct Windows venv runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime/windows_store"
RUNTIME_FILES = (
    "runtime/windows_store/sitecustomize.py",
    "runtime/windows_store/launch.py",
    "runtime/windows_store/dispatch.py",
    "autoresearch_lab/windows_runtime.py",
    "autoresearch_lab/bootstrap.py",
    "autoresearch_lab/run.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_python(launcher: Path) -> dict:
    """Ask a venv launcher which executable, prefix and Python DLL it loaded."""
    if os.name != "nt":
        raise OSError("The direct Studio runtime is Windows-only")
    launcher = Path(launcher).resolve(strict=True)
    code = r'''import ctypes,json,sys
k=ctypes.WinDLL("kernel32",use_last_error=True)
k.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint]
k.GetModuleFileNameW.restype=ctypes.c_uint
k.GetModuleHandleW.argtypes=[ctypes.c_wchar_p]
k.GetModuleHandleW.restype=ctypes.c_void_p
def name(handle):
    buffer=ctypes.create_unicode_buffer(32768)
    if not k.GetModuleFileNameW(handle,buffer,len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value
dll="python%d%d.dll" % sys.version_info[:2]
print(json.dumps({"actual_python":name(None),"sys_executable":sys.executable,
    "prefix":sys.prefix,"python_dll":name(k.GetModuleHandleW(dll)),
    "python_version":list(sys.version_info[:3])}))'''
    completed = subprocess.run(
        [str(launcher), "-I", "-B", "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    try:
        profile = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Studio Python probe did not return the runtime contract") from error
    return profile


def _validated_profile(launcher: Path, profile: dict) -> dict:
    required = {"actual_python", "sys_executable", "prefix", "python_dll", "python_version"}
    if set(profile) != required:
        raise ValueError("Runtime probe fields differ from the expected contract")
    launcher = launcher.resolve(strict=True)
    try:
        actual = Path(profile["actual_python"]).resolve(strict=True)
        reported_launcher = Path(profile["sys_executable"]).resolve(strict=True)
        prefix = Path(profile["prefix"]).resolve(strict=True)
        dll = Path(profile["python_dll"]).resolve(strict=True)
    except OSError as error:
        raise ValueError("Runtime probe path does not exist") from error
    version = list(profile["python_version"])
    if reported_launcher != launcher:
        raise ValueError("Runtime probe used a different venv launcher")
    if prefix != launcher.parent.parent.resolve():
        raise ValueError("Runtime probe prefix does not match the venv launcher")
    if actual.parent != dll.parent:
        raise ValueError("Runtime entry and Python DLL are from different directories")
    if len(version) != 3 or not all(isinstance(part, int) for part in version):
        raise ValueError("Runtime probe returned an invalid Python version")
    expected_dll = f"python{version[0]}{version[1]}.dll"
    if dll.name.lower() != expected_dll:
        raise ValueError("Runtime probe loaded an unexpected Python DLL")
    return {"actual_python": str(actual), "venv_launcher": str(launcher),
            "venv_prefix": str(prefix), "python_dll": str(dll), "python_version": version}


def prepare_manifest(studio_python: Path, overlay: Path, data_manifest: Path, output: Path,
                     *, project_root: Path = ROOT, runtime_dir: Path = RUNTIME_DIR,
                     profile: dict | None = None) -> dict:
    """Write a new machine-local manifest; no package installation or model import."""
    project_root = Path(project_root).resolve(strict=True)
    runtime_dir = Path(runtime_dir).resolve(strict=True)
    launcher = Path(studio_python).resolve(strict=True)
    overlay = Path(overlay).resolve(strict=True)
    data_manifest = Path(data_manifest).resolve(strict=True)
    pyvenv = launcher.parent.parent / "pyvenv.cfg"
    pyvenv.resolve(strict=True)
    runtime = _validated_profile(launcher, profile if profile is not None else probe_python(launcher))

    source_raw = data_manifest.read_bytes()
    source = json.loads(source_raw)
    frozen = dict(source.get("frozen_files", {}))
    for filename, expected in frozen.items():
        path = Path(filename)
        if not path.is_absolute() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Data manifest frozen input differs: {path.name}")
    owned = [project_root / relative for relative in RUNTIME_FILES]
    if runtime_dir != (project_root / "runtime/windows_store").resolve():
        owned[:3] = [runtime_dir / name for name in ("sitecustomize.py", "launch.py", "dispatch.py")]
    owned.extend([data_manifest, pyvenv, Path(runtime["actual_python"]), Path(runtime["python_dll"])])
    for path in owned:
        path.resolve(strict=True)
        frozen[str(path.resolve())] = digest(path)

    record = {
        "schema_version": 1,
        "runtime_kind": "windows_direct_venv",
        "status": "cpu_prepared_gpu_unverified",
        "project_root": str(project_root),
        "runtime_dir": str(runtime_dir),
        **runtime,
        "actual_python_sha256": digest(Path(runtime["actual_python"])),
        "python_dll_sha256": digest(Path(runtime["python_dll"])),
        "overlay": str(overlay),
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": hashlib.sha256(source_raw).hexdigest(),
        "frozen_files": frozen,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record


def build_child_environment(contract: dict, overlay: Path, project_root: Path,
                            base_environment: dict) -> dict:
    """Install direct-runtime settings in the owned child environment only."""
    environment = dict(base_environment)
    environment.update({
        "AUTORESEARCH_DIRECT_PYTHON": contract["actual_python"],
        "AUTORESEARCH_VENV_LAUNCHER": contract["venv_launcher"],
        "AUTORESEARCH_PYTHON_DLL": contract["python_dll"],
        "AUTORESEARCH_PYTHON_DLL_SHA256": contract["python_dll_sha256"],
        "AUTORESEARCH_PYTHON_VERSION": ".".join(map(str, contract["python_version"])),
        "__PYVENV_LAUNCHER__": contract["venv_launcher"],
        "PYTHONPATH": os.pathsep.join((contract["runtime_dir"], str(Path(overlay).resolve()),
                                        str(Path(project_root).resolve()))),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Probe a venv and write a new runtime manifest")
    prepare.add_argument("--studio-python", type=Path, required=True)
    prepare.add_argument("--overlay", type=Path, required=True)
    prepare.add_argument("--data-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = prepare_manifest(args.studio_python, args.overlay, args.data_manifest, args.output)
    print(json.dumps({"runtime_manifest": str(args.output.resolve()),
                      "runtime_manifest_sha256": digest(args.output.resolve()),
                      "status": record["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
