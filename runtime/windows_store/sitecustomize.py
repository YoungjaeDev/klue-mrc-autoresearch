"""Validate and retain one direct Windows venv runtime for child processes."""
import ctypes
import hashlib
import os
from pathlib import Path
import sys


def _module_filename(kernel, handle):
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return Path(buffer.value).resolve(strict=True)


def configure():
    direct = Path(os.environ["AUTORESEARCH_DIRECT_PYTHON"]).resolve(strict=True)
    launcher = Path(os.environ["AUTORESEARCH_VENV_LAUNCHER"]).resolve(strict=True)
    expected_dll = Path(os.environ["AUTORESEARCH_PYTHON_DLL"]).resolve(strict=True)
    expected_digest = os.environ["AUTORESEARCH_PYTHON_DLL_SHA256"]
    expected_version = tuple(int(part) for part in os.environ["AUTORESEARCH_PYTHON_VERSION"].split("."))
    runtime = Path(__file__).resolve().parent
    prefix = launcher.parent.parent
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    actual = _module_filename(kernel, None)
    dll = _module_filename(kernel, kernel.GetModuleHandleW(expected_dll.name))
    if actual != direct or Path(sys.prefix).resolve() != prefix.resolve():
        raise RuntimeError("Direct interpreter or venv prefix mismatch")
    if dll != expected_dll or dll.parent != direct.parent:
        raise RuntimeError("Loaded Python DLL does not match the direct interpreter")
    if hashlib.sha256(dll.read_bytes()).hexdigest() != expected_digest:
        raise RuntimeError("Loaded Python DLL checksum mismatch")
    if sys.version_info[:3] != expected_version:
        raise RuntimeError("Python version differs from the runtime manifest")
    sys.executable = str(direct)
    os.environ["__PYVENV_LAUNCHER__"] = str(launcher)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(runtime), *[path for path in paths if path and Path(path).resolve() != runtime]])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


try:
    configure()
except BaseException:
    # sitecustomize exceptions normally print and continue. This contract must fail closed.
    os.write(2, b"Direct Windows runtime validation failed; refusing execution.\n")
    os._exit(121)
