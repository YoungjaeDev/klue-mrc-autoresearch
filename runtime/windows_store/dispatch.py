"""Run the direct Windows runtime under the repository CPU supervisor."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--deadline")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forbidden = {"__PYVENV_LAUNCHER__", "AUTORESEARCH_DIRECT_PYTHON",
                 "AUTORESEARCH_VENV_LAUNCHER", "AUTORESEARCH_PYTHON_DLL",
                 "AUTORESEARCH_PYTHON_DLL_SHA256", "AUTORESEARCH_PYTHON_VERSION"}
    if forbidden.intersection(os.environ):
        raise RuntimeError("Direct launch settings must not be set in the CPU supervisor")
    raw = args.runtime_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.runtime_manifest_sha256.lower():
        raise ValueError("Runtime manifest checksum mismatch")
    contract = json.loads(raw)
    if contract.get("schema_version") != 1 or contract.get("runtime_kind") != "windows_direct_venv":
        raise ValueError("Unsupported runtime manifest")
    project = Path.cwd().resolve()
    if Path(contract["project_root"]).resolve(strict=True) != project:
        raise ValueError("Run from the project root recorded in the runtime manifest")
    runtime = Path(contract["runtime_dir"]).resolve(strict=True)
    if Path(__file__).resolve().parent != runtime:
        raise ValueError("Dispatch source differs from the runtime manifest")
    overlay = Path(contract["overlay"]).resolve(strict=True)
    sys.path.insert(0, str(project))
    from autoresearch_lab.run import run_command, verify_files
    from autoresearch_lab.windows_runtime import build_child_environment
    verify_files(contract["frozen_files"])
    child_env = build_child_environment(contract, overlay, project, os.environ)
    deadline = None
    if args.deadline:
        timestamp = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Deadline requires a timezone")
        deadline = timestamp.timestamp()
    command = [contract["actual_python"], "-X", "utf8", "-B", str(runtime / "launch.py"),
               "--overlay", str(overlay), "--cache-dir", str(args.cache_dir), "--module", args.module]
    if args.check_only:
        command.append("--check-only")
    rest = args.args[1:] if args.args[:1] == ["--"] else args.args
    command.extend(["--", *rest])
    result = run_command(command, args.run_dir, args.state_dir, args.timeout_seconds,
                         gpu=args.gpu, deadline=deadline, frozen_files=contract["frozen_files"],
                         env=child_env, cwd=project)
    provenance = {"runtime_manifest": str(args.runtime_manifest.resolve()),
                  "runtime_manifest_sha256": args.runtime_manifest_sha256.lower(),
                  "actual_python": contract["actual_python"],
                  "venv_launcher": contract["venv_launcher"],
                  "venv_prefix": contract["venv_prefix"],
                  "python_dll": contract["python_dll"],
                  "python_dll_sha256": contract["python_dll_sha256"],
                  "runtime_dir": str(runtime), "overlay": str(overlay),
                  "supervisor_python": sys.executable, "project_root": str(project)}
    with (args.run_dir / "launch-provenance.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "returncode": result["returncode"],
                      "gpu": result["gpu"], "wall_seconds": result["wall_seconds"],
                      "run_dir": str(args.run_dir.resolve())}, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
