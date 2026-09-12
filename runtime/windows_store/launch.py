"""Activate the pinned overlay, then run one module through the direct runtime."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    runtime = Path(__file__).resolve().parent
    import sitecustomize
    if Path(sitecustomize.__file__).resolve().parent != runtime:
        raise RuntimeError("Direct runtime sitecustomize is missing")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from autoresearch_lab.bootstrap import activate
    record = activate(args.overlay, args.cache_dir)
    # activate replaces PYTHONPATH. Retain this guard for subprocess descendants.
    sitecustomize.configure()
    record["launch_runtime"] = str(runtime)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.check_only:
        return
    rest = args.args[1:] if args.args[:1] == ["--"] else args.args
    sys.argv = [args.module, *rest]
    runpy.run_module(args.module, run_name="__main__")


if __name__ == "__main__":
    main()
