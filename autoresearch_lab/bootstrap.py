"""Use an existing Studio interpreter/overlay without installing or repairing it."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


DEFAULT_OVERLAY = Path.home() / ".unsloth/studio/.venv_t5_530"
EXPECTED = {"transformers": "5.3.0", "unsloth": "2026.9.4", "unsloth_zoo": "2026.9.3",
    "torch": "2.11.0+cu130", "trl": "0.23.1", "peft": "0.18.1",
    "bitsandbytes": "0.50.2", "triton-windows": "3.6.0.post26"}


def activate(overlay: Path, cache_dir: Path) -> dict:
    """Path/metadata operations only. No Studio activation helper or ML import."""
    heavy = {"torch", "transformers", "unsloth", "unsloth_zoo", "trl", "peft"}
    if heavy.intersection(sys.modules):
        raise RuntimeError("bootstrap requires a fresh interpreter before all ML imports")
    overlay = overlay.resolve(strict=True)
    if not (overlay / "transformers/__init__.py").is_file():
        raise ValueError("Existing Transformers overlay is missing; automatic repair is disabled")
    sys.path.insert(0, str(overlay))
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise ValueError(f"Installed runtime differs from pinned bootstrap contract: {versions}")
    origins = {name: importlib.util.find_spec(name).origin for name in heavy}
    if Path(origins["transformers"]).resolve().parent != overlay / "transformers":
        raise RuntimeError("Transformers resolved outside the selected overlay")
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=False)
    cache_paths = {"UNSLOTH_COMPILE_LOCATION": cache_dir / "unsloth",
        "TORCHINDUCTOR_CACHE_DIR": cache_dir / "inductor", "TRITON_CACHE_DIR": cache_dir / "triton",
        "HF_DATASETS_CACHE": cache_dir / "datasets", "PYTHONPYCACHEPREFIX": cache_dir / "pycache"}
    for key, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    # Keep multiprocessing imports on the same overlay; never reuse Studio's compiled modules.
    os.environ["PYTHONPATH"] = os.pathsep.join((str(overlay), str(cache_paths["UNSLOTH_COMPILE_LOCATION"]),
        str(Path(__file__).resolve().parents[1])))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    record = {"python": sys.version, "executable": sys.executable, "overlay": str(overlay),
        "versions": versions, "origins": origins, "cache_dir": str(cache_dir),
        "heavy_modules_loaded": sorted(heavy.intersection(sys.modules))}
    (cache_dir / "bootstrap.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--module", default="autoresearch_lab.train")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    record = activate(args.overlay, args.cache_dir)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.check_only:
        return 0
    rest = args.args[1:] if args.args[:1] == ["--"] else args.args
    sys.argv = [args.module, *rest]
    runpy.run_module(args.module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
