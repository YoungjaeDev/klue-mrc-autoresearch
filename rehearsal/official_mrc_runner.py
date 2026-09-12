"""Hugging Face model/LoRA -> generated answers -> official KLUE-MRC scores."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

from . import official_mrc as task
from . import official_mrc_splits as splits


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def local_identity(directory: Path) -> dict:
    files = sorted({*directory.glob("*.json"), *directory.glob("*.jinja"), *directory.glob("*.safetensors")})
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"No safetensors weights in {directory}; export HF model or LoRA, not GGUF")
    return {"path": str(directory.resolve()), "files": {path.name: task.digest(path) for path in files}}


def select_rows(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    if limit <= 0 or limit > len(rows):
        raise ValueError(f"limit must be between 1 and {len(rows)}")
    return rows[:limit]


def evaluation_rows(args):
    rows = task.load_data(args.data_dir)
    path = getattr(args, "split_manifest", None)
    metadata = {"split": "public-dev", "split_manifest_sha256": None}
    if path:
        rows, metadata = splits.load_selection(path, rows, getattr(args, "split", None),
            getattr(args, "split_manifest_sha256", None), getattr(args, "final_release", None))
    elif getattr(args, "split", None) or getattr(args, "split_manifest_sha256", None):
        raise ValueError("A named search/final split requires its frozen manifest")
    metadata["diagnostic_prefix"] = args.limit is not None
    return select_rows(rows, args.limit), metadata


def model_kind(config) -> str:
    # Studio uses the full Qwen3.5 conditional model, even for text-only LoRA.
    return "image-text-to-text" if config.model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_vl"} else "causal-lm"


def load_backend(args, rows):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; evaluation stopped without automatic device fallback")
    dtype_name = args.dtype if args.dtype != "auto" else ("bfloat16" if args.device == "cuda" else "float32")
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(3407)
    requested = args.model
    adapter_identity = None
    if args.adapter:
        adapter = Path(args.adapter).resolve()
        adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = adapter_config["base_model_name_or_path"]
        adapter_revision = adapter_config.get("revision")
        if adapter_revision and args.revision and adapter_revision != args.revision:
            raise ValueError("Requested base revision conflicts with adapter_config.json")
        if adapter_revision and args.revision is None:
            args.revision = adapter_revision
        if requested is None:
            requested = base_name
        if requested != base_name and not (Path(requested).exists() and Path(base_name).exists() and Path(requested).resolve() == Path(base_name).resolve()):
            raise ValueError(f"Adapter base is {base_name!r}, but --model is {requested!r}; use the actual training base")
        adapter_identity = local_identity(adapter)
    if not requested:
        raise ValueError("Specify --model or --adapter")
    if Path(requested).is_dir() and (Path(requested) / "adapter_config.json").exists():
        raise ValueError("This is an adapter directory. Pass it as --adapter, not --model")
    source = str(Path(requested).resolve()) if Path(requested).is_dir() else requested
    config = AutoConfig.from_pretrained(source, revision=args.revision, trust_remote_code=False)
    if Path(source).is_dir():
        identity = {"local_model": local_identity(Path(source))}
        resolved_revision = None
    else:
        resolved_revision = getattr(config, "_commit_hash", None)
        if not resolved_revision:
            raise ValueError("Could not resolve the model to an immutable Hub revision")
        identity = {"model_id": source, "revision": resolved_revision}
    quantization = config.to_dict().get("quantization_config")
    quantization_kwargs = {}
    if quantization and quantization.get("quant_method") != "bitsandbytes":
        raise ValueError("This runner supports full precision or bitsandbytes checkpoints; GGUF/AWQ/GPTQ need a separate backend")
    if quantization or getattr(args, "load_in_4bit", False):
        if args.device != "cuda":
            raise ValueError("The configured quantized evaluation requires CUDA")
        if not quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            quantization_kwargs["quantization_config"] = bnb_config
            quantization = bnb_config.to_dict()

    tokenizer_source = args.tokenizer or source
    tokenizer_revision = resolved_revision if not args.tokenizer else args.tokenizer_revision
    if args.tokenizer and not Path(args.tokenizer).is_dir() and not tokenizer_revision:
        raise ValueError("A separate Hub tokenizer requires --tokenizer-revision")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=tokenizer_revision, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("The tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(task.messages_for(row), tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=True)
        tokens = encoded["input_ids"]
        if not isinstance(tokens, list) or not tokens or any(not isinstance(token, int) for token in tokens):
            raise ValueError("Expected a flat token ID list for one conversation")
        if len(tokens) + args.max_new_tokens > args.max_length:
            raise ValueError(f"{row['id']}: input + output budget exceeds --max-length; no truncation applied")
        prompt_ids.append(tokens)
    kind = model_kind(config)
    conditions = {
        "base_model": identity,
        "model_class": kind, "dtype": dtype_name, "device": args.device,
        "quantization": quantization or "none", "attention": "sdpa", "seed": 3407,
        "do_sample": False, "enable_thinking": False,
        "max_new_tokens": args.max_new_tokens, "max_length": args.max_length,
        "prompt_token_ids_sha256": fingerprint(prompt_ids),
        "tokenizer_vocab_sha256": fingerprint(tokenizer.get_vocab()),
        "chat_template_sha256": fingerprint(tokenizer.chat_template),
        "question_ids_sha256": fingerprint([row["id"] for row in rows]),
        "dataset_sha256": task.DEV_SHA256, "metric_sha256": task.METRIC_SHA256,
        "runner_sha256_lf": task.digest(Path(__file__), normalize_newlines=True),
        "task_sha256_lf": task.digest(Path(task.__file__), normalize_newlines=True),
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "safetensors", "numpy")},
    }
    if quantization:
        conditions["versions"]["bitsandbytes"] = version("bitsandbytes")
    print(f"Loading {source}; {len(rows)} questions; {dtype_name}; {args.device}", flush=True)
    model_class = AutoModelForImageTextToText if kind == "image-text-to-text" else AutoModelForCausalLM
    model = model_class.from_pretrained(source, revision=resolved_revision, config=config,
        dtype=dtype, device_map={"": args.device}, attn_implementation="sdpa", trust_remote_code=False,
        use_safetensors=True, **quantization_kwargs)
    if args.adapter:
        from peft import PeftModel
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*missing adapter keys.*", category=UserWarning)
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    model.requires_grad_(False)
    conditions["generation_config"] = model.generation_config.to_dict()
    return torch, tokenizer, model, prompt_ids, conditions, adapter_identity


def run(args) -> dict:
    rows, selection = evaluation_rows(args)
    if args.max_new_tokens <= 0 or args.max_length <= args.max_new_tokens:
        raise ValueError("Invalid context/output token budgets")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    stage = "load_model"
    try:
        torch, tokenizer, model, prompt_ids, conditions, adapter = load_backend(args, rows)
        conditions["selection"] = selection
        task.write_json(args.output / "run.json", {
            "started_at": started_at, "conditions": conditions, "adapter": adapter,
            "rows": len(rows), "full_dev": len(rows) == 5841,
            "dataset": "KLUE-MRC v1.1 official public dev",
        })
        stage = "generate"
        predictions = []
        with (args.output / "raw_predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for number, (row, ids) in enumerate(zip(rows, prompt_ids), 1):
                inputs = torch.tensor([ids], dtype=torch.long, device=args.device)
                with torch.inference_mode():
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=False, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id, use_cache=True)
                new_ids = output[0, len(ids):].tolist()
                eos = model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                prediction = {
                    "id": row["id"], "prediction": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "generated_tokens": len(new_ids),
                    "hit_token_limit": len(new_ids) >= args.max_new_tokens and (not new_ids or new_ids[-1] not in eos_ids),
                }
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                stream.flush()
                predictions.append(prediction)
                if number == 1 or number % 25 == 0 or number == len(rows):
                    print(f"Generated {number}/{len(rows)}", flush=True)
        stage = "official_score"
        scores = task.score(rows, predictions)
        scores["selection"] = selection
        task.write_json(args.output / "official_predictions.json", {
            row["id"]: task.official_prediction(row["prediction"]) for row in predictions
        })
        scores["predictions_sha256"] = task.digest(args.output / "raw_predictions.jsonl")
        scores["run_sha256"] = task.digest(args.output / "run.json")
        task.write_json(args.output / "scores.json", scores)
        task.write_json(args.output / "status.json", {
            "status": "completed", "duration_seconds": time.perf_counter() - started,
            "scores_sha256": task.digest(args.output / "scores.json"),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        })
        return scores
    except BaseException as error:
        task.write_json(args.output / "status.json", {
            "status": "failed", "stage": stage, "error_type": type(error).__name__,
            "automatic_retry": False,
        })
        raise


def read_completed(directory):
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError(f"Evaluation did not complete: {directory}")
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    scores = json.loads((directory / "scores.json").read_text(encoding="utf-8"))
    if status.get("scores_sha256") != task.digest(directory / "scores.json"):
        raise ValueError(f"Score evidence changed or was not sealed: {directory}")
    if task.digest(directory / "run.json") != scores["run_sha256"] or task.digest(directory / "raw_predictions.jsonl") != scores["predictions_sha256"]:
        raise ValueError(f"Run evidence changed: {directory}")
    return record, scores


def compare(base: Path, sft: Path) -> dict:
    records = [read_completed(directory) for directory in (base, sft)]
    (before, base_scores), (after, sft_scores) = records
    if before["adapter"] is not None or after["adapter"] is None:
        raise ValueError("Comparison requires an unadapted base run and a LoRA run")
    if before["conditions"] != after["conditions"]:
        differing = sorted(key for key in set(before["conditions"]) | set(after["conditions"]) if before["conditions"].get(key) != after["conditions"].get(key))
        raise ValueError(f"Comparison conditions differ: {', '.join(differing)}")
    return {
        "evaluation": base_scores["evaluation"], "conditions_verified_equal": True,
        "full_dev": base_scores["full_dev"] and sft_scores["full_dev"],
        "base": base_scores, "sft": sft_scores,
        "delta_percentage_points": {name: sft_scores["overall"][name] - base_scores["overall"][name] for name in ("exact_match", "rouge_w")},
    }


def compare_arms(arms):
    if "base" not in arms or len(arms) < 2:
        raise ValueError("Comparison needs base and at least one adapted arm")
    records = {name: read_completed(path) for name, path in arms.items()}
    baseline, base_scores = records["base"]
    if baseline["adapter"] is not None:
        raise ValueError("Base arm must not have an adapter")
    result = {}
    for name, (record, scores) in records.items():
        if name != "base" and record["adapter"] is None:
            raise ValueError("Every trained arm requires a LoRA adapter")
        if record["conditions"] != baseline["conditions"]:
            raise ValueError(f"Comparison conditions differ for {name}")
        result[name] = {"scores": scores, "adapter": record["adapter"],
            "delta_vs_base_pp": {metric: scores["overall"][metric] - base_scores["overall"][metric]
                                 for metric in ("exact_match", "rouge_w")}}
    return {"conditions_verified_equal": True, "conditions": baseline["conditions"], "arms": result,
            "training_equivalence_asserted": False,
            "interpretation": "Same evaluation conditions; historical Studio and new SDK reference training provenance must be reported separately."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download/check official dev data; no GPU")
    prepare.add_argument("--data-dir", type=Path, default=task.DEFAULT_DATA)
    prepare.add_argument("--source", type=Path)
    evaluate = sub.add_parser("run", help="Generate answers with a model or LoRA and call the official scorer")
    evaluate.add_argument("--model", help="Hugging Face ID or local merged HF model directory")
    evaluate.add_argument("--revision")
    evaluate.add_argument("--adapter", help="Local Studio-exported LoRA directory")
    evaluate.add_argument("--tokenizer")
    evaluate.add_argument("--tokenizer-revision")
    evaluate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    evaluate.add_argument("--load-in-4bit", action="store_true", help="Use NF4 for both base and LoRA evaluations; requires CUDA and bitsandbytes")
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--max-length", type=int, default=2048)
    evaluate.add_argument("--data-dir", type=Path, default=task.DEFAULT_DATA)
    evaluate.add_argument("--limit", type=int, help="Diagnostic prefix only; omitted = all 5841 questions")
    evaluate.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score", help="Call the official scorer on saved raw predictions; no GPU")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--data-dir", type=Path, default=task.DEFAULT_DATA)
    score.add_argument("--limit", type=int)
    score.add_argument("--output", type=Path, required=True)
    for command in (evaluate, score):
        command.add_argument("--split-manifest", type=Path)
        command.add_argument("--split-manifest-sha256")
        command.add_argument("--split", choices=("search", "final"))
        command.add_argument("--final-release", type=Path, help="Controller receipt after selection completes")
    comparison = sub.add_parser("compare", help="Compare verified, completed base/LoRA runs; no GPU")
    comparison.add_argument("--base", type=Path, required=True)
    comparison.add_argument("--sft", type=Path)
    comparison.add_argument("--studio", type=Path)
    comparison.add_argument("--reference", type=Path)
    comparison.add_argument("--selected", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = task.prepare(args.data_dir, args.source)
    elif args.command == "run":
        report = run(args)
    else:
        if args.command == "score":
            rows, selection = evaluation_rows(args)
            report = task.score(rows, task.read_jsonl(args.predictions))
            report["selection"] = selection
        else:
            extras = {name: getattr(args, name) for name in ("studio", "reference", "selected") if getattr(args, name)}
            if args.sft and extras:
                raise ValueError("Use --sft pair comparison or named --studio/--reference/--selected arms")
            report = compare(args.base, args.sft) if args.sft else compare_arms({"base": args.base, **extras})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        task.write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
