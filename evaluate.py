"""KLUE-MRC evaluation: official dev data, frozen search/final split, model generation, official EM/ROUGE-W.

Frozen during research. Only the unmodified official scorer in klue_scorer/ computes metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import urllib.request
import warnings

from klue_scorer import klue_baseline_utils as official


ROOT = Path(__file__).resolve().parent
DATA_REVISION = "3efd98708a40ff49251fddde35453f8fbb11f536"
METRIC_REVISION = "8a03c9447e4c225e806877a84242aea11258c790"
DEV_SHA256 = "b31bb073e47cdeb19f2f1bae03d916ecabf945140334c75842d6994f700d4f47"
METRIC_SHA256 = "6ec7e78e0e9687f1601058d23f3e270c281aaa3f3cf9db686705e0ebb723221c"
DEV_URL = f"https://raw.githubusercontent.com/KLUE-benchmark/KLUE/{DATA_REVISION}/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_dev.json"
DEFAULT_DATA = ROOT / "data/generated/klue-mrc-official-v1"
NO_ANSWER = "지문에서 답을 찾을 수 없습니다."
SYSTEM = (
    "지문을 읽고 질문에 답하세요. 답이 있으면 지문에서 답에 해당하는 짧은 표현만 쓰세요. "
    f'지문에 답이 없으면 정확히 "{NO_ANSWER}"라고 답하세요. 설명을 덧붙이지 마세요.'
)


def digest(path: Path, *, normalize_newlines: bool = False) -> str:
    if normalize_newlines:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index_rows(rows)
    return rows


def index_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each prediction must have a nonempty string id")
        if row["id"] in result:
            raise ValueError(f"Duplicate prediction id: {row['id']}")
        result[row["id"]] = row
    if not result:
        raise ValueError("No prediction rows")
    return result


def flatten(data: dict) -> list[dict]:
    rows = []
    for entry in data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                if bool(qa["is_impossible"]) != (qa["question_type"] == 3):
                    raise ValueError(f"Invalid question type: {qa['guid']}")
                if bool(qa["answers"]) == bool(qa["is_impossible"]):
                    raise ValueError(f"Official answer presence does not match type: {qa['guid']}")
                rows.append({
                    "id": qa["guid"], "question_type": qa["question_type"],
                    "is_impossible": qa["is_impossible"],
                    "context": paragraph["context"], "question": qa["question"],
                    "answers": [answer["text"] for answer in qa["answers"]],
                })
    index_rows(rows)
    return rows


def messages_for(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"지문:\n{row['context']}\n\n질문: {row['question']}"},
    ]


def verify_scorer() -> None:
    if digest(Path(official.__file__), normalize_newlines=True) != METRIC_SHA256:
        raise ValueError("Vendored official scoring source changed")


def load_data(directory: Path = DEFAULT_DATA) -> list[dict]:
    verify_scorer()
    path = directory / "official-dev.json"
    if not path.exists():
        raise FileNotFoundError("Run the prepare command before evaluation")
    if digest(path) != DEV_SHA256:
        raise ValueError("Official dev checksum mismatch")
    rows = flatten(json.loads(path.read_text(encoding="utf-8")))
    if Counter(row["question_type"] for row in rows) != {1: 2437, 2: 1571, 3: 1833}:
        raise ValueError("Unexpected official dev counts")
    return rows


def prepare_dev(directory: Path = DEFAULT_DATA, source: Path | None = None) -> dict:
    verify_scorer()
    if source is not None:
        payload = source.read_bytes()
    else:
        with urllib.request.urlopen(DEV_URL, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != DEV_SHA256:
        raise ValueError("Downloaded official dev checksum mismatch")
    rows = flatten(json.loads(payload))
    if len(rows) != 5841:
        raise ValueError("Expected all 5841 official dev questions")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "official-dev.json").write_bytes(payload)
    with (directory / "prompts.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps({"id": row["id"], "messages": messages_for(row)}, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "KLUE-MRC v1.1", "split": "public dev", "rows": len(rows),
        "question_types": dict(Counter(row["question_type"] for row in rows)),
        "dataset_revision": DATA_REVISION, "dataset_sha256": DEV_SHA256,
        "dataset_url": DEV_URL, "license": "CC-BY-SA-4.0",
        "attribution": "Park et al., KLUE: Korean Language Understanding Evaluation (2021)",
        "metric_repository": "KLUE-benchmark/KLUE-baseline", "metric_revision": METRIC_REVISION,
        "metric_source_sha256_lf": METRIC_SHA256,
        "metrics": ["exact_match", "rouge_w"],
        "output_mapping": f'Exactly {NO_ANSWER!r} after strip -> empty string; other outputs unchanged',
        "system_prompt": SYSTEM, "generation_executed": False,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def official_prediction(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("prediction must be a string")
    return "" if raw.strip() == NO_ANSWER else raw


def score(rows: list[dict], predictions: list[dict]) -> dict:
    verify_scorer()
    indexed = index_rows(predictions)
    expected = {row["id"] for row in rows}
    if set(indexed) != expected:
        missing, extra = expected - set(indexed), set(indexed) - expected
        raise ValueError(f"Prediction ID mismatch: {len(missing)} missing, {len(extra)} extra")
    converted = {key: official_prediction(row["prediction"]) for key, row in indexed.items()}
    labels = [{"qid": row["id"], "qtype": row["question_type"], "ground_truth": row["answers"]} for row in rows]

    def evaluate(items):
        if not items:
            return {"count": 0, "exact_match": None, "rouge_w": None}
        result = official.evaluate_for_klue_mrc(items, converted)
        return {"count": len(items), "exact_match": float(result["exact_match"]) * 100,
                "rouge_w": float(result["rouge"]) * 100}

    return {
        "evaluation": "Official KLUE-MRC v1.1 public dev; generative model output adapter",
        "official_test_submission": False, "score_scale": "0-100",
        "dataset_revision": DATA_REVISION, "dataset_sha256": DEV_SHA256,
        "metric_revision": METRIC_REVISION, "metric_source_sha256_lf": METRIC_SHA256,
        "full_dev": len(rows) == 5841,
        "overall": evaluate(labels),
        "by_question_type": {str(kind): evaluate([label for label in labels if label["qtype"] == kind]) for kind in (1, 2, 3)},
        "mapped_no_answer_responses": sum(row["prediction"].strip() == NO_ANSWER for row in predictions),
        "empty_raw_responses": sum(not row["prediction"].strip() for row in predictions),
        "truncated_responses": sum(bool(row.get("hit_token_limit", False)) for row in predictions),
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def row_fingerprint(rows):
    # Metadata is separately frozen in preparation. These fields also exist in official JSON.
    fields = ("id", "context", "question", "question_type", "answers")
    return fingerprint([{key: row[key] for key in fields} for row in sorted(rows, key=lambda r: r["id"])])


def validate_manifest(manifest, rows):
    indexed = index_rows(rows)
    if manifest.get("schema_version") != 1 or manifest["validation_rows_sha256"] != row_fingerprint(rows):
        raise ValueError("Split schema or source row fingerprint mismatch")
    if manifest["dataset_sha256"] != DEV_SHA256 or manifest["validation_rows"] != len(rows):
        raise ValueError("Split dataset identity mismatch")
    if set(manifest["groups"]) != set(indexed) or set(manifest["splits"]) != {"search", "final"}:
        raise ValueError("Split group/partition IDs mismatch")
    seen, used_groups = set(), set()
    for name in ("search", "final"):
        part = manifest["splits"][name]
        ids = part["ids"]
        if not ids or len(set(ids)) != len(ids) or seen & set(ids) or not set(ids) <= set(indexed):
            raise ValueError("Missing, duplicate or overlapping split IDs")
        expected_types = {str(k): v for k, v in sorted(Counter(indexed[i]["question_type"] for i in ids).items())}
        if part["ids_sha256"] != fingerprint(ids) or part["rows"] != len(ids) or part["question_types"] != expected_types:
            raise ValueError("Split counts or ordered ID checksum mismatch")
        own_groups = {manifest["groups"][i] for i in ids}
        if used_groups & own_groups:
            raise ValueError("Connected group crosses search/final")
        seen.update(ids)
        used_groups.update(own_groups)
    if seen != set(indexed):
        raise ValueError("Split does not partition every validation ID")


def load_selection(path: Path, rows, split, expected_sha256, final_release=None):
    if not expected_sha256 or digest(path) != expected_sha256:
        raise ValueError("Frozen split manifest checksum mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, rows)
    if split not in ("search", "final"):
        raise ValueError("Choose search or final")
    if split == "final":
        if not final_release:
            raise ValueError("Final is sealed until the controller records completed selection")
        receipt = json.loads(Path(final_release).read_text(encoding="utf-8"))
        if receipt.get("selection_complete") is not True or receipt.get("split_manifest_sha256") != expected_sha256:
            raise ValueError("Final release does not match completed selection and split")
    indexed = index_rows(rows)
    return ([indexed[i] for i in manifest["splits"][split]["ids"]],
            {"split": split, "split_manifest_sha256": expected_sha256, "prior_exposure": manifest["prior_exposure"]})


def local_identity(directory: Path) -> dict:
    files = sorted({*directory.glob("*.json"), *directory.glob("*.jinja"), *directory.glob("*.safetensors")})
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"No safetensors weights in {directory}; export HF model or LoRA, not GGUF")
    return {"path": str(directory.resolve()), "files": {path.name: digest(path) for path in files}}


def select_rows(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    if limit <= 0 or limit > len(rows):
        raise ValueError(f"limit must be between 1 and {len(rows)}")
    return rows[:limit]


def evaluation_rows(args):
    rows = load_data(args.data_dir)
    path = getattr(args, "split_manifest", None)
    metadata = {"split": "public-dev", "split_manifest_sha256": None}
    if path:
        rows, metadata = load_selection(path, rows, getattr(args, "split", None),
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
        encoded = tokenizer.apply_chat_template(messages_for(row), tokenize=True,
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
        "dataset_sha256": DEV_SHA256, "metric_sha256": METRIC_SHA256,
        "evaluate_sha256_lf": digest(Path(__file__), normalize_newlines=True),
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
        write_json(args.output / "run.json", {
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
        scores = score(rows, predictions)
        scores["selection"] = selection
        write_json(args.output / "official_predictions.json", {
            row["id"]: official_prediction(row["prediction"]) for row in predictions
        })
        scores["predictions_sha256"] = digest(args.output / "raw_predictions.jsonl")
        scores["run_sha256"] = digest(args.output / "run.json")
        write_json(args.output / "scores.json", scores)
        write_json(args.output / "status.json", {
            "status": "completed", "duration_seconds": time.perf_counter() - started,
            "scores_sha256": digest(args.output / "scores.json"),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        })
        return scores
    except BaseException as error:
        write_json(args.output / "status.json", {
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
    if status.get("scores_sha256") != digest(directory / "scores.json"):
        raise ValueError(f"Score evidence changed or was not sealed: {directory}")
    if digest(directory / "run.json") != scores["run_sha256"] or digest(directory / "raw_predictions.jsonl") != scores["predictions_sha256"]:
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
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--limit", type=int, help="Diagnostic prefix only; omitted = all 5841 questions")
    evaluate.add_argument("--output", type=Path, required=True)
    score_command = sub.add_parser("score", help="Call the official scorer on saved raw predictions; no GPU")
    score_command.add_argument("--predictions", type=Path, required=True)
    score_command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    score_command.add_argument("--limit", type=int)
    score_command.add_argument("--output", type=Path, required=True)
    for command in (evaluate, score_command):
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
    if args.command == "run":
        report = run(args)
    else:
        if args.command == "score":
            rows, selection = evaluation_rows(args)
            report = score(rows, read_jsonl(args.predictions))
            report["selection"] = selection
        else:
            extras = {name: getattr(args, name) for name in ("studio", "reference", "selected") if getattr(args, name)}
            if args.sft and extras:
                raise ValueError("Use --sft pair comparison or named --studio/--reference/--selected arms")
            report = compare(args.base, args.sft) if args.sft else compare_arms({"base": args.base, **extras})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
