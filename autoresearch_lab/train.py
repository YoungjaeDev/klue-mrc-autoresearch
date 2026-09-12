"""Agent-editable SDK SFT recipe. Data, evaluation and supervisor live elsewhere.

No ML imports until run_training(). Every candidate starts from the pinned base.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_RECIPE = {"max_steps": 30, "max_seq_length": 4096, "batch_size": 2,
    "gradient_accumulation_steps": 4, "learning_rate": 2e-4, "lr_scheduler_type": "linear",
    "warmup_steps": 3, "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.0,
    "seed": 3407, "data_seed": 3407, "optim": "adamw_8bit", "weight_decay": .001,
    "packing": False, "load_in_4bit": True, "response_only": True}
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_recipe(recipe: dict, diagnostic_steps: int | None) -> dict:
    fixed = {"max_steps": 30, "max_seq_length": 4096, "batch_size": 2,
        "gradient_accumulation_steps": 4, "seed": 3407, "data_seed": 3407,
        "packing": False, "load_in_4bit": True, "response_only": True}
    if any(recipe.get(key) != value for key, value in fixed.items()):
        raise ValueError("Recipe changed the fixed training/data budget")
    if diagnostic_steps not in (None, 1):
        raise ValueError("Only explicit diagnostic_steps=1 is supported")
    result = dict(recipe)
    if diagnostic_steps is not None:
        result["max_steps"] = diagnostic_steps
    for field in ("learning_rate", "weight_decay", "lora_dropout"):
        if not isinstance(result[field], (int, float)) or not math.isfinite(result[field]) or result[field] < 0:
            raise ValueError(f"Invalid {field}")
    if result["learning_rate"] == 0 or not 1 <= result["lora_r"] <= 32 or result["lora_alpha"] <= 0:
        raise ValueError("Invalid learning rate or LoRA size (maximum rank 32)")
    return result


def load_contract(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["model_id"], manifest["model_revision"], manifest["seed"], manifest["train_rows"]) != (
        MODEL_ID, MODEL_REVISION, 3407, 1024):
        raise ValueError("Manifest does not describe the approved reference")
    path = Path(manifest["train_path"])
    if not path.is_absolute() or sha256(path) != manifest["train_sha256"]:
        raise ValueError("Training data hash/path mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1024 or len({row["id"] for row in rows}) != 1024:
        raise ValueError("Expected 1024 unique training rows")
    for row in rows:
        messages = row.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant" or sum(m.get("role") == "assistant" for m in messages) != 1:
            raise ValueError("Each training row must end in exactly one assistant answer")
        if any(m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("Invalid text conversation")
    return manifest, rows


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def prepare_texts(tokenizer, rows):
    """CPU tokenization: never rely on apply_chat_template's tokenized return type."""
    texts, prefix_lengths, full_lengths = [], [], []
    for row in rows:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        prefix_text = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        prefix = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        if not isinstance(tokens, list) or not isinstance(prefix, list) or any(not isinstance(v, int) for v in tokens + prefix):
            raise TypeError("Expected flat tokenizer input_ids")
        if not text.startswith(prefix_text) or tokens[:len(prefix)] != prefix:
            raise ValueError("Serialized system/user prefix does not align with training tokens")
        if len(tokens) > 4096:
            raise ValueError("Training row exceeds frozen sequence length; no truncation allowed")
        texts.append({"text": text})
        prefix_lengths.append(len(prefix))
        full_lengths.append(len(tokens))
    return {"texts": texts, "prefix_lengths": prefix_lengths,
        "rows": len(rows), "max_tokens": max(full_lengths), "all_prefixes_aligned": True,
        "chat_template_sha256": fingerprint(tokenizer.chat_template),
        "tokenizer_vocab_sha256": fingerprint(tokenizer.get_vocab())}


def run_preflight(manifest_path: Path, output: Path):
    """A disposable CPU tokenizer process, before Unsloth/CUDA/model loading."""
    manifest, rows = load_contract(manifest_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        trust_remote_code=False, local_files_only=True)
    prepared = prepare_texts(tokenizer, rows)
    prepared.update(manifest_sha256=sha256(manifest_path), train_sha256=manifest["train_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "prepared.json", prepared)
    return {key: value for key, value in prepared.items() if key not in {"texts", "prefix_lengths"}}


def validate_trainable_parameters(parameters, run_kind):
    if not parameters or any(".language_model.layers." not in item["name"] or ".lora_" not in item["name"]
        or "vision" in item["name"] or "visual" in item["name"] for item in parameters):
        raise ValueError("Only language-layer LoRA parameters may be trainable")
    families = sorted({item["name"].split(".lora_")[0].rsplit(".", 1)[-1] for item in parameters})
    record = {"tensor_count": len(parameters), "total_parameters": sum(item["numel"] for item in parameters),
        "families": families, "parameters": parameters}
    if run_kind == "reference" and (record["tensor_count"] != 256 or record["total_parameters"] != 21233664
        or families != sorted(TARGET_MODULES)):
        raise ValueError("Reference LoRA targets differ from the measured Studio adapter (256 tensors / 21233664 parameters / 7 projection families)")
    return record


class EventWriter:
    """Local scalar telemetry only; never imports a tracking SDK."""
    SCALARS = {"loss", "learning_rate", "grad_norm", "epoch", "step_seconds",
        "gpu_allocated_bytes", "gpu_reserved_bytes", "peak_gpu_allocated_bytes"}

    def __init__(self, path: Path):
        self.stream = path.open("x", encoding="utf-8", buffering=1)
        self.started = time.monotonic()
        self.index = 0

    def emit(self, event_type: str, step: int, values: dict | None = None):
        record = {"event_index": self.index, "event": event_type, "optimizer_step": int(step),
            "timestamp": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - self.started}
        for key, value in (values or {}).items():
            if key in self.SCALARS and isinstance(value, (int, float)) and math.isfinite(value):
                record[key] = value
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        self.index += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


def make_callback(events, torch, TrainerCallback):
    class ScalarCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.step_started = time.monotonic()

        def on_log(self, args, state, control, logs=None, **kwargs):
            # logging_steps=1 gives one loss/LR/grad-norm record per optimizer update.
            if logs is None or "loss" not in logs:
                return
            if not math.isfinite(float(logs["loss"])):
                raise FloatingPointError("Training loss is not finite")
            values = {**logs, "step_seconds": time.monotonic() - getattr(self, "step_started", events.started),
                "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_reserved_bytes": torch.cuda.memory_reserved(),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}
            events.emit("step", state.global_step, values)
    return ScalarCallback()


def run_training(manifest_path: Path, output: Path, diagnostic_steps=None, run_kind="reference") -> dict:
    if run_kind not in {"reference", "candidate"} or (diagnostic_steps and run_kind != "reference"):
        raise ValueError("Diagnostic smoke must use the reference recipe")
    manifest, rows = load_contract(manifest_path)
    recipe = validate_recipe(dict(DEFAULT_RECIPE), diagnostic_steps)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stage = "cpu_tokenizer_preflight"
    contract = {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
        "train_sha256": manifest["train_sha256"], "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "recipe": recipe, "run_kind": run_kind, "diagnostic": diagnostic_steps is not None, "train_code_sha256": sha256(Path(__file__))}
    write_json(output / "run.json", contract)
    try:
        preflight_output = output / "tokenization"
        # The parent has imported no ML modules. A fresh child can import the CPU
        # tokenizer without breaking Unsloth's required import order in this process.
        preflight_env = dict(os.environ)
        preflight_env["CUDA_VISIBLE_DEVICES"] = ""
        preflight_env["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run([sys.executable, "-B", "-m", "autoresearch_lab.train", "--manifest", str(manifest_path.resolve()),
            "--output", str(preflight_output.resolve()), "--preflight-only"], check=True, env=preflight_env)
        prepared = json.loads((preflight_output / "prepared.json").read_text(encoding="utf-8"))
        if prepared["manifest_sha256"] != contract["manifest_sha256"] or prepared["train_sha256"] != contract["train_sha256"]:
            raise ValueError("CPU preflight contract changed")
        stage = "import"
        # Unsloth must patch before transformers/TRL imports.
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA single visible GPU required; no automatic fallback")
        torch.cuda.reset_peak_memory_stats()
        stage = "load_model"
        model, processor = FastLanguageModel.from_pretrained(model_name=MODEL_ID, revision=MODEL_REVISION,
            use_exact_model_name=True, max_seq_length=4096, dtype=torch.bfloat16,
            load_in_4bit=True, device_map={"": 0}, trust_remote_code=False)
        observed_revision = getattr(model.config, "_commit_hash", None)
        if observed_revision is not None and observed_revision != MODEL_REVISION:
            raise ValueError("Loaded model revision differs from the pinned base")
        model = FastLanguageModel.get_peft_model(model, r=recipe["lora_r"],
            target_modules=list(TARGET_MODULES),
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            lora_alpha=recipe["lora_alpha"], lora_dropout=recipe["lora_dropout"], bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407, use_rslora=False)
        tokenizer = getattr(processor, "tokenizer", processor)
        trainable = [{"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
            for name, parameter in model.named_parameters() if parameter.requires_grad]
        trainable_record = validate_trainable_parameters(trainable, run_kind)
        write_json(output / "trainable_parameters.json", trainable_record)
        stage = "prepare_training"
        if prepared["chat_template_sha256"] != fingerprint(tokenizer.chat_template) or prepared["tokenizer_vocab_sha256"] != fingerprint(tokenizer.get_vocab()):
            raise ValueError("Training tokenizer differs from CPU preflight")
        prefix_lengths = prepared["prefix_lengths"]
        dataset = Dataset.from_list(prepared["texts"])
        args = SFTConfig(output_dir=str(output / "checkpoints"), max_steps=recipe["max_steps"],
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            learning_rate=recipe["learning_rate"], lr_scheduler_type=recipe["lr_scheduler_type"],
            warmup_steps=recipe["warmup_steps"], weight_decay=recipe["weight_decay"], optim=recipe["optim"],
            max_length=4096, dataset_text_field="text", dataset_num_proc=1, packing=False,
            completion_only_loss=False, seed=3407, data_seed=3407,
            dataloader_num_workers=0, dataloader_drop_last=False, bf16=True, fp16=False,
            logging_steps=1, logging_first_step=True, report_to="none", save_strategy="no",
            eval_strategy="no", disable_tqdm=True, gradient_checkpointing=True)
        with EventWriter(output / "events.jsonl") as events:
            trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=args,
                callbacks=[make_callback(events, torch, TrainerCallback)])
            trainer = train_on_responses_only(trainer, tokenizer=tokenizer, num_proc=1)
            if len(trainer.train_dataset) != len(rows):
                raise ValueError("Response masking dropped training rows; stop and inspect")
            target_counts = []
            for index, sample in enumerate(trainer.train_dataset):
                labels = sample["labels"]
                if len(labels) < prefix_lengths[index] or any(value != -100 for value in labels[:prefix_lengths[index]]):
                    raise ValueError("System/user prefix has trainable labels")
                count = sum(value != -100 for value in labels)
                if count == 0:
                    raise ValueError("Empty assistant training target")
                target_counts.append(count)
            write_json(output / "label_evidence.json", {"rows": len(rows), "all_prefixes_masked": True,
                "min_target_tokens": min(target_counts), "max_target_tokens": max(target_counts),
                "target_tokens_total": sum(target_counts), "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()})
            stage = "train"
            events.emit("train_start", 0)
            train_started = time.monotonic()
            trainer.train()
            train_seconds = time.monotonic() - train_started
            if trainer.state.global_step != recipe["max_steps"]:
                raise RuntimeError("Training did not complete the requested optimizer steps")
            events.emit("train_end", trainer.state.global_step)
            stage = "save"
            adapter = output / "adapter"
            model.save_pretrained(str(adapter))
            processor.save_pretrained(str(adapter))
            write_json(adapter / "adapter_metadata.json", {"model_id": MODEL_ID,
                "model_revision": MODEL_REVISION, "resolved_model_revision": observed_revision,
                "train_sha256": manifest["train_sha256"], "diagnostic": diagnostic_steps is not None})
            artifacts = {str(path.relative_to(output)): sha256(path) for path in sorted(adapter.rglob("*")) if path.is_file()}
            if not any(name.endswith(".safetensors") for name in artifacts):
                raise RuntimeError("Adapter weights were not saved")
            write_json(output / "artifacts.json", artifacts)
            result = {"status": "completed", "optimizer_steps": trainer.state.global_step,
                "sample_presentations": trainer.state.global_step * 8, "train_seconds": train_seconds,
                "duration_seconds": time.monotonic() - started, "diagnostic": diagnostic_steps is not None,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "adapter_path": str(adapter.resolve()),
                "run_sha256": sha256(output / "run.json"), "events_sha256": sha256(output / "events.jsonl")}
        write_json(output / "status.json", result)
        return result
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "stage": stage,
            "error_type": type(error).__name__, "duration_seconds": time.monotonic() - started,
            "automatic_retry": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-steps", type=int, choices=[1])
    parser.add_argument("--preflight-only", action="store_true", help="CPU tokenizer only, using already cached files")
    parser.add_argument("--run-kind", choices=["reference", "candidate"], default="reference")
    args = parser.parse_args(argv)
    result = run_preflight(args.manifest, args.output) if args.preflight_only else run_training(args.manifest, args.output, args.diagnostic_steps, args.run_kind)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
