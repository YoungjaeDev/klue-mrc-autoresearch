"""
QLoRA fine-tuning of Qwen3.5-4B on the KLUE-MRC train pool. This is the file the agent edits.

Usage:
    uv run --env-file .env train.py --out runs/<name>                # one experiment (128 steps)
    uv run --env-file .env train.py --out runs/diag --max-steps 1    # diagnosis only

Saves the LoRA adapter to --out. Logs only train/loss and eval/loss to W&B;
peak VRAM and training time go to the run summary.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import json
import time

import unsloth  # noqa: F401  (must be imported before transformers)
from unsloth import FastLanguageModel
import torch
import wandb
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from prepare import MAX_SEQ_LEN, MODEL_NAME, SEED, eval_loss_rows, load_train_pool

# ---------------------------------------------------------------------------
# Fixed budget (do not change): 1,024 rows x 1 epoch = 128 optimizer steps
# ---------------------------------------------------------------------------

MAX_STEPS = 128
BATCH_SIZE = 4
GRAD_ACCUM = 2
EVAL_EVERY = 32
EVAL_BATCH_SIZE = 1  # eval/loss is monitoring only; small batch keeps eval memory low

# ---------------------------------------------------------------------------
# Hyperparameters (the loop may change these)
# ---------------------------------------------------------------------------

LEARNING_RATE = 2.5e-4
LR_SCHEDULER = "linear"
WARMUP_STEPS = 3
OPTIM = "adamw_8bit"
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 1.0
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"]
_MLP = ["gate_proj", "up_proj", "down_proj"]
LORA_RANK_PATTERN = {**{m: 16 for m in _ATTN}, **{m: 64 for m in _MLP}}  # capacity moved from attention to MLP
LORA_ALPHA_PATTERN = dict(LORA_RANK_PATTERN)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]

# ---------------------------------------------------------------------------
# Data: prompt tokens are masked, loss only on the assistant answer + <|im_end|>
# ---------------------------------------------------------------------------

def encode(row, tokenizer):
    messages = row["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    prompt = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(messages[2]["content"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    assert len(input_ids) <= MAX_SEQ_LEN, f"{row['id']}: {len(input_ids)} tokens"
    return {"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def make_collator(pad_id):
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}
    return collate


class WandbLoss(TrainerCallback):
    """Log exactly train/loss (every step) and eval/loss (every EVAL_EVERY steps)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            wandb.log({"train/loss": logs["loss"]}, step=state.global_step)
        if logs and "eval_loss" in logs:
            wandb.log({"eval/loss": logs["eval_loss"]}, step=state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        # Unsloth sizes fused-CE chunks from driver-level free memory (mem_get_info), which
        # excludes PyTorch's cached blocks; return eval's cache so the next step sees it as free.
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="adapter output directory")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="diagnosis only")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    hparams = {k: v for k, v in globals().items() if k.isupper()}
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "klue-mrc-autoresearch"),
        name=os.path.basename(os.path.normpath(args.out)),
        config={**hparams, "max_steps": args.max_steps},
    )
    with open(os.path.join(args.out, "wandb_run_id.txt"), "w") as f:
        f.write(run.id)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)  # processor -> text tokenizer
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        rank_pattern=LORA_RANK_PATTERN,
        alpha_pattern=LORA_ALPHA_PATTERN,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LEN,
    )

    train_ds = Dataset.from_list([encode(r, tokenizer) for r in load_train_pool()])
    eval_ds = Dataset.from_list([encode(r, tokenizer) for r in eval_loss_rows()])

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
        callbacks=[WandbLoss()],
        args=TrainingArguments(
            output_dir=os.path.join(args.out, "trainer"),
            max_steps=args.max_steps,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER,
            warmup_steps=WARMUP_STEPS,
            optim=OPTIM,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            eval_strategy="steps",
            eval_steps=EVAL_EVERY,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=SEED,
            data_seed=SEED,
            bf16=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
        ),
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    summary = {
        "train_seconds": round(train_seconds, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "num_steps": trainer.state.global_step,
        "train_loss": round(result.training_loss, 4),
        "load_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model.config, "quantization_config", None)),
    }
    run.summary["peak_vram_gb"] = summary["peak_vram_gb"]
    run.summary["train_seconds"] = summary["train_seconds"]
    with open(os.path.join(args.out, "train_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    wandb.finish()

    print("---")
    for k, v in summary.items():
        print(f"{k + ':':<16}{v}")
