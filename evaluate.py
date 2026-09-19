"""
Fixed evaluation of a saved LoRA adapter on the search split. Do not modify.

Usage:
    uv run --env-file .env evaluate.py --adapter runs/<name> --split search [--limit N]

Loads the base model in NF4 in a fresh process, attaches the adapter with
PeftModel.from_pretrained, asserts every saved LoRA tensor landed in the model,
generates greedily (thinking off, 128 new tokens), and scores with the official
KLUE-baseline evaluate_for_klue_mrc. Writes <adapter>/search_predictions.jsonl
and <adapter>/search_metrics.json, and logs search/em and search/rouge_w to the
adapter's W&B run.
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import json
import time

import unsloth  # noqa: F401  (must be imported before transformers)
from unsloth import FastLanguageModel
import torch
import wandb
from peft import PeftModel
from safetensors.torch import load_file

from klue_mrc_utils import evaluate_for_klue_mrc
from prepare import MAX_NEW_TOKENS, MAX_SEQ_LEN, MODEL_NAME, load_search, load_search_labels, postprocess

EVAL_BATCH_SIZE = 8


def load_model(adapter_dir):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model = PeftModel.from_pretrained(model, adapter_dir)

    # load_adapter silently drops mismatched keys; check every saved tensor is in the model.
    saved = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    state = model.state_dict()
    lora_in_model = [k for k in state if "lora_" in k]
    assert len(saved) == len(lora_in_model), f"saved {len(saved)} LoRA tensors, model has {len(lora_in_model)}"
    for key, tensor in saved.items():
        name = key[: -len(".weight")] + ".default.weight"
        assert name in state, f"adapter tensor not in model: {key}"
        assert torch.equal(state[name].detach().cpu().to(tensor.dtype), tensor), f"adapter tensor differs: {key}"
    print(f"adapter check: all {len(saved)} LoRA tensors loaded")

    FastLanguageModel.for_inference(model)
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, rows):
    tokenizer.padding_side = "left"
    prompts = {
        r["id"]: tokenizer.apply_chat_template(
            r["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for r in rows
    }
    order = sorted(prompts, key=lambda i: len(prompts[i]))
    outputs = {}
    for start in range(0, len(order), EVAL_BATCH_SIZE):
        ids = order[start : start + EVAL_BATCH_SIZE]
        batch = tokenizer([prompts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = batch.to(model.device)
        out = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        outputs.update(zip(ids, texts))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", required=True, choices=["search"])
    parser.add_argument("--limit", type=int, default=None, help="diagnosis only")
    args = parser.parse_args()

    t0 = time.time()
    rows = load_search()[: args.limit]
    labels = [l for l in load_search_labels() if l["qid"] in {r["id"] for r in rows}]
    assert len(labels) == len(rows)

    model, tokenizer = load_model(args.adapter)
    raw = generate(model, tokenizer, rows)
    preds = {qid: postprocess(text) for qid, text in raw.items()}

    overall = evaluate_for_klue_mrc(labels, preds)
    metrics = {
        "n": len(rows),
        "search_em": round(100 * overall["exact_match"], 2),
        "search_rouge_w": round(100 * overall["rouge"], 2),
    }
    for t in (1, 2, 3):
        typed = [l for l in labels if l["qtype"] == t]
        metrics[f"em_type{t}"] = round(100 * evaluate_for_klue_mrc(typed, preds)["exact_match"], 2) if typed else 0.0
    metrics["empty_count"] = sum(1 for p in preds.values() if p == "")
    metrics["eval_seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(args.adapter, "search_predictions.jsonl"), "w", encoding="utf-8") as f:
        for l in labels:
            q = l["qid"]
            em = evaluate_for_klue_mrc([l], {q: preds[q]})["exact_match"]
            f.write(json.dumps({"id": q, "qtype": l["qtype"], "raw": raw[q], "pred": preds[q],
                                "ground_truth": l["ground_truth"], "em": em}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.adapter, "search_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    run_id_path = os.path.join(args.adapter, "wandb_run_id.txt")
    run_id = open(run_id_path).read().strip() if os.path.exists(run_id_path) else None
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "klue-mrc-autoresearch"), id=run_id, resume="allow")
    run.log({"search/em": metrics["search_em"], "search/rouge_w": metrics["search_rouge_w"]})
    wandb.finish()

    print("---")
    for k, v in metrics.items():
        print(f"{k + ':':<16}{v}")
