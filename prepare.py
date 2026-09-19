"""
One-time data preparation for KLUE-MRC autoresearch, plus fixed runtime helpers
imported by train.py and evaluate.py. Do not modify.

Usage:
    uv run prepare.py

Writes data/train_pool_ids.txt, data/search_ids.txt, data/final_ids.txt.
Raw files are cached in ~/.cache/klue-mrc-autoresearch/ and the HF cache.
"""

import collections
import json
import os
import random
import re
import unicodedata

import requests
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SEED = 3407
MODEL_NAME = "Qwen/Qwen3.5-4B"
DATASET = "YoungjaeDev/klue-mrc-messages"
DATASET_REVISION = "ae6c27baff54df9d0a63ed85451badd6aefc131c"
DEV_JSON_URL = (
    "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/"
    "3efd98708a40ff49251fddde35453f8fbb11f536/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_dev.json"
)
TRAIN_POOL_SIZE = 1024
SEARCH_FRACTION = 0.2
MAX_SEQ_LEN = 4096
EVAL_LOSS_ROWS = 128
MAX_NEW_TOKENS = 128
NO_ANSWER = "지문에서 답을 찾을 수 없습니다."

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "klue-mrc-autoresearch")
DEV_JSON_PATH = os.path.join(CACHE_DIR, "klue-mrc-v1.1_dev.json")

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split):
    """Rows of train.jsonl / validation.jsonl at the pinned revision."""
    path = hf_hub_download(DATASET, f"{split}.jsonl", repo_type="dataset", revision=DATASET_REVISION)
    return read_jsonl(path)


def load_dev_json():
    if not os.path.exists(DEV_JSON_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        r = requests.get(DEV_JSON_URL, timeout=120)
        r.raise_for_status()
        with open(DEV_JSON_PATH + ".tmp", "wb") as f:
            f.write(r.content)
        os.replace(DEV_JSON_PATH + ".tmp", DEV_JSON_PATH)
    with open(DEV_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def read_ids(name):
    with open(os.path.join(DATA_DIR, f"{name}_ids.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_ids(name, ids):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"{name}_ids.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(ids) + "\n")

# ---------------------------------------------------------------------------
# Runtime helpers (imported by train.py and evaluate.py)
# ---------------------------------------------------------------------------

def load_train_pool():
    """The 1,024 training rows, in train_pool_ids.txt order."""
    by_id = {r["id"]: r for r in load_split("train")}
    return [by_id[i] for i in read_ids("train_pool")]


def load_search():
    """Validation rows in the search split, in search_ids.txt order."""
    by_id = {r["id"]: r for r in load_split("validation")}
    return [by_id[i] for i in read_ids("search")]


def eval_loss_rows():
    """Fixed 128 search rows used for eval/loss during training."""
    return random.Random(SEED).sample(load_search(), EVAL_LOSS_ROWS)


def load_search_labels():
    """KLUE labels (qid, qtype, ground_truth) for the search split, from the official dev JSON."""
    from klue_mrc_utils import extract_labels_from_dataset_for_klue_mrc
    search = set(read_ids("search"))
    labels = extract_labels_from_dataset_for_klue_mrc(load_dev_json()["data"])
    return [l for l in labels if l["qid"] in search]


def postprocess(output):
    """Strip; the fixed no-answer sentence becomes the empty string."""
    output = output.strip()
    return "" if output == NO_ANSWER else output

# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def normalize_context(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def group_questions(dev):
    """Union questions that share a normalized context or the same (source, title)."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        parent[find(a)] = find(b)

    qtype = {}
    for doc in dev["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                q = ("q", qa["guid"])
                qtype[qa["guid"]] = qa["question_type"]
                parent.setdefault(q, q)
                union(q, ("ctx", normalize_context(para["context"])))
                if doc.get("title"):
                    union(q, ("st", doc.get("source", ""), doc["title"]))

    groups = collections.defaultdict(list)
    for guid in qtype:
        groups[find(("q", guid))].append(guid)
    return [sorted(g) for g in sorted(groups.values())], qtype


def split_search_final(dev):
    """Group-level assignment; greedily track 20% of each question_type in search."""
    groups, qtype = group_questions(dev)
    totals = collections.Counter(qtype.values())
    target = {t: SEARCH_FRACTION * n for t, n in totals.items()}
    random.Random(SEED).shuffle(groups)

    counts = collections.Counter()
    search, final = [], []
    for g in groups:
        gc = collections.Counter(qtype[q] for q in g)
        before = sum(abs(target[t] - counts[t]) for t in target)
        after = sum(abs(target[t] - counts[t] - gc[t]) for t in target)
        if after < before:
            search.extend(g)
            counts.update(gc)
        else:
            final.extend(g)
    return sorted(search), sorted(final), qtype, len(groups)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Dataset: {DATASET}@{DATASET_REVISION}")
    train = load_split("train")
    validation = load_split("validation")
    dev = load_dev_json()
    print(f"train rows: {len(train)}, validation rows: {len(validation)}")

    # ID check: validation.jsonl and official dev JSON must match exactly
    val_ids = [r["id"] for r in validation]
    dev_ids = [qa["guid"] for d in dev["data"] for p in d["paragraphs"] for qa in p["qas"]]
    assert len(val_ids) == len(set(val_ids)) == len(dev_ids) == len(set(dev_ids)), "duplicate or count mismatch"
    assert set(val_ids) == set(dev_ids), "validation.jsonl and dev JSON IDs differ"
    print(f"ID check: {len(val_ids)} validation IDs match the dev JSON")

    # Train pool: shuffle with seed, take first 1,024
    pool = [r["id"] for r in train]
    random.Random(SEED).shuffle(pool)
    pool = pool[:TRAIN_POOL_SIZE]
    write_ids("train_pool", pool)

    # Search / final split
    search, final, qtype, n_groups = split_search_final(dev)
    assert not set(search) & set(final), "search and final overlap"
    assert set(search) | set(final) == set(val_ids)
    write_ids("search", search)
    write_ids("final", final)

    totals = collections.Counter(qtype.values())
    sc = collections.Counter(qtype[q] for q in search)
    print(f"groups: {n_groups}")
    print(f"train_pool: {len(pool)}  search: {len(search)} ({len(search) / len(val_ids):.1%})  final: {len(final)}")
    for t in sorted(totals):
        print(f"  type{t}: all {totals[t] / len(val_ids):.3f}  search {sc[t] / len(search):.3f}")
    print("Done! Ready to train.")
