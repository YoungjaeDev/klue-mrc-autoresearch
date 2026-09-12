"""Download pinned public KLUE data, verify every row and freeze train rows plus search/final IDs. CPU only."""
import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import unicodedata
import urllib.request

import evaluate


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/source/klue-mrc"
MESSAGE_REPO = "YoungjaeDev/klue-mrc-messages"
MESSAGE_REVISION = "ae6c27baff54df9d0a63ed85451badd6aefc131c"
SOURCE_MANIFEST_SHA256 = "c15660aa78eb59f7aad97ed55f012ca8eb73e95b6c6ba624bb29dd0029a9ddb2"
DEFAULT_OUTPUT = ROOT / "data/generated/klue-autoresearch-v1"
SOURCE_REVISION = "349481ec73fff722f88e0453ca05c77a447d967c"
SOURCE_HASHES = {"train": "45fcfd50b99aed4cd55c8c5856e42aa4a5139808cc36b36994b3c1991aca93dd",
                 "validation": "f576c924f471702a6fa3369d314bea2e47545c56736f371a2adb90ed15913738"}
MESSAGE_HASHES = {"train": "cf95dc2a800a088c8dca201b8c0fbffc0b09c37f1f34ffdea7fafe26635de997",
                  "validation": "6d5619a88708b90fa8945b38c70e82f1a06bfb483fe6ac1588df385817073f0e"}
COUNTS = {"train": {1: 7308, 2: 4729, 3: 5517}, "validation": {1: 2437, 2: 1571, 3: 1833}}
FROZEN_RUNTIME_FILES = ("prepare.py", "evaluate.py", "run.py", "program.md", "pyproject.toml", "uv.lock")


def runtime_freeze(root=ROOT):
    """Freeze current execution rules/dependencies; root train.py remains agent-editable."""
    return {str((root / relative).resolve()): evaluate.digest(root / relative) for relative in FROZEN_RUNTIME_FILES}


def download_file(url, destination, expected=None):
    """Anonymous download with atomic promotion; never overwrite different data."""
    destination = Path(destination)
    if destination.exists():
        if expected and evaluate.digest(destination) != expected:
            raise ValueError(f"Existing file checksum mismatch: {destination.name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            with urllib.request.urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
        if expected and evaluate.digest(temporary) != expected:
            raise ValueError(f"Downloaded checksum mismatch: {destination.name}")
        temporary.rename(destination)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return destination


def download_source(source=SOURCE):
    source = Path(source)
    base = f"https://huggingface.co/datasets/{MESSAGE_REPO}/resolve/{MESSAGE_REVISION}"
    download_file(f"{base}/source-manifest.json", source / "source-manifest.json", SOURCE_MANIFEST_SHA256)
    for name in ("train", "validation"):
        download_file(f"{base}/{name}.jsonl", source / f"{name}.jsonl", MESSAGE_HASHES[name])
        url = (f"https://huggingface.co/datasets/klue/klue/resolve/{SOURCE_REVISION}"
               f"/mrc/{name}-00000-of-00001.parquet")
        download_file(url, source / f"{name}.parquet", SOURCE_HASHES[name])
    if not (evaluate.DEFAULT_DATA / "official-dev.json").exists():
        evaluate.prepare_dev()
    evaluate.load_data()


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_message(row, original):
    if set(row) != {"id", "messages"} or row["id"] != original["id"]:
        raise ValueError("Original ID or public fields changed")
    target = evaluate.NO_ANSWER if original["is_impossible"] else original["answers"][0]
    expected = evaluate.messages_for(original) + [{"role": "assistant", "content": target}]
    if row["messages"] != expected:
        raise ValueError("Original prompt, roles or answer changed")


def load_full_source(source=SOURCE):
    """Check every public row against pinned original parquet, without model imports."""
    import pandas as pd
    source = Path(source)
    if evaluate.digest(source / "source-manifest.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Source manifest checksum mismatch")
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("license") != "CC-BY-SA-4.0" or manifest.get("revision") != SOURCE_REVISION:
        raise ValueError("Source license/revision mismatch")
    messages, originals = {}, {}
    for name in ("train", "validation"):
        path = source / f"{name}.jsonl"
        parquet = source / f"{name}.parquet"
        if evaluate.digest(path) != MESSAGE_HASHES[name] or evaluate.digest(parquet) != SOURCE_HASHES[name]:
            raise ValueError("Pinned source or messages checksum mismatch")
        messages[name] = evaluate.read_jsonl(path)
        originals[name] = []
        for record in pd.read_parquet(parquet).itertuples(index=False):
            impossible = bool(record.is_impossible)
            answers = [] if impossible else list(record.answers["text"])
            if impossible != (record.question_type == 3) or (not impossible and not answers):
                raise ValueError("Original answer/type mismatch")
            originals[name].append({"id": record.guid, "context": record.context, "question": record.question,
                "question_type": int(record.question_type), "is_impossible": impossible,
                "answers": answers, "source": record.source, "title": record.title})
        if len(messages[name]) != len(originals[name]) or Counter(r["question_type"] for r in originals[name]) != COUNTS[name]:
            raise ValueError("Full original counts mismatch")
        for message, original in zip(messages[name], originals[name]):
            verify_message(message, original)
    if {r["id"] for r in messages["train"]} & {r["id"] for r in messages["validation"]}:
        raise ValueError("Original train/validation IDs overlap")
    return messages, originals


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def group_keys(row):
    keys = ["context:" + normalize(row["context"])]
    source, title = normalize(row.get("source")), normalize(row.get("title"))
    if source and title:
        keys.append("article:" + json.dumps([source, title], ensure_ascii=False))
    return keys


def connected_groups(train, validation):
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row in [*train, *validation]:
        keys = group_keys(row)
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            parent[max(a, b)] = min(a, b)
        find(keys[0])
    components = defaultdict(list)
    for key in list(parent):
        components[find(key)].append(key)
    names = {root: evaluate.fingerprint(sorted(keys)) for root, keys in components.items()}
    return ([names[find(group_keys(row)[0])] for row in train],
            {row["id"]: names[find(group_keys(row)[0])] for row in validation})


def build_manifest(train, validation, seed=3407, search_fraction=0.2):
    evaluate.index_rows(validation)
    if train:
        evaluate.index_rows(train)
    if not 0 < search_fraction < 1:
        raise ValueError("search_fraction must be between zero and one")
    train_groups, groups = connected_groups(train, validation)
    by_group = defaultdict(list)
    for row in validation:
        by_group[groups[row["id"]]].append(row)
    if len(by_group) < 2:
        raise ValueError("At least two independent groups are needed")
    totals = Counter(row["question_type"] for row in validation)
    targets = {kind: totals[kind] * search_fraction for kind in sorted(totals)}
    counts = Counter()
    search_groups = set()

    def error(values):
        return sum((values[kind] - target) ** 2 / max(target, 1) for kind, target in targets.items())

    ordered = sorted(by_group, key=lambda g: (-len(by_group[g]), evaluate.fingerprint([seed, g])))
    for group in ordered:
        candidate = counts + Counter(row["question_type"] for row in by_group[group])
        if error(candidate) < error(counts):
            search_groups.add(group)
            counts = candidate
    # A large early group can overshoot one type while improving another.
    # Revisit additions/removals using type counts only, never model outputs.
    changed = True
    while changed:
        changed = False
        for group in ordered:
            amount = Counter(row["question_type"] for row in by_group[group])
            candidate = counts - amount if group in search_groups else counts + amount
            if error(candidate) < error(counts) - 1e-12:
                search_groups.symmetric_difference_update({group})
                counts = candidate
                changed = True
    if not search_groups or len(search_groups) == len(by_group):
        raise ValueError("Groups cannot provide a nonempty search and final partition")
    parts = {}
    for name in ("search", "final"):
        selected = sorted((row for row in validation if (groups[row["id"]] in search_groups) == (name == "search")), key=lambda r: r["id"])
        ids = [row["id"] for row in selected]
        parts[name] = {"ids": ids, "rows": len(ids), "ids_sha256": evaluate.fingerprint(ids),
                       "question_types": {str(k): v for k, v in sorted(Counter(row["question_type"] for row in selected).items())}}
    train_contexts = {group_keys(row)[0] for row in train}
    train_articles = {key for row in train for key in group_keys(row)[1:]}
    trained_groups = set(train_groups)
    report = {
        "schema_version": 1, "seed": seed, "search_fraction_target": search_fraction,
        "algorithm": "NFKC whitespace context OR nonempty source/title connected components across train+dev; descending group size, seeded SHA tie order, greedy type-count squared error then improving group toggles",
        "dataset_sha256": evaluate.DEV_SHA256, "validation_rows_sha256": evaluate.row_fingerprint(validation),
        "validation_rows": len(validation), "groups": dict(sorted(groups.items())), "splits": parts,
        "prior_exposure": "Public KLUE dev is not a private test set. Record any earlier evaluation exposure in the campaign. Final is held out from this search and its model selection only.",
        "final_policy": "No final labels, predictions or metrics for candidate selection. Controller releases final only after selection is complete.",
        "overlap_with_train": {
            "normalized_context_rows": sum(group_keys(row)[0] in train_contexts for row in validation),
            "source_title_rows": sum(any(key in train_articles for key in group_keys(row)[1:]) for row in validation),
            "connected_component_rows": sum(groups[row["id"]] in trained_groups for row in validation),
        },
    }
    evaluate.validate_manifest(report, validation)
    return report


def prepare(output=DEFAULT_OUTPUT, source=SOURCE):
    from datasets import Dataset
    messages, originals = load_full_source(source)
    official = evaluate.load_data()
    if evaluate.row_fingerprint(originals["validation"]) != evaluate.row_fingerprint(official):
        raise ValueError("HF validation differs from official public dev")
    split_manifest = build_manifest(originals["train"], originals["validation"])
    # Reproduces the recorded row selection policy independently of Studio code.
    # This is source selection, not a claim of
    # identical downstream sampler order, masked tokens or floating-point weights.
    bounded = list(Dataset.from_list(messages["train"]).shuffle(seed=3407).select(range(1024)))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_path, split_path = output / "train.jsonl", output / "splits.json"
    write_rows(train_path, bounded)
    evaluate.write_json(split_path, split_manifest)
    by_id = evaluate.index_rows(messages["validation"])
    search_path = output / "search.jsonl"
    write_rows(search_path, [by_id[i] for i in split_manifest["splits"]["search"]["ids"]])
    owned = [ROOT / "klue_scorer/klue_baseline_utils.py",
             train_path, split_path, search_path,
             source / "train.jsonl", source / "validation.jsonl", source / "source-manifest.json",
             source / "train.parquet", source / "validation.parquet",
             evaluate.DEFAULT_DATA / "official-dev.json"]
    result = {
        "schema_version": 1, "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "seed": 3407, "train_path": str(train_path), "train_sha256": evaluate.digest(train_path),
        "train_rows": len(bounded), "train_ids": [r["id"] for r in bounded],
        "train_ids_sha256": evaluate.fingerprint([r["id"] for r in bounded]),
        "search_path": str(search_path), "search_sha256": evaluate.digest(search_path),
        "split_manifest_path": str(split_path), "split_manifest_sha256": evaluate.digest(split_path),
        "source": {"dataset": "klue/klue", "revision": SOURCE_REVISION,
                   "messages_dataset": MESSAGE_REPO, "messages_revision": MESSAGE_REVISION,
                   "rows": {"train": 17554, "validation": 5841},
                   "message_sha256": MESSAGE_HASHES, "license": "CC-BY-SA-4.0"},
        "row_bound": {"max_train_rows": 1024, "seed": 3407, "algorithm": "datasets.Dataset.shuffle(seed=3407).select(range(1024))",
                      "datasets_version": version("datasets"), "numpy_version": version("numpy"),
                      "scope": "Input row selection only; Studio dataloader/template/weights equivalence is not asserted"},
        "splits": {k: {key: val for key, val in v.items() if key != "ids"} for k, v in split_manifest["splits"].items()},
        "prior_exposure": split_manifest["prior_exposure"], "overlap_with_train": split_manifest["overlap_with_train"],
        "frozen_files": {**{str(path.resolve()): evaluate.digest(path) for path in owned}, **runtime_freeze()},
    }
    evaluate.write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    download_source(args.source)
    result = prepare(args.output, args.source)
    print(json.dumps({"manifest": str(args.output.resolve() / "manifest.json"), "train_rows": result["train_rows"], "splits": result["splits"]}, indent=2))


if __name__ == "__main__":
    main()
