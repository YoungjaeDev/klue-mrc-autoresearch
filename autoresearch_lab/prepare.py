"""Prepare fixed Studio-style rows and grouped KLUE search/final IDs on CPU."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import urllib.request

from rehearsal import official_mrc as task
from rehearsal import official_mrc_splits as splitters

ROOT = Path(__file__).resolve().parents[1]
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
FROZEN_RUNTIME_FILES = ("prepare.py", "autoresearch_lab/bootstrap.py", "autoresearch_lab/run.py",
                        "program.md", "pyproject.toml", "uv.lock")


def runtime_freeze(root=ROOT):
    """Freeze current execution rules/dependencies; root train.py remains agent-editable."""
    return {str((root / relative).resolve()): task.digest(root / relative) for relative in FROZEN_RUNTIME_FILES}


def download_file(url, destination, expected=None):
    """Anonymous download with atomic promotion; never overwrite different data."""
    destination = Path(destination)
    if destination.exists():
        if expected and task.digest(destination) != expected:
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
        if expected and task.digest(temporary) != expected:
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
    if not (task.DEFAULT_DATA / "official-dev.json").exists():
        task.prepare()
    task.load_data()


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_message(row, original):
    if set(row) != {"id", "messages"} or row["id"] != original["id"]:
        raise ValueError("Original ID or public fields changed")
    target = task.NO_ANSWER if original["is_impossible"] else original["answers"][0]
    expected = task.messages_for(original) + [{"role": "assistant", "content": target}]
    if row["messages"] != expected:
        raise ValueError("Original prompt, roles or answer changed")


def load_full_source(source=SOURCE):
    """Check every public row against pinned original parquet, without model imports."""
    import pandas as pd
    source = Path(source)
    if task.digest(source / "source-manifest.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Source manifest checksum mismatch")
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("license") != "CC-BY-SA-4.0" or manifest.get("revision") != SOURCE_REVISION:
        raise ValueError("Source license/revision mismatch")
    messages, originals = {}, {}
    for name in ("train", "validation"):
        path = source / f"{name}.jsonl"
        parquet = source / f"{name}.parquet"
        if task.digest(path) != MESSAGE_HASHES[name] or task.digest(parquet) != SOURCE_HASHES[name]:
            raise ValueError("Pinned source or messages checksum mismatch")
        messages[name] = task.read_jsonl(path)
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


def prepare(output=DEFAULT_OUTPUT, source=SOURCE):
    from datasets import Dataset
    messages, originals = load_full_source(source)
    official = task.load_data()
    if splitters.row_fingerprint(originals["validation"]) != splitters.row_fingerprint(official):
        raise ValueError("HF validation differs from official public dev")
    split_manifest = splitters.build_manifest(originals["train"], originals["validation"])
    # Reproduces the recorded row selection policy independently of Studio code.
    # This is source selection, not a claim of
    # identical downstream sampler order, masked tokens or floating-point weights.
    bounded = list(Dataset.from_list(messages["train"]).shuffle(seed=3407).select(range(1024)))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_path, split_path = output / "train.jsonl", output / "splits.json"
    write_rows(train_path, bounded)
    task.write_json(split_path, split_manifest)
    by_id = task.index_rows(messages["validation"])
    search_path = output / "search.jsonl"
    write_rows(search_path, [by_id[i] for i in split_manifest["splits"]["search"]["ids"]])
    owned = [Path(__file__), Path(task.__file__), Path(splitters.__file__),
             ROOT / "rehearsal/official_mrc_runner.py", ROOT / "rehearsal/vendor/klue_baseline_utils.py",
             train_path, split_path, search_path,
             source / "train.jsonl", source / "validation.jsonl", source / "source-manifest.json",
             source / "train.parquet", source / "validation.parquet",
             task.DEFAULT_DATA / "official-dev.json"]
    result = {
        "schema_version": 1, "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "seed": 3407, "train_path": str(train_path), "train_sha256": task.digest(train_path),
        "train_rows": len(bounded), "train_ids": [r["id"] for r in bounded],
        "train_ids_sha256": splitters.fingerprint([r["id"] for r in bounded]),
        "search_path": str(search_path), "search_sha256": task.digest(search_path),
        "split_manifest_path": str(split_path), "split_manifest_sha256": task.digest(split_path),
        "source": {"dataset": "klue/klue", "revision": SOURCE_REVISION,
                   "messages_dataset": MESSAGE_REPO, "messages_revision": MESSAGE_REVISION,
                   "rows": {"train": 17554, "validation": 5841},
                   "message_sha256": MESSAGE_HASHES, "license": "CC-BY-SA-4.0"},
        "row_bound": {"max_train_rows": 1024, "seed": 3407, "algorithm": "datasets.Dataset.shuffle(seed=3407).select(range(1024))",
                      "datasets_version": version("datasets"), "numpy_version": version("numpy"),
                      "scope": "Input row selection only; Studio dataloader/template/weights equivalence is not asserted"},
        "splits": {k: {key: val for key, val in v.items() if key != "ids"} for k, v in split_manifest["splits"].items()},
        "prior_exposure": split_manifest["prior_exposure"], "overlap_with_train": split_manifest["overlap_with_train"],
        "frozen_files": {**{str(path.resolve()): task.digest(path) for path in owned}, **runtime_freeze()},
    }
    task.write_json(output / "manifest.json", result)
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
