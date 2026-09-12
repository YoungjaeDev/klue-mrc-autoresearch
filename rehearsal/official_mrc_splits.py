"""Deterministic grouped public-dev split; labels never determine group assignment."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

from . import official_mrc as task


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def group_keys(row):
    keys = ["context:" + normalize(row["context"])]
    source, title = normalize(row.get("source")), normalize(row.get("title"))
    if source and title:
        keys.append("article:" + json.dumps([source, title], ensure_ascii=False))
    return keys


def row_fingerprint(rows):
    # Metadata is separately frozen in preparation. These fields also exist in official JSON.
    fields = ("id", "context", "question", "question_type", "answers")
    return fingerprint([{key: row[key] for key in fields} for row in sorted(rows, key=lambda r: r["id"])])


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
    names = {root: fingerprint(sorted(keys)) for root, keys in components.items()}
    return ([names[find(group_keys(row)[0])] for row in train],
            {row["id"]: names[find(group_keys(row)[0])] for row in validation})


def build_manifest(train, validation, seed=3407, search_fraction=0.2):
    task.index_rows(validation)
    if train:
        task.index_rows(train)
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

    ordered = sorted(by_group, key=lambda g: (-len(by_group[g]), fingerprint([seed, g])))
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
        parts[name] = {"ids": ids, "rows": len(ids), "ids_sha256": fingerprint(ids),
                       "question_types": {str(k): v for k, v in sorted(Counter(row["question_type"] for row in selected).items())}}
    train_contexts = {group_keys(row)[0] for row in train}
    train_articles = {key for row in train for key in group_keys(row)[1:]}
    trained_groups = set(train_groups)
    report = {
        "schema_version": 1, "seed": seed, "search_fraction_target": search_fraction,
        "algorithm": "NFKC whitespace context OR nonempty source/title connected components across train+dev; descending group size, seeded SHA tie order, greedy type-count squared error then improving group toggles",
        "dataset_sha256": task.DEV_SHA256, "validation_rows_sha256": row_fingerprint(validation),
        "validation_rows": len(validation), "groups": dict(sorted(groups.items())), "splits": parts,
        "prior_exposure": "Public KLUE dev is not a private test set. Record any earlier evaluation exposure in the campaign. Final is held out from this search and its model selection only.",
        "final_policy": "No final labels, predictions or metrics for candidate selection. Controller releases final only after selection is complete.",
        "overlap_with_train": {
            "normalized_context_rows": sum(group_keys(row)[0] in train_contexts for row in validation),
            "source_title_rows": sum(any(key in train_articles for key in group_keys(row)[1:]) for row in validation),
            "connected_component_rows": sum(groups[row["id"]] in trained_groups for row in validation),
        },
    }
    validate_manifest(report, validation)
    return report


def validate_manifest(manifest, rows):
    indexed = task.index_rows(rows)
    if manifest.get("schema_version") != 1 or manifest["validation_rows_sha256"] != row_fingerprint(rows):
        raise ValueError("Split schema or source row fingerprint mismatch")
    if manifest["dataset_sha256"] != task.DEV_SHA256 or manifest["validation_rows"] != len(rows):
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
    if not expected_sha256 or task.digest(path) != expected_sha256:
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
    indexed = task.index_rows(rows)
    return ([indexed[i] for i in manifest["splits"][split]["ids"]],
            {"split": split, "split_manifest_sha256": expected_sha256, "prior_exposure": manifest["prior_exposure"]})
