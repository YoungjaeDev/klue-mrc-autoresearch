"""Official KLUE-MRC v1.1 dev data and unmodified official EM/ROUGE-W scorer."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

from .vendor import klue_baseline_utils as official


ROOT = Path(__file__).resolve().parents[1]
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


def prepare(directory: Path = DEFAULT_DATA, source: Path | None = None) -> dict:
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
