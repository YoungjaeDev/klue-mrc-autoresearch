"""Opt-in W&B telemetry; local evidence is always written before network calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

SCALAR_FIELDS = {"loss", "learning_rate", "grad_norm", "epoch", "step_seconds",
                 "gpu_allocated_bytes", "gpu_reserved_bytes", "peak_gpu_allocated_bytes"}
METADATA_FIELDS = {"model_id", "model_revision", "train_sha256", "train_code_sha256",
                   "adapter_sha256", "hypothesis", "diff_sha256", "source_duration_seconds",
                   "source_run_started_at", "source_run_ended_at", "source_time_evidence", "run_kind", "diagnostic",
                   "conditions_sha256", "scores_sha256", "split", "adapter_identity_sha256"}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def filter_metadata(values: dict) -> dict:
    """Return only explicitly approved scalar/string run metadata."""
    result = {}
    for key in METADATA_FIELDS:
        value = values.get(key)
        if value is None or isinstance(value, bool):
            if value is not None:
                result[key] = value
        elif isinstance(value, (str, int)) or _finite_number(value):
            result[key] = value
    return result


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sanitize_event(row: dict) -> dict:
    if not isinstance(row, dict) or not isinstance(row.get("event_index"), int):
        raise ValueError("event_index must be an integer")
    if not isinstance(row.get("optimizer_step"), int) or row["optimizer_step"] < 0:
        raise ValueError("optimizer_step must be a nonnegative integer")
    if row.get("event") not in {"train_start", "step", "train_end"}:
        raise ValueError("Unknown event type")
    result = {key: row[key] for key in ("event_index", "event", "optimizer_step")}
    timestamp = row.get("timestamp")
    if isinstance(timestamp, str):
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        result["timestamp"] = timestamp
    for key in {"elapsed_seconds", *SCALAR_FIELDS}:
        if key in row and _finite_number(row[key]):
            result[key] = row[key]
    return result


def _read_complete_jsonl(path: Path) -> list[dict]:
    payload = path.read_text(encoding="utf-8")
    lines = payload.splitlines()
    if payload and not payload.endswith(("\n", "\r")):
        lines = lines[:-1]
    rows = [sanitize_event(json.loads(line)) for line in lines if line.strip()]
    for expected, row in enumerate(rows):
        if row["event_index"] != expected:
            raise ValueError("event_index must be contiguous from zero")
    return rows


def _mirror_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    next_index = len(existing)
    if any(row.get("event_index") != index for index, row in enumerate(existing)):
        raise ValueError("Sanitized event mirror is not contiguous")
    if next_index > len(rows) or existing != rows[:next_index]:
        raise ValueError("Sanitized event mirror differs from source")
    if next_index == len(rows):
        return
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows[next_index:]:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def configure_live_metrics(run) -> None:
    run.define_metric("optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("candidate_index")
    run.define_metric("candidate/*", step_metric="candidate_index")


def _wandb_payload(event: dict, candidate_index: int | None) -> dict:
    payload = {
        "optimizer_step": event["optimizer_step"],
        "train/event_index": event["event_index"],
        "train/event": event["event"],
    }
    if candidate_index is not None:
        payload["candidate_index"] = candidate_index
    if "elapsed_seconds" in event:
        payload["train/elapsed_seconds"] = event["elapsed_seconds"]
    if "timestamp" in event:
        parsed = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        payload["_timestamp"] = parsed.timestamp()
    for key in SCALAR_FIELDS:
        if key in event:
            payload[f"train/{key}"] = event[key]
    return payload


def sync_live_events(events_path: Path, state_path: Path, mirror_path: Path, run, candidate_index: int | None) -> dict:
    rows = _read_complete_jsonl(events_path)
    _mirror_events(mirror_path, rows)
    state = read_json(state_path) if state_path.exists() else {"next_event_index": 0, "run_id": run.id}
    if state.get("run_id") != run.id:
        raise ValueError("W&B run id differs from local cursor")
    cursor = state.get("next_event_index")
    if not isinstance(cursor, int) or cursor < 0 or cursor > len(rows):
        raise ValueError("Invalid event cursor")
    configure_live_metrics(run)
    logged = 0
    try:
        for row in rows[cursor:]:
            run.log(_wandb_payload(row, candidate_index), step=row["event_index"], commit=True)
            cursor = row["event_index"] + 1
            write_json(state_path, {"next_event_index": cursor, "run_id": run.id})
            logged += 1
    except BaseException as error:
        if not state_path.exists():
            write_json(state_path, {"next_event_index": cursor, "run_id": run.id})
        return {"status": "online_failed", "logged": logged, "next_event_index": cursor,
                "error_type": type(error).__name__}
    if not state_path.exists():
        write_json(state_path, {"next_event_index": cursor, "run_id": run.id})
    return {"status": "synced", "logged": logged, "next_event_index": cursor}


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("Use a short run/group name containing letters, digits, dot, dash or underscore")
    return value


def local_directory(args):
    return args.tracking_dir.resolve() / safe_name(args.group)


def sanitized_status(status):
    result = {"status": status.get("status", "running")}
    if isinstance(status.get("diagnostic"), bool):
        result["diagnostic"] = status["diagnostic"]
    for key in ("train_seconds", "duration_seconds", "peak_gpu_allocated_bytes", "optimizer_steps"):
        if _finite_number(status.get(key)):
            result[key] = status[key]
    return result


def _load_key(path):
    if os.environ.get("WANDB_API_KEY"):
        return True
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == "WANDB_API_KEY" and value.strip().strip("\"'"):
                    os.environ["WANDB_API_KEY"] = value.strip().strip("\"'")
                    return True
    return False


def _online_run(args, directory, metadata):
    """No network or W&B import without explicit --online and a credential."""
    if not args.online:
        return None, {"status": "local_only"}
    if not _load_key(args.env_file):
        return None, {"status": "local_only", "reason": "MissingCredential"}
    if not args.entity or not args.project:
        raise ValueError("Online logging requires explicit --entity and --project")
    import wandb
    from wandb_gql import gql
    api = wandb.Api(timeout=20)
    response = api.client.execute(gql("""query Privacy($project: String!, $entity: String!) {
        project(name: $project, entityName: $entity) { access }
    }"""), {"project": args.project, "entity": args.entity})
    if str((response.get("project") or {}).get("access", "")).upper() != "PRIVATE":
        raise ValueError("Create a private project first; this tool never changes project visibility")
    remote = {"entity": args.entity, "project": args.project, "group": args.group, "name": args.run_name}
    state_path = directory / "online.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["remote"] != remote:
            raise ValueError("Online destination changed; use a new local run name")
    else:
        import uuid
        state = {"run_id": uuid.uuid4().hex[:12], "remote": remote}
        write_json(state_path, state)
    (directory / ".wandb").mkdir(parents=True, exist_ok=True)
    run = wandb.init(entity=args.entity, project=args.project, group=args.group, name=args.run_name,
        id=state["run_id"], resume="allow", config=filter_metadata(metadata),
        settings=wandb.Settings(root_dir=str(directory / ".wandb"), console="off", silent=True,
            disable_code=True, disable_git=True, save_code=False, x_disable_stats=True, x_disable_meta=True,
            x_disable_machine_info=True, x_save_requirements=False))
    if run is None:
        raise RuntimeError("W&B run initialization failed")
    return run, {"status": "online", "run_id": run.id}


def _finish(run, result):
    if run is not None:
        try:
            run.finish(exit_code=1 if result.get("status") == "online_failed" else 0)
        except Exception as error:
            result = {"status": "online_failed", "error_type": type(error).__name__}
    return result


def collect(args):
    directory = local_directory(args) / "runs" / safe_name(args.run_name)
    metadata = filter_metadata(read_json(args.metadata)) if args.metadata else {}
    write_json(directory / "metadata.json", metadata)
    run, online = None, {"status": "local_only"}
    initialized = False
    while True:
        if args.events.exists():
            rows = _read_complete_jsonl(args.events)
            _mirror_events(directory / "sanitized-events.jsonl", rows)
        status = sanitized_status(read_json(args.status)) if args.status.exists() else {"status": "running"}
        write_json(directory / "source-status.json", status)
        if not initialized:
            try:
                run, online = _online_run(args, directory, metadata)
            except Exception as error:
                online = {"status": "online_failed", "error_type": type(error).__name__}
            initialized = True
        if run is not None and args.events.exists() and online["status"] != "online_failed":
            online = sync_live_events(args.events, directory / "cursor.json",
                directory / "sanitized-events.jsonl", run, args.candidate_index)
        write_json(directory / "tracking-status.json", {**online, "source_status": status["status"]})
        if not args.follow or status["status"] not in {"running", "pending"}:
            break
        time.sleep(args.poll_seconds)
    online = _finish(run, online)
    result = {**online, "source_status": status["status"]}
    write_json(directory / "tracking-status.json", result)
    return result


def validated_scores(path):
    """Read sealed scalar scores only; raw predictions are hashed, never uploaded."""
    from rehearsal import official_mrc_runner as evaluator
    from rehearsal import official_mrc as task
    task.verify_scorer()
    if path.name != "scores.json":
        raise ValueError("Pass the evaluator's scores.json with its sibling evidence files")
    record, scores = evaluator.read_completed(path.parent)
    if record["conditions"].get("metric_sha256") != task.METRIC_SHA256:
        raise ValueError("Evaluation used a different official metric")
    result = {"overall": scores["overall"], "by_question_type": scores["by_question_type"]}
    safe = {}
    for name, group in [("overall", result["overall"]), *result["by_question_type"].items()]:
        if not isinstance(group.get("count"), int) or group["count"] < 0:
            raise ValueError("Invalid score count")
        safe[name] = {"count": group["count"]}
        for metric in ("exact_match", "rouge_w"):
            value = group.get(metric)
            if not _finite_number(value) or not 0 <= value <= 100:
                raise ValueError("Invalid official score")
            safe[name][metric] = value
    if set(safe) != {"overall", "1", "2", "3"}:
        raise ValueError("All three question types are required")
    if sum(safe[k]["count"] for k in ("1", "2", "3")) != safe["overall"]["count"]:
        raise ValueError("Score counts do not add up")
    selection = record["conditions"].get("selection", {})
    if selection.get("split") not in {"search", "final"} or scores.get("selection") != selection:
        raise ValueError("Only matching named search/final evaluation is accepted")
    if selection.get("diagnostic_prefix") is not False:
        raise ValueError("Diagnostic prefix scores cannot initialize a baseline or compare candidates")
    return record, {"overall": safe.pop("overall"), "by_question_type": safe}


def training_provenance(source, kind, training_dir=None, adapter_dir=None):
    """Bind the evaluated adapter bytes to a completed SDK run or explicit Studio adapter."""
    from rehearsal.official_mrc_runner import local_identity
    evaluated = source.get("adapter")
    if kind == "base":
        if evaluated is not None or training_dir or adapter_dir:
            raise ValueError("Base requires no adapter or training source")
        return {"status": "completed"}, {}, {"source_kind": "base", "evaluated_adapter": None}
    if not isinstance(evaluated, dict) or not evaluated.get("files"):
        raise ValueError("A trained arm requires the evaluator's adapter file map")
    trusted, status, provenance = {}, {"status": "completed"}, {}
    if kind == "studio":
        if training_dir or not adapter_dir:
            raise ValueError("Studio requires an explicit --adapter and no SDK training directory")
        adapter_dir = adapter_dir.resolve(strict=True)
        provenance = {"source_kind": "historical_studio", "adapter_directory": str(adapter_dir)}
    else:
        if not training_dir or adapter_dir:
            raise ValueError("Reference/candidate/selected requires --training-dir")
        training_dir = training_dir.resolve(strict=True)
        raw_status = read_json(training_dir / "status.json")
        run = read_json(training_dir / "run.json")
        artifacts = read_json(training_dir / "artifacts.json")
        if (raw_status.get("status") != "completed" or raw_status.get("diagnostic") is not False
                or run.get("diagnostic") is not False or raw_status.get("optimizer_steps") != 30
                or raw_status.get("sample_presentations") != 240):
            raise ValueError("Training source must be completed, non-diagnostic 30-step SDK output")
        for filename, field in (("run.json", "run_sha256"), ("events.jsonl", "events_sha256")):
            if sha256(training_dir / filename) != raw_status.get(field):
                raise ValueError("Training status does not seal its run/events")
        recipe = run.get("recipe", {})
        if any(recipe.get(k) != v for k, v in {"max_steps": 30, "batch_size": 2, "gradient_accumulation_steps": 4}.items()):
            raise ValueError("Training source changed the fixed update budget")
        allowed = {"reference", "candidate"} if kind == "selected" else {kind}
        if run.get("run_kind") not in allowed:
            raise ValueError("Requested arm differs from the training run kind")
        base = source["conditions"].get("base_model")
        if base != {"model_id": run.get("model_id"), "revision": run.get("model_revision")}:
            raise ValueError("Evaluated base differs from the training model/revision")
        adapter_dir = training_dir / "adapter"
        if Path(raw_status.get("adapter_path", "")).resolve() != adapter_dir.resolve():
            raise ValueError("Training status points outside its canonical adapter directory")
        if not artifacts:
            raise ValueError("Training artifact manifest is empty")
        for relative, expected in artifacts.items():
            artifact = (training_dir / relative).resolve(strict=True)
            if not artifact.is_relative_to(adapter_dir.resolve()) or sha256(artifact) != expected:
                raise ValueError("Training artifact checksum/path mismatch")
        actual_files = {path.relative_to(training_dir).as_posix() for path in adapter_dir.rglob("*") if path.is_file()}
        if actual_files != set(artifacts):
            raise ValueError("Adapter files differ from the training artifact manifest")
        trusted = {key: run[key] for key in ("model_id", "model_revision", "train_sha256", "train_code_sha256", "run_kind")}
        status = sanitized_status(raw_status)
        provenance = {"source_kind": "sdk", "training_directory": str(training_dir),
                      "training_run_sha256": sha256(training_dir / "run.json"),
                      "training_status_sha256": sha256(training_dir / "status.json"),
                      "artifacts_sha256": sha256(training_dir / "artifacts.json")}
    identity = local_identity(adapter_dir)
    if identity["files"] != evaluated["files"]:
        raise ValueError("Evaluated adapter differs from the declared training/Studio artifact")
    identity_hash = hashlib.sha256(json.dumps(identity["files"], sort_keys=True).encode()).hexdigest()
    trusted["adapter_identity_sha256"] = identity_hash
    trusted["adapter_sha256"] = identity["files"].get("adapter_model.safetensors", identity_hash)
    provenance.update(evaluated_adapter=evaluated, adapter_identity_sha256=identity_hash)
    return status, trusted, provenance


def score_record(args):
    source, scores = validated_scores(args.scores)
    status, trusted, provenance = training_provenance(source, args.kind, args.training_dir, args.adapter)
    directory = local_directory(args)
    condition_hash = hashlib.sha256(json.dumps(source["conditions"], sort_keys=True).encode()).hexdigest()
    split = source["conditions"]["selection"]["split"]
    metadata = filter_metadata(read_json(args.metadata)) if args.metadata else {}
    if any(key in metadata and metadata[key] != value for key, value in trusted.items()):
        raise ValueError("Supplied metadata differs from verified training provenance")
    record = {**metadata, **trusted, **status, "name": args.run_name, "kind": args.kind,
        "provenance": provenance, "official_scores": scores,
        "conditions_sha256": condition_hash, "scores_sha256": sha256(args.scores), "split": split,
        "evidence_path": str(args.scores.resolve())}
    if args.kind == "candidate":
        if split != "search" or not args.candidate_index or args.candidate_index < 1:
            raise ValueError("A candidate needs a positive index and search scores")
        baseline = read_json(directory / "search-baseline.json")
        if baseline["conditions_sha256"] != condition_hash:
            raise ValueError("Candidate evaluation conditions differ from reference")
        if baseline.get("train_sha256") != record["train_sha256"]:
            raise ValueError("Candidate training data differs from the SDK reference")
        path = directory / "candidates.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        previous = [r for r in rows if r["candidate_index"] < args.candidate_index]
        best = max([baseline["official_scores"]["overall"]["exact_match"],
                    *[r["official_scores"]["overall"]["exact_match"] for r in previous]])
        expected = "keep" if scores["overall"]["exact_match"] > best else "discard"
        if args.decision != expected:
            raise ValueError("Decision differs from strict EM improvement over current best")
        record.update(candidate_index=args.candidate_index, decision=args.decision)
        same = [r for r in rows if r["candidate_index"] == args.candidate_index]
        if same and same != [record]:
            raise ValueError("Candidate evidence changed; preserve the original record")
        rows = [r for r in rows if r["candidate_index"] != args.candidate_index] + [record]
        write_json(path, sorted(rows, key=lambda r: r["candidate_index"]))
    elif split == "search":
        if args.kind != "reference":
            raise ValueError("Only SDK reference initializes the search best curve")
        record["candidate_index"] = 0
        path = directory / "search-baseline.json"
        if path.exists() and read_json(path) != record:
            raise ValueError("SDK search baseline is immutable within a group")
        write_json(path, record)
    else:
        arm = {"base": "base", "studio": "historical_studio30", "reference": "sdk_reference30", "selected": "selected"}[args.kind]
        record["arm"] = arm
        path = directory / "arms.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if any(r["conditions_sha256"] != condition_hash for r in rows):
            raise ValueError("Final arms have different evaluation conditions")
        same = [r for r in rows if r["arm"] == arm]
        if same and same != [record]:
            raise ValueError("Final arm evidence changed")
        write_json(path, [r for r in rows if r["arm"] != arm] + [record])
    return record


def score(args):
    record = score_record(args)  # Fully verified and mirrored before any network operation.
    directory = local_directory(args) / "runs" / safe_name(args.run_name)
    run, result = None, {"status": "local_only"}
    try:
        run, result = _online_run(args, directory, record)
        if run is not None:
            payload = {"official/split": record["split"], "official/scores_sha256": record["scores_sha256"]}
            for group, values in [("overall", record["official_scores"]["overall"]),
                                  *record["official_scores"]["by_question_type"].items()]:
                for metric, value in values.items():
                    payload[f"official/{group}/{metric}"] = value
            if "candidate_index" in record:
                payload["candidate_index"] = record["candidate_index"]
            if "decision" in record:
                payload["candidate/decision"] = record["decision"]
            run.summary.update(payload)
            result = {"status": "synced", "run_id": run.id}
    except Exception as error:
        result = {"status": "online_failed", "error_type": type(error).__name__}
    result = _finish(run, result)
    write_json(directory / "score-tracking-status.json", result)
    return result


def historical(args):
    """Import user-supplied loss history without inventing timing or adapter identity."""
    state = read_json(args.trainer_state)
    steps = state.get("global_step")
    train_events, eval_events = [], []
    for row in state.get("log_history", []):
        if isinstance(row.get("step"), int) and _finite_number(row.get("loss")):
            train_events.append({"optimizer_step": row["step"], "loss": row["loss"]})
        if isinstance(row.get("step"), int) and _finite_number(row.get("eval_loss")):
            eval_events.append({"optimizer_step": row["step"], "eval_loss": row["eval_loss"]})
    if not isinstance(steps, int) or steps < 1 or [r["optimizer_step"] for r in train_events] != list(range(1, steps + 1)):
        raise ValueError("Historical loss must contain every optimizer step")
    metadata = filter_metadata(read_json(args.metadata)) if args.metadata else {}
    record = {"metadata": metadata, "trainer_state_sha256": sha256(args.trainer_state),
              "per_step_timestamps_available": False, "train_events": train_events,
              "eval_events": eval_events, "pure_train_seconds": None}
    path = local_directory(args) / "historical-studio30.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("Historical source changed; use a new group")
    write_json(path, record)
    # Historical import is deliberately local. Score/collect are the opt-in online surfaces.
    return {"status": "local_only", "optimizer_steps": steps}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    def common(command):
        command.add_argument("--tracking-dir", type=Path, default=Path("outputs/tracking"))
        command.add_argument("--group", type=safe_name, required=True)
        command.add_argument("--run-name", type=safe_name, required=True)
        command.add_argument("--metadata", type=Path)
        command.add_argument("--online", action="store_true")
        command.add_argument("--entity")
        command.add_argument("--project")
        command.add_argument("--env-file", type=Path, default=Path(".env"))
    command = sub.add_parser("collect", help="Mirror SDK telemetry locally; --online opts into W&B")
    common(command)
    command.add_argument("--events", type=Path, required=True)
    command.add_argument("--status", type=Path, required=True)
    command.add_argument("--candidate-index", type=int)
    command.add_argument("--follow", action="store_true")
    command.add_argument("--poll-seconds", type=float, default=2.0)
    command.set_defaults(func=collect)
    command = sub.add_parser("score", help="Verify sealed official scores before local/online tracking")
    common(command)
    command.add_argument("--scores", type=Path, required=True)
    command.add_argument("--training-dir", type=Path, help="Canonical completed SDK output containing status/run/events/artifacts and adapter/")
    command.add_argument("--adapter", type=Path, help="Explicit historical Studio adapter directory; only for --kind studio")
    command.add_argument("--kind", choices=["base", "studio", "reference", "selected", "candidate"], required=True)
    command.add_argument("--candidate-index", type=int)
    command.add_argument("--decision", choices=["keep", "discard"])
    command.set_defaults(func=score)
    command = sub.add_parser("historical-import", help="Import optional Studio trainer_state locally")
    command.add_argument("--tracking-dir", type=Path, default=Path("outputs/tracking"))
    command.add_argument("--group", type=safe_name, required=True)
    command.add_argument("--trainer-state", type=Path, required=True)
    command.add_argument("--metadata", type=Path)
    command.set_defaults(func=historical)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "poll_seconds", 2) <= 0:
        raise ValueError("poll-seconds must be positive")
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["status"] == "online_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
