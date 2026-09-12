"""Render local autoresearch evidence as PNG and SVG figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARM_LABELS = {
    "base": "원 모델",
    "historical_studio30": "과거 Studio 30step",
    "sdk_reference30": "SDK 기준 30step",
    "selected": "최종 선택",
}


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_best_series(rows: list[dict], baseline: dict | None = None) -> dict[str, list]:
    result = {"candidate_index": [], "exact_match": [], "best_exact_match": [], "rouge_w": []}
    best = None
    if baseline:
        overall = baseline.get("official_scores", {}).get("overall")
        if baseline.get("status") == "completed" and isinstance(overall, dict):
            em, rouge = overall.get("exact_match"), overall.get("rouge_w")
            if isinstance(em, (int, float)) and isinstance(rouge, (int, float)):
                best = em
                result["candidate_index"].append(0)
                result["exact_match"].append(em)
                result["best_exact_match"].append(em)
                result["rouge_w"].append(rouge)
    completed_candidates = [row for row in rows
                            if row.get("status") == "completed" and isinstance(row.get("official_scores", {}).get("overall"), dict)]
    if completed_candidates and best is None:
        raise ValueError("Verified SDK reference search baseline is required before candidate best curve")
    for row in sorted(completed_candidates, key=lambda item: item.get("candidate_index", -1)):
        overall = row.get("official_scores", {}).get("overall")
        if row.get("status") != "completed" or not isinstance(overall, dict):
            continue
        em, rouge = overall.get("exact_match"), overall.get("rouge_w")
        if not isinstance(em, (int, float)) or not isinstance(rouge, (int, float)):
            continue
        best = em if best is None else max(best, em)
        result["candidate_index"].append(row["candidate_index"])
        result["exact_match"].append(em)
        result["best_exact_match"].append(best)
        result["rouge_w"].append(rouge)
    return result


def arm_score_series(rows: list[dict]) -> list[dict]:
    result = []
    indexed = {row.get("arm"): row for row in rows}
    for arm in ("base", "historical_studio30", "sdk_reference30", "selected"):
        row = indexed.get(arm, {})
        scores = row.get("official_scores")
        if row.get("status") == "completed" and isinstance(scores, dict):
            result.append({"arm": arm, "scores": scores})
    return result


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for family in ("Malgun Gothic", "맑은 고딕", "NanumGothic"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams["font.family"] = family
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    return plt


def _save(fig, figures_dir: Path, stem: str) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = figures_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(str(path.resolve()))
    return outputs


def _empty(ax, message="실측값 대기 중"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#777777")
    ax.set_xticks([])
    ax.set_yticks([])


def _load_live_events(tracking_dir: Path) -> list[tuple[str, list[dict]]]:
    result = []
    runs = tracking_dir / "runs"
    if not runs.exists():
        return result
    for path in sorted(runs.glob("*/sanitized-events.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result.append((path.parent.name, rows))
    return result


def time_vram_series(historical: dict, candidates: list[dict],
                     live_runs: list[tuple[str, list[dict]]],
                     live_statuses: dict[str, dict]) -> dict[str, list]:
    durations = {}
    vrams = {}
    boundary_duration = historical.get("metadata", {}).get("source_duration_seconds")
    for name, events in live_runs:
        peak = [row["peak_gpu_allocated_bytes"] for row in events
                if isinstance(row.get("peak_gpu_allocated_bytes"), (int, float))]
        if peak:
            vrams[name] = max(peak) / 2**30
    for name, status in live_statuses.items():
        if status.get("status") == "completed" and not status.get("diagnostic") and isinstance(status.get("train_seconds"), (int, float)):
            durations[name] = status["train_seconds"]
    for row in sorted(candidates, key=lambda item: item.get("candidate_index", -1)):
        label = row.get("name") or f"후보 {row.get('candidate_index')}"
        if row.get("status") == "completed" and isinstance(row.get("train_seconds"), (int, float)):
            durations[label] = row["train_seconds"]
        if isinstance(row.get("peak_gpu_allocated_bytes"), (int, float)):
            vrams[label] = row["peak_gpu_allocated_bytes"] / 2**30
    return {
        "duration_labels": list(durations), "duration_seconds": list(durations.values()),
        "vram_labels": list(vrams), "vram_gib": list(vrams.values()),
        "historical_total_seconds": boundary_duration if isinstance(boundary_duration, (int, float)) else None,
        "historical_train_seconds": None,
    }


def _load_live_statuses(tracking_dir: Path) -> dict[str, dict]:
    result = {}
    runs_dir = tracking_dir / "runs"
    if not runs_dir.exists():
        return result
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        status_path = run_dir / "source-status.json"
        if status_path.exists():
            result[run_dir.name] = read_json(status_path, {})
    return result


def plot_training_curves(plt, tracking_dir: Path, figures_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    historical = read_json(tracking_dir / "historical-studio30.json", {})
    train_events = historical.get("train_events", [])
    eval_events = historical.get("eval_events", [])
    if train_events:
        axes[0].plot([row["optimizer_step"] for row in train_events], [row["loss"] for row in train_events],
                     color="#6B7280", label="과거 Studio 30step 학습 loss")
    for name, events in _load_live_events(tracking_dir):
        points = [row for row in events if "loss" in row]
        if points:
            axes[0].plot([row["optimizer_step"] for row in points], [row["loss"] for row in points], label=name)
    if axes[0].lines:
        axes[0].legend(frameon=False)
        axes[0].set_xlabel("optimizer step")
        axes[0].set_ylabel("train loss")
    else:
        _empty(axes[0])
    axes[0].set_title("학습 곡선")
    if eval_events:
        axes[1].plot([row["optimizer_step"] for row in eval_events], [row["eval_loss"] for row in eval_events],
                     marker="o", color="#7C3AED", label="과거 전체 validation")
        axes[1].legend(frameon=False)
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("eval loss")
    else:
        _empty(axes[1], "과거 validation loss 없음")
    axes[1].set_title("과거 Studio 평가 loss\n새 search 공식 점수와 다른 지표")
    fig.suptitle("Qwen3.5-4B 학습 기록")
    fig.tight_layout()
    outputs = _save(fig, figures_dir, "training-curves")
    plt.close(fig)
    return outputs


def plot_candidate_curve(plt, tracking_dir: Path, figures_dir: Path) -> list[str]:
    series = candidate_best_series(read_json(tracking_dir / "candidates.json", []),
                                   read_json(tracking_dir / "search-baseline.json", None))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if series["candidate_index"]:
        ax.plot(series["candidate_index"], series["exact_match"], marker="o", label="SDK 기준·후보 EM")
        ax.plot(series["candidate_index"], series["best_exact_match"], marker="s", label="현재 최고 EM")
        ax.plot(series["candidate_index"], series["rouge_w"], marker="^", label="SDK 기준·후보 ROUGE-W")
        ax.set_xlabel("candidate index")
        ax.set_ylabel("공식 점수 (0–100)")
        ax.legend(frameon=False)
    else:
        _empty(ax, "공식 search 평가 점수 대기 중\n가짜 점수는 표시하지 않음")
    ax.set_title("후보별 EM·ROUGE-W와 EM best curve")
    fig.tight_layout()
    outputs = _save(fig, figures_dir, "candidate-best-curve")
    plt.close(fig)
    return outputs


def plot_time_vram(plt, tracking_dir: Path, figures_dir: Path) -> list[str]:
    rows = read_json(tracking_dir / "candidates.json", [])
    historical = read_json(tracking_dir / "historical-studio30.json", {})
    series = time_vram_series(historical, rows, _load_live_events(tracking_dir), _load_live_statuses(tracking_dir))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if series["duration_seconds"]:
        axes[0].bar(series["duration_labels"], series["duration_seconds"], color="#111827")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_ylabel("초")
    else:
        _empty(axes[0])
    axes[0].set_title("학습 호출 시간 (train_seconds)")
    axes[0].text(0.02, 0.96,
                 "Studio 전체 실행 시간은 이 축에 넣지 않습니다.\nSDK 학습 호출에는 첫 실행의 컴파일이 포함될 수 있습니다.",
                 transform=axes[0].transAxes, va="top", fontsize=10, color="#6B7280")
    if series["vram_gib"]:
        axes[1].bar(series["vram_labels"], series["vram_gib"], color="#2563EB")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_ylabel("GiB")
    else:
        _empty(axes[1], "학습 프로세스 VRAM 실측값 대기 중")
    axes[1].set_title("peak GPU allocated memory")
    fig.suptitle("평가 포함 범위가 달라 학습 속도 배수로 비교하지 않습니다", fontsize=13, fontweight="bold")
    fig.tight_layout()
    outputs = _save(fig, figures_dir, "time-vram")
    plt.close(fig)
    return outputs


def plot_arm_scores(plt, tracking_dir: Path, figures_dir: Path) -> list[str]:
    arms = arm_score_series(read_json(tracking_dir / "arms.json", []))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if arms:
        labels = [ARM_LABELS[row["arm"]] for row in arms]
        overall_em = [row["scores"]["overall"]["exact_match"] for row in arms]
        overall_rouge = [row["scores"]["overall"]["rouge_w"] for row in arms]
        import numpy as np
        x = np.arange(len(labels))
        axes[0].bar(x - 0.18, overall_em, 0.36, label="EM")
        axes[0].bar(x + 0.18, overall_rouge, 0.36, label="ROUGE-W")
        axes[0].set_xticks(x, labels, rotation=20)
        axes[0].set_ylabel("공식 점수 (0–100)")
        axes[0].legend(frameon=False)
        width = 0.22
        for offset, kind in zip((-width, 0, width), ("1", "2", "3")):
            axes[1].bar(x + offset, [row["scores"]["by_question_type"][kind]["exact_match"] for row in arms],
                        width, label=f"유형 {kind} EM")
        axes[1].set_xticks(x, labels, rotation=20)
        axes[1].set_ylabel("공식 EM (0–100)")
        axes[1].legend(frameon=False)
    else:
        _empty(axes[0], "selection release 뒤 4개 모델 점수 표시")
        _empty(axes[1], "유형별 실제 점수 대기 중")
    axes[0].set_title("4개 모델 전체 점수")
    axes[1].set_title("질문 유형별 EM")
    fig.tight_layout()
    outputs = _save(fig, figures_dir, "arm-scores")
    plt.close(fig)
    return outputs


def render(tracking_dir: Path, figures_dir: Path) -> list[str]:
    from .tracking import validated_scores, sha256
    baseline = read_json(tracking_dir / "search-baseline.json", None)
    records = ([baseline] if baseline else []) + read_json(tracking_dir / "candidates.json", []) + read_json(tracking_dir / "arms.json", [])
    for record in records:
        if record.get("official_scores"):
            source = Path(record["evidence_path"])
            _, verified = validated_scores(source)
            if sha256(source) != record["scores_sha256"] or verified != record["official_scores"]:
                raise ValueError("Tracked scores differ from their sealed evidence")
    plt = _matplotlib()
    outputs = []
    outputs.extend(plot_training_curves(plt, tracking_dir, figures_dir))
    outputs.extend(plot_candidate_curve(plt, tracking_dir, figures_dir))
    outputs.extend(plot_time_vram(plt, tracking_dir, figures_dir))
    outputs.extend(plot_arm_scores(plt, tracking_dir, figures_dir))
    return outputs


def main(argv=None) -> int:
    from .tracking import safe_name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-dir", type=Path, default=Path("outputs/tracking"))
    parser.add_argument("--group", type=safe_name, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = render(args.tracking_dir / args.group, args.figures_dir)
    print(json.dumps({"status": "completed", "figures": outputs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
