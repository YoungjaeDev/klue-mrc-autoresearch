import builtins
import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from autoresearch_lab import tracking, plotting
from rehearsal import official_mrc


def sealed_evaluation(directory, em=60.0, rouge=65.0, split="search", adapter=True, diagnostic=False):
    directory.mkdir()
    selection = {"split": split, "diagnostic_prefix": diagnostic, "split_manifest_sha256": "fixed"}
    record = {"conditions": {"metric_sha256": official_mrc.METRIC_SHA256, "selection": selection},
              "adapter": {"files": {"adapter.safetensors": "fixed"}} if adapter else None}
    tracking.write_json(directory / "run.json", record)
    (directory / "raw_predictions.jsonl").write_text('{"prediction":"private fixture"}\n', encoding="utf-8")
    group = {"count": 1, "exact_match": em, "rouge_w": rouge}
    scores = {"overall": {**group, "count": 3}, "by_question_type": {str(i): group for i in (1, 2, 3)},
              "selection": selection, "run_sha256": tracking.sha256(directory / "run.json"),
              "predictions_sha256": tracking.sha256(directory / "raw_predictions.jsonl")}
    tracking.write_json(directory / "scores.json", scores)
    tracking.write_json(directory / "status.json", {"status": "completed", "scores_sha256": tracking.sha256(directory / "scores.json")})
    return directory / "scores.json"


def score_args(root, path, kind="reference", name="reference", **extra):
    args = tracking.parser().parse_args(["score", "--tracking-dir", str(root / "tracking"),
        "--group", "example", "--run-name", name, "--kind", kind, "--scores", str(path)])
    for key, value in extra.items():
        setattr(args, key, value)
    return args


class FakeRun:
    id = "fake-run"
    def __init__(self, fail_on=None):
        self.logged, self.metrics = [], []
        self.fail_on = fail_on
    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))
    def log(self, payload, **kwargs):
        if len(self.logged) + 1 == self.fail_on:
            raise ConnectionError("a secret that must not be reported")
        self.logged.append(payload)


class TrackingTests(unittest.TestCase):
    def test_local_collect_does_not_import_wandb_and_preserves_only_allowed_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            event = {"event_index": 0, "event": "step", "optimizer_step": 1, "loss": .3, "raw_prompt": "private"}
            (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
            tracking.write_json(root / "status.json", {"status": "completed", "train_seconds": 8, "adapter_path": "private"})
            args = tracking.parser().parse_args(["collect", "--group", "example", "--run-name", "reference",
                "--tracking-dir", str(root / "tracking"), "--events", str(root / "events.jsonl"), "--status", str(root / "status.json")])
            original_import = builtins.__import__
            def guarded(name, *a, **kw):
                if name.startswith("wandb"):
                    raise AssertionError("Local-only must not import W&B")
                return original_import(name, *a, **kw)
            with patch("builtins.__import__", side_effect=guarded):
                self.assertEqual(tracking.collect(args)["status"], "local_only")
            run = root / "tracking/example/runs/reference"
            self.assertNotIn("private", (run / "sanitized-events.jsonl").read_text())
            self.assertNotIn("adapter_path", (run / "source-status.json").read_text())

    def test_missing_key_keeps_online_request_local(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            args = SimpleNamespace(online=True, env_file=Path(temp) / "absent")
            run, result = tracking._online_run(args, Path(temp), {})
            self.assertIsNone(run)
            self.assertEqual(result, {"status": "local_only", "reason": "MissingCredential"})

    def test_network_error_preserves_mirror_and_first_unsent_cursor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "events.jsonl"
            path.write_text("".join(json.dumps({"event_index": i, "event": "step", "optimizer_step": i+1, "loss": .3}) + "\n" for i in range(2)))
            result = tracking.sync_live_events(path, root / "cursor.json", root / "mirror.jsonl", FakeRun(fail_on=2), 1)
            self.assertEqual(result["status"], "online_failed")
            self.assertEqual(result["next_event_index"], 1)
            self.assertNotIn("secret", json.dumps(result))
            self.assertEqual(len((root / "mirror.jsonl").read_text().splitlines()), 2)
            run = FakeRun()
            tracking.sync_live_events(path, root / "cursor.json", root / "mirror.jsonl", run, 1)
            tracking.sync_live_events(path, root / "cursor.json", root / "mirror.jsonl", run, 1)
            self.assertEqual(len(run.logged), 1)

    def test_score_hash_chain_detects_tampered_score_run_and_prediction(self):
        for name in ("scores.json", "run.json", "raw_predictions.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path = sealed_evaluation(Path(temp) / "eval")
                tracking.validated_scores(path)
                with path.with_name(name).open("a") as stream:
                    stream.write(" ")
                with self.assertRaises(ValueError):
                    tracking.validated_scores(path)

    def test_reference_initializes_best_and_tied_rouge_cannot_keep(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = tracking.score_record(score_args(root, sealed_evaluation(root / "ref")))
            path = sealed_evaluation(root / "candidate", em=58, rouge=70)
            args = score_args(root, path, "candidate", "candidate-01", candidate_index=1, decision="discard")
            candidate = tracking.score_record(args)
            series = plotting.candidate_best_series([candidate], baseline)
            self.assertEqual(series["candidate_index"], [0, 1])
            self.assertEqual(series["best_exact_match"], [60, 60])
            tied = sealed_evaluation(root / "tied", em=60, rouge=80)
            args = score_args(root, tied, "candidate", "candidate-02", candidate_index=2, decision="keep")
            with self.assertRaisesRegex(ValueError, "strict EM"):
                tracking.score_record(args)
            args.decision = "discard"
            tracking.score_record(args)

    def test_score_network_failure_keeps_verified_local_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = score_args(root, sealed_evaluation(root / "eval"))
            with patch.object(tracking, "_online_run", side_effect=ConnectionError("private credential")):
                result = tracking.score(args)
            self.assertEqual(result["status"], "online_failed")
            self.assertTrue((root / "tracking/example/search-baseline.json").exists())
            self.assertNotIn("private credential", json.dumps(result))

    def test_no_baseline_no_curve_and_missing_scores_no_fake_points(self):
        candidate = {"candidate_index": 1, "status": "completed", "official_scores": {"overall": {"exact_match": 50, "rouge_w": 60}}}
        with self.assertRaises(ValueError):
            plotting.candidate_best_series([candidate])
        self.assertEqual(plotting.candidate_best_series([])["exact_match"], [])

    def test_training_time_never_uses_or_overwrites_with_total_duration(self):
        series = plotting.time_vram_series({"metadata": {"source_duration_seconds": 1400}},
            [{"candidate_index": 1, "name": "candidate-01", "status": "completed", "duration_seconds": 300, "train_seconds": 150}],
            [], {"reference": {"status": "completed", "train_seconds": 140, "duration_seconds": 280}})
        self.assertEqual(series["duration_seconds"], [140, 150])
        self.assertEqual(series["historical_total_seconds"], 1400)
        self.assertIsNone(series["historical_train_seconds"])

    def test_metadata_and_names_cannot_include_arbitrary_fields_or_escape_group(self):
        self.assertEqual(tracking.filter_metadata({"train_sha256": "hash", "WANDB_API_KEY": "secret", "raw_config": {}}), {"train_sha256": "hash"})
        for name in ("../other", "a/b", "a\\b", ".."):
            with self.assertRaises(ValueError):
                tracking.safe_name(name)

    def test_diagnostic_scores_cannot_initialize_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = score_args(root, sealed_evaluation(root / "diagnostic", diagnostic=True))
            with self.assertRaisesRegex(ValueError, "Diagnostic prefix"):
                tracking.score_record(args)

    def test_online_settings_disable_automatic_uploads_and_require_private_project(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WANDB_API_KEY": "test-placeholder"}):
            root = Path(temp)
            client = SimpleNamespace(execute=Mock(return_value={"project": {"access": "PRIVATE"}}))
            fake = SimpleNamespace(Api=Mock(return_value=SimpleNamespace(client=client)),
                Settings=Mock(side_effect=lambda **kwargs: kwargs), init=Mock(return_value=FakeRun()))
            args = SimpleNamespace(online=True, env_file=None, entity="example", project="private-example", group="group", run_name="reference")
            with patch.dict(sys.modules, {"wandb": fake, "wandb_gql": SimpleNamespace(gql=lambda q: q)}):
                tracking._online_run(args, root, {"raw_prompt": "private", "model_revision": "revision"})
                settings = fake.init.call_args.kwargs["settings"]
                for key in ("disable_code", "disable_git", "x_disable_stats", "x_disable_meta", "x_disable_machine_info"):
                    self.assertTrue(settings[key])
                self.assertFalse(settings["save_code"])
                self.assertFalse(settings["x_save_requirements"])
                self.assertNotIn("raw_prompt", fake.init.call_args.kwargs["config"])
                client.execute.return_value = {"project": {"access": "PUBLIC"}}
                with self.assertRaises(ValueError):
                    tracking._online_run(args, root, {})
                self.assertEqual(fake.init.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Install tracking extra for PNG/SVG integration")
    def test_render_outputs_and_rejects_local_score_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = tracking.score_record(score_args(root, sealed_evaluation(root / "reference")))
            path = sealed_evaluation(root / "candidate", em=58)
            tracking.score_record(score_args(root, path, "candidate", "candidate-01", candidate_index=1, decision="discard"))
            files = plotting.render(root / "tracking/example", root / "figures")
            self.assertEqual(len(files), 8)
            self.assertTrue(all(Path(p).stat().st_size > 1000 for p in files))
            baseline["official_scores"]["overall"]["exact_match"] = 99
            tracking.write_json(root / "tracking/example/search-baseline.json", baseline)
            with self.assertRaisesRegex(ValueError, "sealed evidence"):
                plotting.render(root / "tracking/example", root / "figures2")


if __name__ == "__main__":
    unittest.main()
