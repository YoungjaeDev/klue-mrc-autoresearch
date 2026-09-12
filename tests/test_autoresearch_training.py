import json
from pathlib import Path
import tempfile
import unittest

from autoresearch_lab.train import EventWriter, load_contract, validate_recipe, DEFAULT_RECIPE, prepare_texts, validate_trainable_parameters


class TrainingContractTests(unittest.TestCase):
    def test_reference_target_count_is_checked_but_candidates_can_change_rank(self):
        parameters = [{"name": "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight", "numel": 100}]
        with self.assertRaises(ValueError):
            validate_trainable_parameters(parameters, "reference")
        self.assertEqual(validate_trainable_parameters(parameters, "candidate")["total_parameters"], 100)
        parameters[0]["name"] = "base_model.model.visual.layers.0.up_proj.lora_A.default.weight"
        with self.assertRaises(ValueError):
            validate_trainable_parameters(parameters, "candidate")
    def test_prefix_is_counted_from_flat_ids_not_mapping_length(self):
        class Tokenizer:
            chat_template = "fixed"
            def apply_chat_template(self, messages, *, tokenize, **kwargs):
                if tokenize:
                    raise AssertionError("Never count len(BatchEncoding)")
                return "".join(row["content"] for row in messages)
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(char) for char in text], "attention_mask": [1] * len(text)}
            def get_vocab(self):
                return {"a": 1}
        result = prepare_texts(Tokenizer(), [{"messages": [{"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"}]}])
        self.assertEqual(result["prefix_lengths"], [8])
        self.assertEqual(result["max_tokens"], 14)
    def test_fixed_training_budget_and_explicit_smoke(self):
        self.assertEqual(validate_recipe(dict(DEFAULT_RECIPE), None)["max_steps"], 30)
        self.assertEqual(validate_recipe(dict(DEFAULT_RECIPE), 1)["max_steps"], 1)
        for updates in ({"max_steps": 31}, {"batch_size": 1}, {"seed": 1}, {"packing": True}):
            with self.assertRaises(ValueError):
                validate_recipe({**DEFAULT_RECIPE, **updates}, None)
        with self.assertRaises(ValueError):
            validate_recipe(dict(DEFAULT_RECIPE), 2)

    def test_events_allow_scalars_not_raw_text_or_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventWriter(path) as events:
                events.emit("step", 1, {"loss": .2, "learning_rate": .0002,
                    "raw_prompt": "private", "grad_norm": float("nan")})
                events.emit("step", 2, {"loss": .1})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["event_index"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["optimizer_step"], 1)
            self.assertNotIn("raw_prompt", rows[0])
            self.assertNotIn("grad_norm", rows[0])
            self.assertLessEqual(rows[0]["elapsed_seconds"], rows[1]["elapsed_seconds"])

    def test_training_manifest_rejects_changed_data_and_non_training_rows(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "train.jsonl"
            rows = [{"id": str(i), "messages": [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"}]} for i in range(1024)]
            data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"train_path": str(data),
                "train_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "train_rows": 1024, "seed": 3407, "model_id": "Qwen/Qwen3.5-4B",
                "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}))
            self.assertEqual(len(load_contract(manifest)[1]), 1024)
            data.write_text(data.read_text() + "\n ", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_contract(manifest)


if __name__ == "__main__":
    unittest.main()
