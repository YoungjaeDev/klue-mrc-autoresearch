import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rehearsal import official_mrc as task
from rehearsal import official_mrc_runner as runner


class OfficialMRCMetricTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "a", "question_type": 1, "is_impossible": False, "answers": ["서울", "서울시"], "context": "지문", "question": "질문"},
            {"id": "b", "question_type": 2, "is_impossible": False, "answers": ["가나마라"], "context": "지문", "question": "질문"},
            {"id": "c", "question_type": 3, "is_impossible": True, "answers": [], "context": "지문", "question": "질문"},
        ]

    def test_vendor_matches_pinned_official_source(self):
        task.verify_scorer()

    def test_official_rouge_uses_consecutive_sequence_not_character_bag_f1(self):
        em, rouge = task.official.compute_em_and_rouge_w_score_for_klue_mrc("가나다라", ["가나마라"])
        self.assertEqual(em, 0)
        self.assertEqual(rouge, 0.5)

    def test_full_pipeline_matches_official_function(self):
        predictions = [{"id": "a", "prediction": '"서울시"'}, {"id": "b", "prediction": "가나다라"}, {"id": "c", "prediction": task.NO_ANSWER}]
        report = task.score(self.rows, predictions)
        labels = [{"qid": row["id"], "qtype": row["question_type"], "ground_truth": row["answers"]} for row in self.rows]
        official = task.official.evaluate_for_klue_mrc(labels, {"a": '"서울시"', "b": "가나다라", "c": ""})
        self.assertEqual(report["overall"]["exact_match"], official["exact_match"] * 100)
        self.assertEqual(report["overall"]["rouge_w"], official["rouge"] * 100)
        self.assertEqual(report["by_question_type"]["3"]["exact_match"], 100)
        self.assertFalse(report["official_test_submission"])

    def test_output_mapping_never_uses_reference_labels(self):
        pred = [{"id": row["id"], "prediction": task.NO_ANSWER} for row in self.rows]
        scores = task.score(self.rows, pred)
        self.assertEqual(scores["by_question_type"]["1"]["exact_match"], 0)
        self.assertEqual(scores["by_question_type"]["3"]["exact_match"], 100)
        self.assertEqual(task.official_prediction("정답은 서울입니다"), "정답은 서울입니다")
        self.assertEqual(task.official_prediction("모르겠습니다"), "모르겠습니다")

    def test_actual_empty_string_is_correct_for_official_no_answer(self):
        result = task.score([self.rows[2]], [{"id": "c", "prediction": ""}])
        self.assertEqual(result["overall"]["exact_match"], 100)
        self.assertEqual(result["overall"]["rouge_w"], 100)
        self.assertEqual(result["empty_raw_responses"], 1)

    def test_missing_extra_duplicate_and_nonstring_predictions_fail(self):
        for rows, predictions in (
            ([self.rows[0], self.rows[2]], [{"id": "a", "prediction": "서울"}]),
            ([self.rows[0]], [{"id": "x", "prediction": "서울"}]),
            ([self.rows[0]], [{"id": "a", "prediction": "서울"}, {"id": "a", "prediction": "서울"}]),
            ([self.rows[0]], [{"id": "a", "prediction": None}]),
        ):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                task.score(rows, predictions)

    def test_prompt_has_no_assistant_target_or_reference_metadata(self):
        messages = task.messages_for(self.rows[0])
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "지문:\n지문\n\n질문: 질문")

    def test_loader_uses_answers_not_plausible_answers(self):
        rows = task.flatten({"data": [{"paragraphs": [{"context": "지문", "qas": [{
            "guid": "x", "question": "질문", "question_type": 3, "is_impossible": True,
            "answers": [], "plausible_answers": [{"text": "가짜 정답"}],
        }]}]}]})
        self.assertEqual(rows[0]["answers"], [])

    def test_selecting_a_subset_is_explicit_and_validated(self):
        self.assertEqual(runner.select_rows(self.rows, None), self.rows)
        self.assertEqual(runner.select_rows(self.rows, 1), self.rows[:1])
        for bad in (0, -1, 4):
            with self.assertRaises(ValueError):
                runner.select_rows(self.rows, bad)

    def test_named_split_cannot_fall_back_to_full_public_dev(self):
        args = argparse.Namespace(data_dir=task.DEFAULT_DATA, limit=None, split="final", split_manifest=None)
        with patch.object(task, "load_data", return_value=self.rows), self.assertRaises(ValueError):
            runner.evaluation_rows(args)


@unittest.skipUnless(os.environ.get("RUN_MODEL_INTEGRATION") == "1" and (task.DEFAULT_DATA / "official-dev.json").exists(), "Model integration is opt-in; CPU contracts never load models")
class OfficialMRCBackendTests(unittest.TestCase):
    """Real tiny CPU models exercise the same loader/generate/PEFT/scoring path."""

    @classmethod
    def setUpClass(cls):
        # Applies only to this test process, never to Studio or the system.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from peft import LoraConfig, get_peft_model
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.base = cls.root / "base-model"
        vocab = {"[EOS]": 0, "[UNK]": 1, "[PAD]": 2, "system": 3, "user": 4, "assistant": 5, "지문": 6, "질문": 7, "서울": 8}
        inner = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        inner.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]")
        tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=2, n_embd=16, n_positions=2048, bos_token_id=0, eos_token_id=0, pad_token_id=2))
        model.save_pretrained(cls.base)
        tokenizer.save_pretrained(cls.base)
        # Use the actual base path so the saved adapter carries valid provenance.
        loaded = GPT2LMHeadModel.from_pretrained(cls.base)
        peft = get_peft_model(loaded, LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM"))
        cls.adapter = cls.root / "adapter"
        peft.save_pretrained(cls.adapter)
        cls.merged = cls.root / "merged"
        peft.merge_and_unload().save_pretrained(cls.merged)
        tokenizer.save_pretrained(cls.merged)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, name, **updates):
        result = dict(model=str(self.base), adapter=None, revision=None, tokenizer=None,
            tokenizer_revision=None, device="cpu", dtype="float32", max_new_tokens=2,
            max_length=2048, data_dir=task.DEFAULT_DATA, limit=1, output=self.root / name)
        result.update(updates)
        return argparse.Namespace(**result)

    def test_real_base_lora_merged_generation_and_comparison(self):
        for name, args in (
            ("base-run", self.args("base-run")),
            ("sft-run", self.args("sft-run", adapter=str(self.adapter))),
            ("merged-run", self.args("merged-run", model=str(self.merged))),
        ):
            with self.subTest(name=name):
                report = runner.run(args)
                self.assertEqual(report["overall"]["count"], 1)
                self.assertFalse(report["full_dev"])
                self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "completed")
                self.assertEqual(len(task.read_jsonl(args.output / "raw_predictions.jsonl")), 1)
        comparison = runner.compare(self.root / "base-run", self.root / "sft-run")
        self.assertTrue(comparison["conditions_verified_equal"])
        self.assertFalse(comparison["full_dev"])
        multi = runner.compare_arms({"base": self.root / "base-run", "studio": self.root / "sft-run",
                                    "reference": self.root / "sft-run", "selected": self.root / "sft-run"})
        self.assertEqual(set(multi["arms"]), {"base", "studio", "reference", "selected"})
        self.assertFalse(multi["training_equivalence_asserted"])

    def test_modified_score_cannot_be_used_for_selection_or_comparison(self):
        args = self.args("tampered-score")
        runner.run(args)
        scores_path = args.output / "scores.json"
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        scores["overall"]["exact_match"] = 123
        scores_path.write_text(json.dumps(scores), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Score evidence"):
            runner.read_completed(args.output)

    def test_wrong_adapter_base_is_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "Adapter base"):
            runner.load_backend(self.args("unused", model="wrong/model", adapter=str(self.adapter)), [])

    def test_qwen35_conditional_model_and_lora_on_cpu(self):
        from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        config = Qwen3_5Config(
            text_config=dict(vocab_size=9, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                layer_types=["linear_attention", "full_attention"], max_position_embeddings=2048,
                linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
                rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
                bos_token_id=0, eos_token_id=0, pad_token_id=2),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                patch_size=2, temporal_patch_size=1, spatial_merge_size=1, out_hidden_size=32, num_position_embeddings=16),
        )
        qwen = self.root / "qwen-tiny"
        Qwen3_5ForConditionalGeneration(config).save_pretrained(qwen)
        AutoTokenizer.from_pretrained(self.base).save_pretrained(qwen)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(qwen)
        adapter = self.root / "qwen-adapter"
        get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type="CAUSAL_LM")).save_pretrained(adapter)
        for name, adapter_path in (("qwen-base-run", None), ("qwen-sft-run", str(adapter))):
            result = runner.run(self.args(name, model=str(qwen), adapter=adapter_path))
            self.assertEqual(result["overall"]["count"], 1)
            record = json.loads((self.root / name / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["conditions"]["model_class"], "image-text-to-text")
        self.assertTrue(runner.compare(self.root / "qwen-base-run", self.root / "qwen-sft-run")["conditions_verified_equal"])

    def test_adapter_passed_as_full_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "adapter directory"):
            runner.load_backend(self.args("unused", model=str(self.adapter)), [])

    def test_too_short_context_fails_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            runner.load_backend(self.args("unused", max_length=3), task.load_data()[:1])

    def test_quantized_gpu_configuration_is_not_silently_run_on_cpu(self):
        with self.assertRaisesRegex(ValueError, "quantized evaluation requires CUDA"):
            runner.load_backend(self.args("unused", load_in_4bit=True), [])


if __name__ == "__main__":
    unittest.main()
