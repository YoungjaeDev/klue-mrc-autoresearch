import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoresearch_lab.prepare import verify_message, download_file, runtime_freeze, FROZEN_RUNTIME_FILES
from autoresearch_lab.run import verify_files
from rehearsal import official_mrc as task


class PublicSourceContractTests(unittest.TestCase):
    def test_runtime_freeze_detects_supervisor_and_program_changes_but_allows_train_edits(self):
        required = {"autoresearch_lab/bootstrap.py", "autoresearch_lab/run.py",
                    "autoresearch_lab/windows_runtime.py", "runtime/windows_store/sitecustomize.py",
                    "runtime/windows_store/launch.py", "runtime/windows_store/dispatch.py",
                    "program.md", "pyproject.toml", "uv.lock"}
        self.assertTrue(required <= set(FROZEN_RUNTIME_FILES))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in FROZEN_RUNTIME_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            (root / "train.py").write_text("initial recipe")
            frozen = runtime_freeze(root)
            (root / "train.py").write_text("candidate recipe")
            verify_files(frozen)
            for relative in ("autoresearch_lab/run.py", "program.md"):
                path = root / relative
                path.write_text("changed")
                with self.assertRaises(ValueError):
                    verify_files(frozen)
                path.write_text("original")

    def test_download_checks_hash_and_reuses_verified_file_without_network(self):
        payload = b"public data"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
                download_file("https://example.invalid/pinned", path, digest)
            with patch("urllib.request.urlopen", side_effect=AssertionError("No network for cached file")):
                self.assertEqual(download_file("unused", path, digest).read_bytes(), payload)
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                download_file("unused", path, digest)

    def test_failed_checksum_is_not_promoted_and_previous_partial_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            partial = path.with_suffix(".jsonl.partial")
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaises(ValueError):
                    download_file("https://example.invalid/pinned", path, "wrong hash")
            self.assertFalse(path.exists())
            self.assertFalse(partial.exists())
            partial.write_bytes(b"another process owns this")
            with self.assertRaises(FileExistsError):
                download_file("unused", path)
            self.assertEqual(partial.read_bytes(), b"another process owns this")

    def test_original_roles_prompt_and_short_answer_must_match(self):
        original = {"id": "id", "context": "서울은 도시다", "question": "어느 도시?", "answers": ["서울"], "is_impossible": False}
        row = {"id": "id", "messages": task.messages_for(original) + [{"role": "assistant", "content": "서울"}]}
        verify_message(row, original)
        for field in ("role", "content"):
            changed = copy.deepcopy(row)
            changed["messages"][2][field] = "tampered"
            with self.assertRaises(ValueError):
                verify_message(changed, original)
        row["secret"] = "unexpected extra field"
        with self.assertRaises(ValueError):
            verify_message(row, original)

    def test_unanswerable_never_uses_a_plausible_answer(self):
        original = {"id": "id", "context": "지문", "question": "질문", "answers": [], "is_impossible": True}
        row = {"id": "id", "messages": task.messages_for(original) + [{"role": "assistant", "content": task.NO_ANSWER}]}
        verify_message(row, original)
        row["messages"][-1]["content"] = "plausible but wrong"
        with self.assertRaises(ValueError):
            verify_message(row, original)


if __name__ == "__main__":
    unittest.main()
