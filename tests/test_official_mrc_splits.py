import copy
from pathlib import Path
import tempfile
import unittest

from rehearsal import official_mrc_splits as splits


def row(i, context=None, source="", title=""):
    return {"id": str(i), "context": context or f"context {i}", "question": f"q{i}",
            "question_type": i % 3 + 1, "answers": [], "source": source, "title": title}


class SplitTests(unittest.TestCase):
    def test_partition_is_reproducible_stratified_and_complete(self):
        rows = [row(i) for i in range(300)]
        manifest = splits.build_manifest([], rows)
        self.assertEqual(manifest, splits.build_manifest([], list(reversed(rows))))
        search, final = (manifest["splits"][name] for name in ("search", "final"))
        self.assertFalse(set(search["ids"]) & set(final["ids"]))
        self.assertEqual(set(search["ids"]) | set(final["ids"]), {r["id"] for r in rows})
        for count in search["question_types"].values():
            self.assertLessEqual(abs(count - 20), 1)

    def test_transitive_context_article_links_through_train_stay_together(self):
        train = [row(1000, "shared context", "source", "article")]
        val = [row(1, "shared  context"), row(2, "different", "source", "article")]
        val += [row(i) for i in range(3, 60)]
        manifest = splits.build_manifest(train, val)
        self.assertEqual(manifest["groups"]["1"], manifest["groups"]["2"])
        for part in manifest["splits"].values():
            self.assertEqual("1" in part["ids"], "2" in part["ids"])
        self.assertEqual(manifest["overlap_with_train"]["connected_component_rows"], 2)

    def test_empty_article_metadata_does_not_join_every_row(self):
        manifest = splits.build_manifest([], [row(i) for i in range(30)])
        self.assertEqual(len(set(manifest["groups"].values())), 30)

    def test_mutated_data_partition_or_group_is_rejected(self):
        rows = [row(i) for i in range(60)]
        manifest = splits.build_manifest([], rows)
        changed = copy.deepcopy(rows)
        changed[0]["question"] = "tampered"
        with self.assertRaises(ValueError):
            splits.validate_manifest(manifest, changed)
        changed = copy.deepcopy(manifest)
        changed["splits"]["search"]["ids"].append(changed["splits"]["final"]["ids"][0])
        with self.assertRaises(ValueError):
            splits.validate_manifest(changed, rows)
        changed = copy.deepcopy(manifest)
        a = changed["splits"]["search"]["ids"][0]
        b = changed["splits"]["final"]["ids"][0]
        changed["groups"][b] = changed["groups"][a]
        with self.assertRaises(ValueError):
            splits.validate_manifest(changed, rows)

    def test_final_requires_matching_release_and_fixed_manifest_hash(self):
        rows = [row(i) for i in range(60)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.json"
            splits.task.write_json(path, splits.build_manifest([], rows))
            digest = splits.task.digest(path)
            with self.assertRaises(ValueError):
                splits.load_selection(path, rows, "search", "wrong")
            with self.assertRaises(ValueError):
                splits.load_selection(path, rows, "final", digest)
            release = Path(temp) / "release.json"
            splits.task.write_json(release, {"selection_complete": True, "split_manifest_sha256": digest})
            selected, metadata = splits.load_selection(path, rows, "final", digest, release)
            self.assertTrue(selected)
            self.assertEqual(metadata["split"], "final")


if __name__ == "__main__":
    unittest.main()
