"""CPU-only unit tests for JSONL resume integrity hardening.

Covers:
  - utils/jsonl_integrity.py: inspect_jsonl() strict per-line validation
  - pred_long_bench.py: resolve_dataset_resume_plan() fail-closed resume
  - scripts/repair_partial_jsonl.py: tail-only corruption repair, with a
    hard refusal on middle corruption and a byte-verified backup

No GPU, no model download, no dataset download, no network access.

Run with:
    CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m unittest \
        tests.test_jsonl_resume_integrity -v
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pred_long_bench as plb  # noqa: E402
from utils.jsonl_integrity import inspect_jsonl  # noqa: E402
from scripts import repair_partial_jsonl as repair  # noqa: E402


def row(i):
    return json.dumps({"pred": f"answer {i}", "answers": ["a"], "all_classes": None, "length": 10})


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, content_bytes):
        path = os.path.join(self.tmp_dir, name)
        with open(path, "wb") as f:
            f.write(content_bytes)
        return path


class TestCleanPartial(TempDirCase):
    """Test 1 -- a fully valid partial file resumes from its true row count."""

    def test_clean_partial_resumes_at_valid_row_count(self):
        content = (row(0) + "\n" + row(1) + "\n" + row(2) + "\n").encode("utf-8")
        path = self.write("task.jsonl.partial", content)

        info = inspect_jsonl(path)
        self.assertEqual(info.valid_rows, 3)
        self.assertEqual(info.invalid_rows, 0)
        self.assertEqual(info.nonempty_rows, 3)

        out_path = os.path.join(self.tmp_dir, "task.jsonl")
        action, active_path, done = plb.resolve_dataset_resume_plan(
            "task", out_path, path, expected=10
        )
        self.assertEqual(action, "resume")
        self.assertEqual(active_path, path)
        self.assertEqual(done, 3)


class TestNulTailCorruption(TempDirCase):
    """Test 2 -- a NUL-filled tail (as observed in the real K4/V16 incident)
    must never be silently counted as a completed row."""

    def test_nul_tail_fails_closed_and_is_repairable(self):
        content = (row(0) + "\n" + row(1) + "\n" + row(2) + "\n").encode("utf-8")
        content += b"\x00" * 76  # no trailing newline, mirrors the real corruption
        path = self.write("hotpotqa.jsonl.partial", content)

        info = inspect_jsonl(path)
        self.assertEqual(info.valid_rows, 3)
        self.assertEqual(info.invalid_rows, 1)
        self.assertTrue(info.corruption_is_tail_only)
        self.assertEqual(info.valid_prefix_rows, 3)

        out_path = os.path.join(self.tmp_dir, "hotpotqa.jsonl")
        with self.assertRaises(RuntimeError):
            plb.resolve_dataset_resume_plan("hotpotqa", out_path, path, expected=200)

        plan = repair.plan_repair(path)
        self.assertTrue(plan["repairable"])
        self.assertEqual(plan["planned_retained_rows"], 3)
        self.assertEqual(plan["planned_removed_invalid_rows"], 1)


class TestTruncatedJsonTail(TempDirCase):
    """Test 3 -- a mid-write truncated JSON object (crash before the closing
    brace/newline) is the same class of tail corruption as a NUL fill."""

    def test_truncated_tail_fails_closed_and_is_repairable(self):
        content = (row(0) + "\n" + row(1) + "\n" + row(2) + "\n").encode("utf-8")
        content += b'{"pred": "abc'
        path = self.write("task.jsonl.partial", content)

        out_path = os.path.join(self.tmp_dir, "task.jsonl")
        with self.assertRaises(RuntimeError):
            plb.resolve_dataset_resume_plan("task", out_path, path, expected=10)

        plan = repair.plan_repair(path)
        self.assertTrue(plan["repairable"])
        self.assertEqual(plan["planned_retained_rows"], 3)


class TestMiddleCorruptionRefused(TempDirCase):
    """Test 4 -- corruption sandwiched between valid rows must never be
    auto-repaired: the tool cannot guess which sample index the surviving
    valid row belongs to."""

    def test_middle_corruption_is_refused(self):
        content = (row(0) + "\n" + "{not json\n" + row(2) + "\n").encode("utf-8")
        path = self.write("task.jsonl.partial", content)

        info = inspect_jsonl(path)
        self.assertFalse(info.corruption_is_tail_only)

        plan = repair.plan_repair(path)
        self.assertFalse(plan["repairable"])
        self.assertIn("middle corruption", plan["reason"])

        exit_code, applied_plan = repair.apply_repair(
            path, os.path.join(self.tmp_dir, "backups")
        )
        self.assertNotEqual(exit_code, 0)
        # Refusal must not touch the original file at all.
        with open(path, "rb") as f:
            self.assertEqual(f.read(), content)


class TestValidFinalFileSkips(TempDirCase):
    """Test 5 -- a complete, fully valid final *.jsonl is skipped."""

    def test_valid_final_file_is_skipped(self):
        content = (row(0) + "\n" + row(1) + "\n" + row(2) + "\n").encode("utf-8")
        out_path = self.write("task.jsonl", content)
        partial_path = out_path + ".partial"

        action, active_path, done = plb.resolve_dataset_resume_plan(
            "task", out_path, partial_path, expected=3
        )
        self.assertEqual(action, "skip")
        self.assertEqual(done, 3)


class TestCorruptFinalFileFailsClosed(TempDirCase):
    """Test 6 -- a corrupted *final* file must never be silently skipped."""

    def test_corrupt_final_file_raises(self):
        content = (row(0) + "\n" + row(1) + "\n" + "not json at all\n").encode("utf-8")
        out_path = self.write("task.jsonl", content)
        partial_path = out_path + ".partial"

        with self.assertRaises(RuntimeError):
            plb.resolve_dataset_resume_plan("task", out_path, partial_path, expected=3)


class TestRepairNulTail(TempDirCase):
    """Test 7 -- after --apply repair, only the valid prefix remains."""

    def test_repair_removes_nul_tail_cleanly(self):
        content = (row(0) + "\n" + row(1) + "\n" + row(2) + "\n").encode("utf-8")
        content += b"\x00" * 76
        path = self.write("hotpotqa.jsonl.partial", content)
        backup_dir = os.path.join(self.tmp_dir, "backups")

        original_sha = sha256_bytes(content)
        exit_code, plan = repair.apply_repair(path, backup_dir)
        self.assertEqual(exit_code, 0)

        with open(path, "rb") as f:
            repaired = f.read()
        self.assertEqual(repaired.count(b"\x00"), 0)
        self.assertTrue(repaired.endswith(b"\n"))

        info = inspect_jsonl(path)
        self.assertEqual(info.valid_rows, 3)
        self.assertEqual(info.invalid_rows, 0)

        repaired_sha = sha256_bytes(repaired)
        self.assertNotEqual(original_sha, repaired_sha)
        self.assertEqual(plan["original_sha256"], original_sha)
        self.assertEqual(plan["repaired_sha256"], repaired_sha)


class TestBackupCorrectness(TempDirCase):
    """Test 8 -- the backup must be byte-identical to the pre-repair original."""

    def test_backup_sha256_matches_pre_repair_original(self):
        content = (row(0) + "\n" + row(1) + "\n").encode("utf-8") + b"\x00" * 20
        path = self.write("task.jsonl.partial", content)
        backup_dir = os.path.join(self.tmp_dir, "backups")

        original_sha = sha256_bytes(content)
        exit_code, plan = repair.apply_repair(path, backup_dir)
        self.assertEqual(exit_code, 0)

        with open(plan["backup_path"], "rb") as f:
            backup_bytes = f.read()
        self.assertEqual(sha256_bytes(backup_bytes), original_sha)
        self.assertTrue(os.path.exists(plan["manifest_path"]))
        with open(plan["manifest_path"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["original_sha256"], original_sha)


class TestRepairIdempotency(TempDirCase):
    """Test 9 -- repairing an already-clean file must be a no-op, not an
    unnecessary rewrite."""

    def test_clean_file_is_not_rewritten(self):
        content = (row(0) + "\n" + row(1) + "\n").encode("utf-8")
        path = self.write("task.jsonl.partial", content)
        mtime_before = os.stat(path).st_mtime_ns

        exit_code, plan = repair.apply_repair(
            path, os.path.join(self.tmp_dir, "backups")
        )
        self.assertNotEqual(exit_code, 0)
        self.assertTrue(plan.get("already_clean"))

        with open(path, "rb") as f:
            self.assertEqual(f.read(), content)
        self.assertEqual(os.stat(path).st_mtime_ns, mtime_before)
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "backups")))


class TestEmptyFile(TempDirCase):
    """Test 10 -- an empty partial must be handled explicitly, not crash."""

    def test_empty_file_does_not_crash(self):
        path = self.write("task.jsonl.partial", b"")

        info = inspect_jsonl(path)
        self.assertTrue(info.exists)
        self.assertEqual(info.valid_rows, 0)
        self.assertEqual(info.invalid_rows, 0)
        self.assertEqual(info.nonempty_rows, 0)
        self.assertTrue(info.ends_with_newline)

        out_path = os.path.join(self.tmp_dir, "task.jsonl")
        action, active_path, done = plb.resolve_dataset_resume_plan(
            "task", out_path, path, expected=10
        )
        self.assertEqual(action, "start")
        self.assertEqual(done, 0)

        plan = repair.plan_repair(path)
        self.assertFalse(plan["repairable"])
        self.assertTrue(plan.get("already_clean"))


if __name__ == "__main__":
    unittest.main()
