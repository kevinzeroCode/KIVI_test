#!/usr/bin/env python
"""Explicitly repair a LongBench *.jsonl.partial file whose only damage is a
contiguous, tail-only run of invalid JSON lines (e.g. a NUL-filled tail left
by an abrupt host crash/freeze while a row was being written).

This tool NEVER runs automatically as part of pred_long_bench.py. It is a
separate, explicit, auditable step: dry-run by default, and --apply only
after a human has reviewed the dry-run output.

Safety rules (see also inspect_jsonl() in utils/jsonl_integrity.py):
  - The file must contain at least one valid JSON row.
  - Every invalid line must come after every valid line (contiguous valid
    prefix, tail-only corruption). If a valid row appears after an invalid
    row ("middle corruption"), the tool REFUSES -- it will not guess which
    rows correspond to which dataset samples.
  - --apply always makes a full byte-for-byte backup of the original file
    (with a recovery_manifest.json recording both SHA256s) before touching
    anything.
  - The repaired file is built in a sibling *.repair.tmp file, fsync'd, and
    re-validated (every line must parse and the row count must match the
    planned valid prefix) before it is atomically os.replace()'d over the
    original. If any check fails, the original partial is left untouched
    and the tool exits non-zero.

Usage:
    python scripts/repair_partial_jsonl.py --path <partial-path> --dry-run
    python scripts/repair_partial_jsonl.py --path <partial-path> --apply \\
        --backup-dir outputs/recovery_backups
"""
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jsonl_integrity import inspect_jsonl  # noqa: E402


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def read_lines_preserving_content(path):
    """Split the raw file the same way inspect_jsonl() does, so line numbers
    and valid/invalid classification line up exactly."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="surrogateescape")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def plan_repair(path):
    """Inspect `path` and decide whether tail-only repair is possible.

    Returns a dict describing the plan; does not touch the filesystem.
    """
    info = inspect_jsonl(path)
    plan = {
        "path": path,
        "exists": info.exists,
        "nonempty_rows": info.nonempty_rows,
        "valid_rows": info.valid_rows,
        "invalid_rows": info.invalid_rows,
        "first_invalid_line": info.first_invalid_line,
        "invalid_line_numbers": info.invalid_line_numbers,
        "valid_prefix_rows": info.valid_prefix_rows,
        "corruption_is_tail_only": info.corruption_is_tail_only,
        "ends_with_newline": info.ends_with_newline,
    }

    if not info.exists:
        plan["repairable"] = False
        plan["reason"] = "file does not exist"
        return plan

    if info.invalid_rows == 0:
        plan["repairable"] = False
        plan["reason"] = "no corruption detected; nothing to repair"
        plan["already_clean"] = True
        return plan

    if info.valid_prefix_rows == 0:
        plan["repairable"] = False
        plan["reason"] = "no valid JSON row found before the first invalid line"
        return plan

    if not info.corruption_is_tail_only:
        plan["repairable"] = False
        plan["reason"] = (
            "corruption is not tail-only: a valid JSON row appears after an "
            "invalid line (middle corruption). Refusing to guess sample "
            "alignment; this file needs manual/expert inspection, not "
            "automated repair."
        )
        return plan

    plan["repairable"] = True
    plan["reason"] = "contiguous tail-only corruption after a clean valid prefix"
    plan["planned_retained_rows"] = info.valid_prefix_rows
    plan["planned_removed_invalid_rows"] = info.invalid_rows
    return plan


def print_plan(plan):
    print(f"path                     = {plan['path']}")
    print(f"exists                   = {plan['exists']}")
    if not plan["exists"]:
        print(f"reason                   = {plan['reason']}")
        return
    print(f"nonempty_rows            = {plan['nonempty_rows']}")
    print(f"valid_rows               = {plan['valid_rows']}")
    print(f"invalid_rows             = {plan['invalid_rows']}")
    print(f"first_invalid_line       = {plan['first_invalid_line']}")
    print(f"invalid_line_numbers     = {plan['invalid_line_numbers']}")
    print(f"valid_prefix_rows        = {plan['valid_prefix_rows']}")
    print(f"corruption_is_tail_only  = {plan['corruption_is_tail_only']}")
    print(f"ends_with_newline        = {plan['ends_with_newline']}")
    print(f"repairable               = {plan['repairable']}")
    print(f"reason                   = {plan['reason']}")
    if plan.get("repairable"):
        print(f"planned_retained_rows        = {plan['planned_retained_rows']}")
        print(f"planned_removed_invalid_rows = {plan['planned_removed_invalid_rows']}")


def apply_repair(path, backup_dir):
    plan = plan_repair(path)
    if not plan["repairable"]:
        print(f"REFUSING to repair {path}: {plan['reason']}", file=sys.stderr)
        return 1, plan

    original_sha256 = sha256_of_file(path)
    original_bytes = os.path.getsize(path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(backup_dir, timestamp)
    os.makedirs(backup_subdir, exist_ok=True)
    backup_path = os.path.join(
        backup_subdir, os.path.basename(path) + ".corrupt"
    )

    # Byte-for-byte backup of the untouched original before anything else.
    with open(path, "rb") as src, open(backup_path, "wb") as dst:
        dst.write(src.read())
        dst.flush()
        os.fsync(dst.fileno())
    backup_sha256 = sha256_of_file(backup_path)
    if backup_sha256 != original_sha256:
        print(
            f"ABORTING: backup SHA256 ({backup_sha256}) does not match "
            f"original SHA256 ({original_sha256}). Original partial left "
            "untouched.",
            file=sys.stderr,
        )
        return 1, plan

    # Build the repaired content: only the contiguous valid prefix, in
    # order, using each line's original raw text (not re-serialized).
    lines = read_lines_preserving_content(path)
    kept_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            is_valid = isinstance(obj, dict)
        except (json.JSONDecodeError, ValueError):
            is_valid = False
        if not is_valid:
            break
        kept_lines.append(line)
        if len(kept_lines) >= plan["planned_retained_rows"]:
            break

    if len(kept_lines) != plan["planned_retained_rows"]:
        print(
            f"ABORTING: reconstructed {len(kept_lines)} valid rows, expected "
            f"{plan['planned_retained_rows']}. Original partial left untouched.",
            file=sys.stderr,
        )
        return 1, plan

    tmp_path = path + ".repair.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line)
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        # Re-read and strictly re-validate the tmp file before replacing.
        with open(tmp_path, "rb") as f:
            repaired_raw = f.read()
        verify_rows = 0
        for line in repaired_raw.decode("utf-8", errors="surrogateescape").split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)  # must not raise
            if not isinstance(obj, dict):
                raise ValueError("repaired line did not parse to a JSON object")
            verify_rows += 1

        if verify_rows != plan["planned_retained_rows"]:
            raise ValueError(
                f"post-write verification found {verify_rows} valid rows, "
                f"expected {plan['planned_retained_rows']}"
            )
        if not repaired_raw.endswith(b"\n"):
            raise ValueError("repaired file does not end with a newline")
        if b"\x00" in repaired_raw:
            raise ValueError("repaired file still contains NUL bytes")

        repaired_sha256 = hashlib.sha256(repaired_raw).hexdigest()
        os.replace(tmp_path, path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(
            f"ABORTING repair of {path}: {exc}. Original partial left "
            "untouched.",
            file=sys.stderr,
        )
        return 1, plan

    manifest = {
        "timestamp": timestamp,
        "source_path": os.path.abspath(path),
        "backup_path": os.path.abspath(backup_path),
        "original_sha256": original_sha256,
        "repaired_sha256": repaired_sha256,
        "original_bytes": original_bytes,
        "repaired_bytes": len(repaired_raw),
        "valid_prefix_rows": plan["planned_retained_rows"],
        "invalid_line_numbers": plan["invalid_line_numbers"],
        "detected_corruption_type": "tail_only",
        "git_commit": get_git_commit(),
        "hostname": socket.gethostname(),
        "action": "truncate_to_valid_prefix",
    }
    manifest_path = os.path.join(backup_subdir, "recovery_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    plan["applied"] = True
    plan["backup_path"] = backup_path
    plan["manifest_path"] = manifest_path
    plan["original_sha256"] = original_sha256
    plan["repaired_sha256"] = repaired_sha256
    return 0, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Path to the *.jsonl.partial file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inspect and report only (default)")
    mode.add_argument("--apply", action="store_true", help="Actually repair the file")
    parser.add_argument(
        "--backup-dir",
        default="outputs/recovery_backups",
        help="Directory to store the pre-repair backup + manifest (default: outputs/recovery_backups)",
    )
    args = parser.parse_args()

    if args.apply:
        exit_code, plan = apply_repair(args.path, args.backup_dir)
        print_plan(plan)
        if exit_code == 0:
            print(f"backup_path              = {plan['backup_path']}")
            print(f"manifest_path            = {plan['manifest_path']}")
            print(f"original_sha256          = {plan['original_sha256']}")
            print(f"repaired_sha256          = {plan['repaired_sha256']}")
            print("REPAIR_APPLIED")
        else:
            print("REPAIR_FAILED_OR_REFUSED")
        sys.exit(exit_code)
    else:
        plan = plan_repair(args.path)
        print_plan(plan)
        print("(dry-run: no file was modified)")
        sys.exit(0 if plan.get("repairable") or plan.get("already_clean") else 1)


if __name__ == "__main__":
    main()
