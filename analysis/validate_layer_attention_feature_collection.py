"""Stage H3: strict, read-only validation of a Stage-H scientific
attention-feature collection.

CPU-only. Does NOT compute any Spearman correlation and does NOT evaluate
the Stage-H A/B/C/D gate -- that is a separate, later, explicitly
authorized step. This script only checks structural/provenance integrity:
exact counts, exact identities, per-record internal consistency (reusing
scripts.run_layer_attention_feature_pilot.validate_record_shape, never an
independent reimplementation), and manifest/commit provenance.

Usage:
    ./.venv/bin/python analysis/validate_layer_attention_feature_collection.py \\
        --primary-root outputs/layer_attention_feature_pilot/primary \\
        --diagnostic-root outputs/layer_attention_feature_pilot/diagnostic
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from scripts.run_layer_attention_feature_pilot import (  # noqa: E402 -- reuse, never reimplement
    DEFAULT_REQUESTED_STEPS,
    RecordValidationError,
    build_trajectory_plan,
    filter_plan,
    trajectory_identity,
    validate_record_shape,
)

DEFAULT_PRIMARY_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_attention_feature_pilot", "primary")
DEFAULT_DIAGNOSTIC_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_attention_feature_pilot", "diagnostic")


class ValidationError(RuntimeError):
    pass


def get_current_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def discover_run_dirs(root):
    """Every immediate subdirectory of `root` containing a features.jsonl
    -- supports multiple resumed/labeled runs under one scope root (unlike
    Stage G's single-run-per-scope discovery), since Stage H may be
    resumed across several invocations."""
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "features.jsonl"))
    )


def load_records_from_run_dirs(run_dirs, scope_label):
    records = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "features.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValidationError(f"{path}:{line_no}: malformed JSON: {e}") from e
                if record.get("scope") != scope_label:
                    raise ValidationError(f"{path}:{line_no}: expected scope={scope_label!r}, got {record.get('scope')!r}")
                records.append((f"{path}:{line_no}", record))
    return records


def _validate_records_against_expected_plan(all_records, expected_plan, run_dirs, expected_commit=None):
    """Shared core reused by BOTH the full 256/16/272 validator and the
    selector-filtered partial-plan validator (Part 2, Stage H3C-CANARY):
    per-record structural validity (reused, never reimplemented), locked
    requested-step check, exact identity completeness/no-duplicates
    against `expected_plan`, and manifest provenance. `expected_plan` is a
    list of trajectory dicts (from build_trajectory_plan() for the full
    case, or filter_plan(build_trajectory_plan(), ...) for a partial
    case) -- never independently re-derived.

    Manifest provenance reads the REAL H3B manifest field name
    (`collection_head`) written by scripts.run_layer_attention_feature_pilot.run_real_collection
    -- NOT `git_commit` (a pre-H3B placeholder key this function used to
    check, which never matched any real manifest ever produced).
    """
    problems = []

    per_record_ok = 0
    for loc, record in all_records:
        try:
            validate_record_shape(record)
            per_record_ok += 1
        except RecordValidationError as e:
            problems.append(f"{loc}: {e}")

    if list(DEFAULT_REQUESTED_STEPS) != [1, 2, 4, 8, 16]:
        # Defensive: this constant must never silently drift.
        problems.append(f"DEFAULT_REQUESTED_STEPS is not [1,2,4,8,16]: {list(DEFAULT_REQUESTED_STEPS)}")
    for loc, record in all_records:
        if record.get("requested_decode_steps") != [1, 2, 4, 8, 16]:
            problems.append(f"{loc}: requested_decode_steps != [1,2,4,8,16]: {record.get('requested_decode_steps')}")

    expected_identities = {trajectory_identity(t) for t in expected_plan}
    seen_identities = {}
    for loc, record in all_records:
        identity = (record.get("task"), record.get("dataset_index"), record.get("layer_idx"), record.get("tensor_axis"))
        if identity in seen_identities:
            problems.append(f"{loc}: duplicate identity {identity} (also at {seen_identities[identity]})")
        else:
            seen_identities[identity] = loc

    missing = expected_identities - set(seen_identities)
    extra = set(seen_identities) - expected_identities
    if missing:
        problems.append(f"missing {len(missing)} expected identities: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        problems.append(f"found {len(extra)} unexpected identities not in the expected plan: {sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")

    manifest_problems = []
    manifests = []
    for run_dir in run_dirs:
        manifest_path = os.path.join(run_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifests.append((manifest_path, manifest))
            commit = manifest.get("collection_head")
            if expected_commit is not None and commit != expected_commit:
                manifest_problems.append(f"{manifest_path}: collection_head={commit!r} != expected {expected_commit!r}")
    problems.extend(manifest_problems)

    return {
        "problems": problems,
        "per_record_ok": per_record_ok,
        "expected_identities": expected_identities,
        "seen_identities": seen_identities,
        "manifests": manifests,
    }


def validate_collection(primary_root, diagnostic_root, expected_commit=None):
    primary_run_dirs = discover_run_dirs(primary_root)
    diagnostic_run_dirs = discover_run_dirs(diagnostic_root)

    primary_records = load_records_from_run_dirs(primary_run_dirs, "primary")
    diagnostic_records = load_records_from_run_dirs(diagnostic_run_dirs, "diagnostic")
    all_records = primary_records + diagnostic_records

    core = _validate_records_against_expected_plan(
        all_records, build_trajectory_plan(), primary_run_dirs + diagnostic_run_dirs, expected_commit
    )
    problems = core["problems"]

    # 24 unique prompt identities (task, dataset_index) -- descriptive check.
    prompt_identities = {(r.get("task"), r.get("dataset_index")) for _, r in all_records}

    result = {
        "primary_run_dirs": primary_run_dirs,
        "diagnostic_run_dirs": diagnostic_run_dirs,
        "primary_record_count": len(primary_records),
        "diagnostic_record_count": len(diagnostic_records),
        "total_record_count": len(all_records),
        "expected_primary_count": 256,
        "expected_diagnostic_count": 16,
        "expected_total_count": 272,
        "per_record_structurally_valid": core["per_record_ok"],
        "unique_prompt_identities": len(prompt_identities),
        "expected_unique_prompt_identities": 24,
        "unique_scientific_identities": len(core["seen_identities"]),
        "expected_scientific_identities": 272,
        "problems": problems,
        "ok": (
            not problems
            and len(primary_records) == 256
            and len(diagnostic_records) == 16
            and len(prompt_identities) == 24
        ),
    }
    return result


def validate_partial_collection(output_root, scope_label, expected_plan, expected_commit=None):
    """Stage H3C-CANARY (Part 2): strict read-only validation for a
    SELECTOR-FILTERED SUBSET of the full 272-trajectory plan -- e.g. the
    exact single-trajectory canaries this stage runs. Does NOT assume
    256/16/272; the caller supplies `expected_plan` (typically
    filter_plan(build_trajectory_plan(), ...)) so the expected identity
    set is EXACT, never "at least one row exists".

    Also validates manifest planned_count/completed_count/state (COMPLETE)
    -- the full validate_collection() does not check this today because
    Stage G's manifest never had a lifecycle state; H3B's does.

    `output_root` is a single canary run root (e.g.
    outputs/layer_attention_feature_runner_canary/h3c_key_l0_lcc122) --
    this function discovers run-dirs under it exactly like the full
    validator (discover_run_dirs), supporting a resumed run across
    multiple run-label subdirectories.
    """
    if not expected_plan:
        raise ValueError("expected_plan must not be empty -- a partial validator with no expectation validates nothing")

    run_dirs = discover_run_dirs(output_root)
    records = load_records_from_run_dirs(run_dirs, scope_label)

    core = _validate_records_against_expected_plan(records, expected_plan, run_dirs, expected_commit)
    problems = list(core["problems"])

    manifest_problems = []
    manifest_states = []
    for manifest_path, manifest in core["manifests"]:
        manifest_states.append(manifest.get("state"))
        if manifest.get("planned_count") != len(expected_plan):
            manifest_problems.append(
                f"{manifest_path}: planned_count={manifest.get('planned_count')!r} != expected {len(expected_plan)}"
            )
        if manifest.get("completed_count") != len(expected_plan):
            manifest_problems.append(
                f"{manifest_path}: completed_count={manifest.get('completed_count')!r} != expected {len(expected_plan)}"
            )
        if manifest.get("state") != "COMPLETE":
            manifest_problems.append(f"{manifest_path}: state={manifest.get('state')!r} != 'COMPLETE'")
    problems.extend(manifest_problems)

    manifest_found = len(core["manifests"]) > 0
    if not manifest_found:
        problems.append(f"no manifest.json found under any run dir in {output_root!r}")

    result = {
        "output_root": output_root,
        "run_dirs": run_dirs,
        "record_count": len(records),
        "expected_count": len(expected_plan),
        "per_record_structurally_valid": core["per_record_ok"],
        "unique_scientific_identities": len(core["seen_identities"]),
        "manifest_states": manifest_states,
        "problems": problems,
        "ok": (
            not problems
            and len(records) == len(expected_plan)
            and manifest_found
        ),
    }
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary-root", default=DEFAULT_PRIMARY_ROOT)
    p.add_argument("--diagnostic-root", default=DEFAULT_DIAGNOSTIC_ROOT)
    p.add_argument("--expected-commit", default=None, help="Defaults to the current HEAD.")
    return p.parse_args(argv)


def main():
    args = parse_args()
    expected_commit = args.expected_commit or get_current_head()
    result = validate_collection(args.primary_root, args.diagnostic_root, expected_commit)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
