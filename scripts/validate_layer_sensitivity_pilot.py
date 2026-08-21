"""Read-only post-run validation for the layer-sensitivity pilot (Stage D
post-run check). Never loads a model, never touches the GPU, never modifies
any prediction file. Reuses utils.jsonl_integrity.inspect_jsonl and
utils.pilot_policy's condition-discovery logic (never re-derives policy
resolution by hand).

This is integrity/experiment validation only -- it does NOT compute any
layer-sensitivity score.

Usage:
    ./.venv/bin/python scripts/validate_layer_sensitivity_pilot.py \\
        --output-dir analysis/results/layer_sensitivity_pilot_validation
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from utils.jsonl_integrity import inspect_jsonl  # noqa: E402
from utils.pilot_policy import (  # noqa: E402
    PILOT_AXES,
    PILOT_LAYERS,
    PILOT_TASK_COUNTS,
    discover_and_validate_pilot_policies,
    output_dir_name,
)

DEFAULT_PILOT_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_sensitivity_pilot")
DEFAULT_POLICIES_DIR = os.path.join(REPO_ROOT, "analysis", "policies", "layer_sensitivity_pilot")
DEFAULT_FP16_BASELINE_DIR = os.path.join(
    REPO_ROOT, "pred", "longchat-7b-v1.5-32k_31500_16bits_group32_residual128"
)
DEFAULT_MASTER_LOG = os.path.join(DEFAULT_PILOT_ROOT, "full_pilot_20260819_152137.log")

REQUIRED_RUN_CONFIG_VALUES = {
    "model_name_or_path": "lmsys/longchat-7b-v1.5-32k",
    "max_length": 31500,
    "group_size": 32,
    "residual_length": 128,
    "seed": 42,
}

# Keywords whose presence anywhere in the master log is a real-failure
# signal; matched case-insensitively as whole-ish tokens, not substrings of
# known-benign boilerplate (see _classify_log_lines below).
FAILURE_KEYWORDS = [
    "traceback", "cuda error", "ptxas error", "runtimeerror", "nan", "inf",
    "failed", "corrupt", "mismatch", "over-count", "overcount", "exception",
]

# Exact substrings that are known-benign and must not trip FAILURE_KEYWORDS
# matching, because the keyword happens to appear inside otherwise-routine
# text. Only lines containing one of these are exempted.
KNOWN_BENIGN_LINE_SUBSTRINGS = [
    "you may observe exceptions, performance degradation",  # generic HF long-generation reminder
    "Found GPU0 NVIDIA GB10 which is of cuda capability 12.1",
    "Unrecognized keys in `rope_scaling`",
    "temperature",  # do_sample/top_p/temperature config warnings
    "top_p",
    "Token indices sequence length is longer than the specified maximum",
    "Loading checkpoint shards",
    "nohup: ignoring input",
]


class PilotValidationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 1. Global output inventory
# ---------------------------------------------------------------------------

def inventory_conditions(pilot_root, policies_dir):
    expected_conditions = discover_and_validate_pilot_policies(policies_dir, PILOT_LAYERS, PILOT_AXES)
    expected_by_dirname = {output_dir_name(c): c for c in expected_conditions}

    actual_dirnames = sorted(
        d.name for d in Path(pilot_root).iterdir()
        if d.is_dir() and re.match(r"^layer\d{2}_(key|value)_[0-9a-f]{12}$", d.name)
    )

    missing = sorted(set(expected_by_dirname) - set(actual_dirnames))
    unexpected = sorted(set(actual_dirnames) - set(expected_by_dirname))

    matrix = []
    for dirname in sorted(expected_by_dirname):
        cond = expected_by_dirname[dirname]
        condition_dir = os.path.join(pilot_root, dirname)
        present = os.path.isdir(condition_dir)
        task_files_present = 0
        if present:
            task_files_present = sum(
                1 for t in PILOT_TASK_COUNTS if os.path.exists(os.path.join(condition_dir, f"{t}.jsonl"))
            )
        matrix.append(
            {
                "condition_id": cond["condition_id"],
                "layer_idx": cond["layer_idx"],
                "axis": cond["axis"],
                "policy_hash": cond["resolved_policy_hash"],
                "dir_present": present,
                "task_files_present": task_files_present,
                "task_files_expected": len(PILOT_TASK_COUNTS),
            }
        )

    hashes = [c["resolved_policy_hash"] for c in expected_conditions]
    return {
        "expected_condition_count": len(expected_conditions),
        "actual_condition_dir_count": len(actual_dirnames),
        "missing_conditions": missing,
        "unexpected_directories": unexpected,
        "all_16_hashes_unique": len(set(hashes)) == len(hashes),
        "matrix": matrix,
    }, expected_by_dirname


# ---------------------------------------------------------------------------
# 2. Strict JSONL validation (exact count, never >=)
# ---------------------------------------------------------------------------

def validate_all_jsonl(pilot_root, expected_by_dirname, task_counts=PILOT_TASK_COUNTS):
    results = []
    total_rows = 0
    n_files_checked = 0
    n_files_pass = 0
    n_partial_remaining = 0

    for dirname in sorted(expected_by_dirname):
        condition_dir = os.path.join(pilot_root, dirname)
        for task, expected in task_counts.items():
            path = os.path.join(condition_dir, f"{task}.jsonl")
            partial_path = path + ".partial"
            if os.path.exists(partial_path):
                n_partial_remaining += 1
            info = inspect_jsonl(path)
            n_files_checked += 1
            ok = info.exists and info.invalid_rows == 0 and info.valid_rows == expected and info.ends_with_newline
            if ok:
                n_files_pass += 1
                total_rows += info.valid_rows
            results.append(
                {
                    "condition_id": expected_by_dirname[dirname]["condition_id"],
                    "task": task,
                    "exists": info.exists,
                    "valid_rows": info.valid_rows,
                    "invalid_rows": info.invalid_rows,
                    "expected": expected,
                    "ends_with_newline": info.ends_with_newline,
                    "leftover_partial": os.path.exists(partial_path),
                    "ok": ok,
                }
            )

    return {
        "n_files_checked": n_files_checked,
        "n_files_pass": n_files_pass,
        "n_partial_remaining": n_partial_remaining,
        "total_rows": total_rows,
        "expected_total_rows": len(expected_by_dirname) * sum(task_counts.values()),
        "results": results,
    }


# ---------------------------------------------------------------------------
# 3. Pairing validation against the approved FP16 baseline
# ---------------------------------------------------------------------------

def _load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_pairing_all(pilot_root, expected_by_dirname, baseline_dir, task_counts=PILOT_TASK_COUNTS):
    baseline_rows = {}
    for task in task_counts:
        baseline_rows[task] = _load_rows(os.path.join(baseline_dir, f"{task}.jsonl"))

    results = []
    n_files_paired = 0
    n_rows_paired = 0
    for dirname in sorted(expected_by_dirname):
        condition_dir = os.path.join(pilot_root, dirname)
        condition_id = expected_by_dirname[dirname]["condition_id"]
        for task, expected in task_counts.items():
            path = os.path.join(condition_dir, f"{task}.jsonl")
            file_ok = True
            row_mismatches = []
            if not os.path.exists(path):
                file_ok = False
            else:
                rows = _load_rows(path)
                base = baseline_rows[task]
                if len(rows) != len(base):
                    file_ok = False
                else:
                    for i, (r, b) in enumerate(zip(rows, base)):
                        for field in ("answers", "all_classes", "length"):
                            if r.get(field) != b.get(field):
                                row_mismatches.append({"row": i, "field": field})
                    file_ok = len(row_mismatches) == 0
            if file_ok:
                n_files_paired += 1
                n_rows_paired += expected
            results.append(
                {
                    "condition_id": condition_id,
                    "task": task,
                    "paired": file_ok,
                    "n_mismatches": len(row_mismatches),
                    "first_mismatches": row_mismatches[:5],
                }
            )

    return {
        "n_files_paired": n_files_paired,
        "n_files_total": len(expected_by_dirname) * len(task_counts),
        "n_rows_paired": n_rows_paired,
        "n_rows_total": len(expected_by_dirname) * sum(task_counts.values()),
        "baseline_dir": baseline_dir,
        "results": results,
    }


# ---------------------------------------------------------------------------
# 4. Condition / policy validation
# ---------------------------------------------------------------------------

def validate_condition_policies(pilot_root, expected_by_dirname):
    results = []
    for dirname, cond in sorted(expected_by_dirname.items()):
        condition_dir = os.path.join(pilot_root, dirname)
        run_config_path = os.path.join(condition_dir, "run_config.json")
        problems = []
        if not os.path.exists(run_config_path):
            problems.append("run_config.json missing")
            results.append({"condition_id": cond["condition_id"], "ok": False, "problems": problems})
            continue

        with open(run_config_path, "r", encoding="utf-8") as f:
            rc = json.load(f)

        if rc.get("layer_idx") != cond["layer_idx"]:
            problems.append(f"layer_idx mismatch: run_config={rc.get('layer_idx')} expected={cond['layer_idx']}")
        if rc.get("axis") != cond["axis"]:
            problems.append(f"axis mismatch: run_config={rc.get('axis')} expected={cond['axis']}")
        if rc.get("k_bits") != cond["k_bits"]:
            problems.append(f"k_bits mismatch: run_config={rc.get('k_bits')} expected={cond['k_bits']}")
        if rc.get("v_bits") != cond["v_bits"]:
            problems.append(f"v_bits mismatch: run_config={rc.get('v_bits')} expected={cond['v_bits']}")
        if rc.get("policy_hash") != cond["resolved_policy_hash"]:
            problems.append(f"policy_hash mismatch: run_config={rc.get('policy_hash')} expected={cond['resolved_policy_hash']}")
        if rc.get("policy_path") != cond["policy_path"]:
            problems.append(f"policy_path mismatch: run_config={rc.get('policy_path')} expected={cond['policy_path']}")

        for key, expected_value in REQUIRED_RUN_CONFIG_VALUES.items():
            if rc.get(key) != expected_value:
                problems.append(f"{key} mismatch: run_config={rc.get(key)} expected={expected_value}")

        # Cross-check the resolved per-layer policy: probed layer matches
        # the axis-specific bits, every other layer is K16/V16.
        resolved = cond["resolved_layer_policy"]
        for i, entry in enumerate(resolved.layers):
            if i == cond["layer_idx"]:
                want = (cond["k_bits"], cond["v_bits"])
            else:
                want = (16, 16)
            got = (entry.k_bits, entry.v_bits)
            if got != want:
                problems.append(f"resolved policy layer {i}: expected {want}, got {got}")

        results.append({"condition_id": cond["condition_id"], "ok": len(problems) == 0, "problems": problems})

    return results


# ---------------------------------------------------------------------------
# 5. Manifest / exit validation
# ---------------------------------------------------------------------------

def validate_manifests(pilot_root, expected_by_dirname):
    pilot_manifest_path = os.path.join(pilot_root, "pilot_manifest.json")
    top_level = {"exists": os.path.exists(pilot_manifest_path)}
    condition_statuses = {}
    if top_level["exists"]:
        with open(pilot_manifest_path, "r", encoding="utf-8") as f:
            pm = json.load(f)
        top_level["num_conditions"] = pm.get("num_conditions")
        for c in pm.get("conditions", []):
            condition_statuses[c["condition_id"]] = c.get("status")

    results = []
    for dirname, cond in sorted(expected_by_dirname.items()):
        condition_dir = os.path.join(pilot_root, dirname)
        condition_id = cond["condition_id"]
        manifest_path = os.path.join(condition_dir, "manifest.json")
        host_monitor_path = os.path.join(condition_dir, "host_monitor.log")
        problems = []

        pilot_status = condition_statuses.get(condition_id)
        if pilot_status != "complete":
            problems.append(f"pilot_manifest status = {pilot_status!r}, expected 'complete'")

        if not os.path.exists(manifest_path):
            problems.append("condition manifest.json missing")
            m = {}
        else:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            if m.get("exit_code") != 0:
                problems.append(f"exit_code = {m.get('exit_code')}, expected 0")
            if m.get("boot_id_start") != m.get("boot_id_end"):
                problems.append("boot_id_start != boot_id_end")
            if not m.get("start_time") or not m.get("end_time"):
                problems.append("start_time/end_time missing")
            if m.get("nonfinite_logits_seen") is not False:
                problems.append(f"nonfinite_logits_seen = {m.get('nonfinite_logits_seen')!r}, expected False")
            # This schema has no explicit "generation_error" field (unlike
            # scripts/layer_policy_smoke.py's manifest). The semantic
            # equivalent here is exit_code == 0: run_condition()'s
            # try/finally only reaches `exit_code = 0` after its whole body
            # (all requested tasks) completes without an exception, so a
            # generation error would already show up as exit_code != 0
            # above, never silently as a separate untracked field.

        n_monitor_samples = 0
        if not os.path.exists(host_monitor_path):
            problems.append("host_monitor.log missing")
        else:
            with open(host_monitor_path, "r", encoding="utf-8") as f:
                n_monitor_samples = sum(1 for line in f if line.strip())
            if n_monitor_samples < 1:
                problems.append("host_monitor.log has zero samples")

        results.append(
            {
                "condition_id": condition_id,
                "pilot_manifest_status": pilot_status,
                "exit_code": m.get("exit_code"),
                "boot_id_start": m.get("boot_id_start"),
                "boot_id_end": m.get("boot_id_end"),
                "start_time": m.get("start_time"),
                "end_time": m.get("end_time"),
                "nonfinite_logits_seen": m.get("nonfinite_logits_seen"),
                "host_monitor_samples": n_monitor_samples,
                "ok": len(problems) == 0,
                "problems": problems,
            }
        )

    return top_level, results


# ---------------------------------------------------------------------------
# 6. Host reboot check
# ---------------------------------------------------------------------------

def check_boot_ids(manifest_results):
    boot_ids = set()
    n_matched = 0
    for r in manifest_results:
        if r["boot_id_start"]:
            boot_ids.add(r["boot_id_start"])
        if r["boot_id_end"]:
            boot_ids.add(r["boot_id_end"])
        if r["boot_id_start"] and r["boot_id_start"] == r["boot_id_end"]:
            n_matched += 1
    return {
        "n_conditions_checked": len(manifest_results),
        "n_conditions_boot_id_matched": n_matched,
        "unique_boot_ids_observed": sorted(boot_ids),
        "crossed_reboot": len(boot_ids) > 1,
        "kernel_log_limitation": (
            "journalctl kernel-event inspection is permission-blocked for this account "
            "in this environment; this check does NOT claim kernel logs are clean, only "
            "that boot_id was stable per-condition per available OS-level signal."
        ),
    }


# ---------------------------------------------------------------------------
# 7. Canary reuse sanity (layer00_key/trec specifically)
# ---------------------------------------------------------------------------

def check_canary_reuse(pilot_root, expected_by_dirname, canary_end_time_iso="2026-08-19T14:34:35"):
    dirname = next(d for d, c in expected_by_dirname.items() if c["condition_id"] == "layer00_key")
    trec_path = os.path.join(pilot_root, dirname, "trec.jsonl")
    manifest_path = os.path.join(pilot_root, dirname, "manifest.json")

    info = inspect_jsonl(trec_path)
    mtime = os.path.getmtime(trec_path) if os.path.exists(trec_path) else None

    condition_manifest_start = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            condition_manifest_start = json.load(f).get("start_time")

    # If trec.jsonl's mtime predates the *condition's* full-pilot start_time
    # (recorded in manifest.json, which the finalize step overwrites even
    # when trec itself was skipped -- see report), that's direct evidence
    # the file's bytes were never rewritten during the full pilot run.
    mtime_iso = None
    reused = None
    if mtime is not None and condition_manifest_start:
        import datetime as _dt
        mtime_iso = _dt.datetime.fromtimestamp(mtime).astimezone().isoformat()
        try:
            manifest_start_dt = _dt.datetime.fromisoformat(condition_manifest_start)
            file_mtime_dt = _dt.datetime.fromtimestamp(mtime).astimezone()
            reused = file_mtime_dt < manifest_start_dt
        except ValueError:
            reused = None

    return {
        "trec_jsonl_valid_rows": info.valid_rows,
        "trec_jsonl_invalid_rows": info.invalid_rows,
        "trec_jsonl_row_count_matches_no_duplication": info.valid_rows == 200,
        "trec_jsonl_mtime": mtime_iso,
        "condition_manifest_start_time_full_run": condition_manifest_start,
        "file_predates_full_run_start_evidence_of_reuse": reused,
    }


# ---------------------------------------------------------------------------
# 8. Master log scan
# ---------------------------------------------------------------------------

def scan_master_log(log_path, failure_keywords=FAILURE_KEYWORDS, benign_substrings=KNOWN_BENIGN_LINE_SUBSTRINGS):
    if not os.path.exists(log_path):
        return {"exists": False, "log_path": log_path}

    flagged = []
    n_lines = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            n_lines += 1
            lower = line.lower()
            for kw in failure_keywords:
                if kw in lower:
                    if any(b.lower() in lower for b in benign_substrings):
                        continue
                    flagged.append({"line_no": i, "keyword": kw, "text": line.strip()[:200]})
                    break

    return {
        "exists": True,
        "log_path": log_path,
        "n_lines": n_lines,
        "n_flagged_lines": len(flagged),
        "flagged_lines": flagged,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot-root", default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--policies-dir", default=DEFAULT_POLICIES_DIR)
    parser.add_argument("--fp16-baseline-dir", default=DEFAULT_FP16_BASELINE_DIR)
    parser.add_argument("--master-log", default=DEFAULT_MASTER_LOG)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "analysis", "results", "layer_sensitivity_pilot_validation"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Global output inventory...")
    inventory, expected_by_dirname = inventory_conditions(args.pilot_root, args.policies_dir)

    print("2. Strict JSONL validation...")
    jsonl_result = validate_all_jsonl(args.pilot_root, expected_by_dirname)

    print("3. Pairing validation vs FP16 baseline...")
    pairing_result = validate_pairing_all(args.pilot_root, expected_by_dirname, args.fp16_baseline_dir)

    print("4. Condition / policy validation...")
    policy_results = validate_condition_policies(args.pilot_root, expected_by_dirname)

    print("5. Manifest / exit validation...")
    top_manifest, manifest_results = validate_manifests(args.pilot_root, expected_by_dirname)

    print("6. Host reboot check...")
    boot_id_check = check_boot_ids(manifest_results)

    print("7. Canary reuse sanity...")
    canary_check = check_canary_reuse(args.pilot_root, expected_by_dirname)

    print("8. Master log scan...")
    log_scan = scan_master_log(args.master_log)

    overall_pass = (
        inventory["expected_condition_count"] == inventory["actual_condition_dir_count"]
        and not inventory["missing_conditions"]
        and not inventory["unexpected_directories"]
        and inventory["all_16_hashes_unique"]
        and jsonl_result["n_files_pass"] == jsonl_result["n_files_checked"]
        and jsonl_result["n_partial_remaining"] == 0
        and jsonl_result["total_rows"] == jsonl_result["expected_total_rows"] == 17600
        and pairing_result["n_files_paired"] == pairing_result["n_files_total"]
        and all(r["ok"] for r in policy_results)
        and all(r["ok"] for r in manifest_results)
        and not boot_id_check["crossed_reboot"]
        and canary_check["trec_jsonl_row_count_matches_no_duplication"]
        and log_scan.get("n_flagged_lines", 0) == 0
    )

    summary = {
        "verdict": "FULL_STAGE_D_GENERATION = PASS" if overall_pass else "FULL_STAGE_D_GENERATION = FAIL",
        "inventory": inventory,
        "jsonl_validation": {k: v for k, v in jsonl_result.items() if k != "results"},
        "pairing_validation": {k: v for k, v in pairing_result.items() if k != "results"},
        "policy_validation_all_ok": all(r["ok"] for r in policy_results),
        "manifest_validation_all_ok": all(r["ok"] for r in manifest_results),
        "top_level_manifest": top_manifest,
        "boot_id_check": boot_id_check,
        "canary_reuse_check": canary_check,
        "master_log_scan": {k: v for k, v in log_scan.items() if k != "flagged_lines"},
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "jsonl_validation_detail.json", "w", encoding="utf-8") as f:
        json.dump(jsonl_result["results"], f, indent=2)
    with open(output_dir / "pairing_validation_detail.json", "w", encoding="utf-8") as f:
        json.dump(pairing_result["results"], f, indent=2)
    with open(output_dir / "policy_validation_detail.json", "w", encoding="utf-8") as f:
        json.dump(policy_results, f, indent=2)
    with open(output_dir / "manifest_validation_detail.json", "w", encoding="utf-8") as f:
        json.dump(manifest_results, f, indent=2)
    with open(output_dir / "master_log_scan_detail.json", "w", encoding="utf-8") as f:
        json.dump(log_scan, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote validation report to {output_dir}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
