"""Stage H4: preregistered decode-aware attention feature analysis.

CPU-only, read-only against the frozen Stage-H scientific collection
(outputs/layer_attention_feature_pilot/{primary,diagnostic}/). Never
modifies those files. Reuses, never reimplements:
  - analysis.validate_layer_attention_feature_collection (discover_run_dirs,
    load_records_from_run_dirs) to load the frozen JSONL records.
  - analysis.analyze_layer_sensitivity_pilot.spearman_corr (the project's
    one Spearman implementation, with its None-on-zero-variance
    "undefined" convention).
  - analysis.feature_pilot_gate (evaluate_criterion_a/b/c/d,
    evaluate_axis_gate, evaluate_feature_pilot_gate, to_gate_rho,
    compute_layer0_diagnostic_rho, DEFAULT_EFFECT_SIZE_THRESHOLD,
    PRIMARY_STAGE_E_TASKS, LOTO_REMOVED_TASK, LAYER0_DIAGNOSTIC_TASKS) --
    the exact same locked A/B/C/D gate logic used by Stage G, applied here
    to a different feature (decode-aware attention distortion instead of
    raw relative_l2 reconstruction error).
  - scripts.run_layer_attention_feature_pilot (PRIMARY_TASKS, PRIMARY_LAYERS,
    DIAGNOSTIC_TASKS, DIAGNOSTIC_LAYERS, AXES) for the exact scientific
    scope definitions, never re-derived.

Stage-E / Stage-F0 SignedSensitivity values are read verbatim from their
existing frozen CSV outputs (analysis/results/layer_sensitivity_pilot/
task_layer_sensitivity.csv, analysis/results/layer_sensitivity_f0/
task_results.csv) -- never recomputed here.

Does NOT compute anything beyond the preregistered H4 design: no feature
leaderboard, no alternative-aggregation search, no CONDITIONAL_GO branch.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from analysis.analyze_layer_sensitivity_pilot import spearman_corr  # noqa: E402 -- reuse, never reimplement
from analysis.feature_pilot_gate import (  # noqa: E402 -- reuse, never reimplement
    DEFAULT_EFFECT_SIZE_THRESHOLD,
    LAYER0_DIAGNOSTIC_TASKS,
    LOTO_REMOVED_TASK,
    PRIMARY_STAGE_E_TASKS,
    compute_layer0_diagnostic_rho,
    evaluate_criterion_a,
    evaluate_criterion_b,
    evaluate_criterion_c,
    evaluate_criterion_d,
    evaluate_feature_pilot_gate,
    to_gate_rho,
)
from analysis.validate_layer_attention_feature_collection import (  # noqa: E402 -- reuse, never reimplement
    discover_run_dirs,
    load_records_from_run_dirs,
)
from scripts.run_layer_attention_feature_pilot import (  # noqa: E402 -- reuse, never reimplement
    AXES,
    DIAGNOSTIC_LAYERS,
    DIAGNOSTIC_TASKS,
    PRIMARY_LAYERS,
    PRIMARY_TASKS,
)

DEFAULT_PRIMARY_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_attention_feature_pilot", "primary")
DEFAULT_DIAGNOSTIC_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_attention_feature_pilot", "diagnostic")
DEFAULT_STAGE_E_TASK_LAYER_CSV = os.path.join(REPO_ROOT, "analysis", "results", "layer_sensitivity_pilot", "task_layer_sensitivity.csv")
DEFAULT_F0_TASK_RESULTS_CSV = os.path.join(REPO_ROOT, "analysis", "results", "layer_sensitivity_f0", "task_results.csv")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "analysis", "results", "layer_attention_feature_pilot")

EXPECTED_PRIMARY_FEATURES = os.path.join(DEFAULT_PRIMARY_ROOT, "stage_h_primary_v1", "features.jsonl")
EXPECTED_PRIMARY_MANIFEST = os.path.join(DEFAULT_PRIMARY_ROOT, "stage_h_primary_v1", "manifest.json")
EXPECTED_DIAGNOSTIC_FEATURES = os.path.join(DEFAULT_DIAGNOSTIC_ROOT, "stage_h_diagnostic_v1", "features.jsonl")
EXPECTED_DIAGNOSTIC_MANIFEST = os.path.join(DEFAULT_DIAGNOSTIC_ROOT, "stage_h_diagnostic_v1", "manifest.json")

# Frozen checksums, recorded by the human-reviewed Stage-H collection report.
# A mismatch here means the input data changed since freeze -- refuse to
# analyze it rather than silently proceeding.
FROZEN_CHECKSUMS = {
    EXPECTED_PRIMARY_FEATURES: "b8b575a2a27e888d7e4d54f606edeea4634f105e46a38b90d38e9701914e25f9",
    EXPECTED_PRIMARY_MANIFEST: "185ecfd34b92ed13782ca9254a9c8eb0f45b49dc4f62c0118df423d7b31c282f",
    EXPECTED_DIAGNOSTIC_FEATURES: "a3ecf98f3ccc3f30b0a13d5fe3beb511593c163a3c00cf41700d78066cb51cf2",
    EXPECTED_DIAGNOSTIC_MANIFEST: "96f18234ad2e9f30ebbe8d3b52c5f0285765a944ee7ee0c9b0939eb301309617",
}

COLLECTION_HEAD = "72e24f9241451c7aa799dde9773eb52e8ced88af"


class FrozenInputError(RuntimeError):
    """A frozen scientific input file is missing or its checksum no longer
    matches -- fail closed rather than analyze data that may have changed
    since the human-reviewed freeze."""


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_checksums(expected=FROZEN_CHECKSUMS):
    """Returns dict[path] -> {"expected", "actual", "match"}. Raises
    FrozenInputError immediately on any missing file or mismatch -- this
    analysis must never run against data that differs from what was
    reviewed and frozen."""
    result = {}
    problems = []
    for path, expected_hash in expected.items():
        if not os.path.exists(path):
            problems.append(f"missing frozen input: {path}")
            result[path] = {"expected": expected_hash, "actual": None, "match": False}
            continue
        actual_hash = sha256_of(path)
        match = actual_hash == expected_hash
        result[path] = {"expected": expected_hash, "actual": actual_hash, "match": match}
        if not match:
            problems.append(f"checksum mismatch: {path} expected {expected_hash} got {actual_hash}")
    if problems:
        raise FrozenInputError("; ".join(problems))
    return result


# ---------------------------------------------------------------------------
# Loading frozen scientific records (reused discovery/load, never reimplemented)
# ---------------------------------------------------------------------------

def load_all_records(primary_root=DEFAULT_PRIMARY_ROOT, diagnostic_root=DEFAULT_DIAGNOSTIC_ROOT):
    primary_run_dirs = discover_run_dirs(primary_root)
    diagnostic_run_dirs = discover_run_dirs(diagnostic_root)
    primary = [r for _, r in load_records_from_run_dirs(primary_run_dirs, "primary")]
    diagnostic = [r for _, r in load_records_from_run_dirs(diagnostic_run_dirs, "diagnostic")]
    return primary, diagnostic


# ---------------------------------------------------------------------------
# Part 3: primary (task, layer_idx, tensor_axis) aggregation -- equal-sample
# arithmetic mean over the 4 preregistered prompts. Never token/length-weighted.
# ---------------------------------------------------------------------------

def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def aggregate_task_layer_axis(primary_records, tasks=PRIMARY_TASKS, layers=PRIMARY_LAYERS, axes=AXES, expected_n=4):
    """Returns dict[(task, layer_idx, axis)] -> {"mean", "median", "n",
    "values"}. `mean` is the PRIMARY (gate-feeding) aggregate: equal-sample
    arithmetic mean of sample_attention_distortion across the 4 prompts --
    no reweighting by token count, prompt length, decode length,
    valid_step_count, or cache length. `median` is reported ONLY as a
    secondary robustness summary (never substituted into the gate).
    Raises if any group does not have exactly `expected_n` samples.
    """
    grouped = {}
    for r in primary_records:
        key = (r["task"], r["layer_idx"], r["tensor_axis"])
        grouped.setdefault(key, []).append(r["sample_attention_distortion"])

    result = {}
    for task in tasks:
        for layer in layers:
            for axis in axes:
                key = (task, layer, axis)
                values = grouped.get(key, [])
                if len(values) != expected_n:
                    raise FrozenInputError(f"group {key} has {len(values)} samples, expected exactly {expected_n}")
                result[key] = {
                    "mean": sum(values) / len(values),
                    "median": _median(values),
                    "n": len(values),
                    "values": list(values),
                }
    return result


def aggregate_diagnostic_layer0(diagnostic_records, tasks=DIAGNOSTIC_TASKS, layers=DIAGNOSTIC_LAYERS, axes=AXES, expected_n=4):
    """Same aggregation rule, applied to the 16 diagnostic (multifieldqa_en,
    samsum) x layer-0 x {key,value} records."""
    grouped = {}
    for r in diagnostic_records:
        key = (r["task"], r["layer_idx"], r["tensor_axis"])
        grouped.setdefault(key, []).append(r["sample_attention_distortion"])

    result = {}
    for task in tasks:
        for layer in layers:
            for axis in axes:
                key = (task, layer, axis)
                values = grouped.get(key, [])
                if len(values) != expected_n:
                    raise FrozenInputError(f"group {key} has {len(values)} samples, expected exactly {expected_n}")
                result[key] = {
                    "mean": sum(values) / len(values),
                    "median": _median(values),
                    "n": len(values),
                    "values": list(values),
                }
    return result


# ---------------------------------------------------------------------------
# Stage-E / Stage-F0 SignedSensitivity lookups -- read verbatim, never recomputed
# ---------------------------------------------------------------------------

def load_stage_e_task_layer_sensitivity(path=DEFAULT_STAGE_E_TASK_LAYER_CSV):
    """dict[(task, layer_idx, axis)] -> delta (the per-task, per-layer,
    per-axis Signed{Key,Value}Sensitivity component -- i.e. the exact
    per-task value that SignedKeySensitivity/SignedValueSensitivity in
    layer_sensitivity.csv averages ACROSS tasks; this is the frozen
    Stage-E per-task decomposition, read verbatim)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["task"], int(row["layer_idx"]), row["axis"])] = float(row["delta"])
    return out


def load_f0_layer0_task_deltas(path=DEFAULT_F0_TASK_RESULTS_CSV):
    """dict[(task, axis)] -> Layer-0 delta, from Stage-F0's task_results.csv
    (key_delta/value_delta columns), covering multifieldqa_en and samsum
    (and, redundantly but consistently, the original 4 tasks -- only
    multifieldqa_en/samsum are used from this source; lcc's Layer-0 value
    is taken from the Stage-E CSV instead, per the preregistered Criterion
    D task->source mapping)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["task"], "key")] = float(row["key_delta"])
            out[(row["task"], "value")] = float(row["value_delta"])
    return out


# ---------------------------------------------------------------------------
# Primary per-task Spearman (Part 5): n=8 layers, never pooled across tasks
# ---------------------------------------------------------------------------

def compute_primary_task_rho(agg, stage_e, axis, tasks=PRIMARY_STAGE_E_TASKS, layers=PRIMARY_LAYERS):
    """Returns dict[task] -> {"raw_rho", "gate_rho", "n", "x", "y",
    "feature_has_variance", "target_has_variance"}. x = mean decode
    distortion per layer (this axis); y = Stage-E per-task signed
    sensitivity per layer (this axis). Uses the shared spearman_corr,
    which itself returns None (-> undefined) whenever either vector has
    zero variance."""
    result = {}
    for task in tasks:
        x = [agg[(task, layer, axis)]["mean"] for layer in layers]
        y = [stage_e[(task, layer, axis)] for layer in layers]
        raw_rho = spearman_corr(x, y)
        result[task] = {
            "raw_rho": raw_rho,
            "gate_rho": to_gate_rho(raw_rho),
            "n": len(layers),
            "x": x,
            "y": y,
            "feature_has_variance": len(set(x)) > 1,
            "target_has_variance": len(set(y)) > 1,
        }
    return result


# ---------------------------------------------------------------------------
# Criterion D inputs (Part 10): n=3 Layer-0 diagnostic
# ---------------------------------------------------------------------------

def compute_layer0_diagnostic_inputs(agg_primary, agg_diagnostic, stage_e, f0, axis, tasks=LAYER0_DIAGNOSTIC_TASKS):
    """Returns (x_by_task, y_by_task) for compute_layer0_diagnostic_rho:
    x = mean Layer-0 decode distortion per task (this axis); y = the
    corresponding finalized SignedSensitivity (lcc from Stage-E,
    multifieldqa_en/samsum from Stage-F0)."""
    x_by_task, y_by_task = {}, {}
    for task in tasks:
        if task == "lcc":
            x_by_task[task] = agg_primary[(task, 0, axis)]["mean"]
            y_by_task[task] = stage_e[(task, 0, axis)]
        else:
            x_by_task[task] = agg_diagnostic[(task, 0, axis)]["mean"]
            y_by_task[task] = f0[(task, axis)]
    return x_by_task, y_by_task


# ---------------------------------------------------------------------------
# Part 13: secondary/descriptive-only summaries (never gating)
# ---------------------------------------------------------------------------

def secondary_full_loto_table(raw_rho_by_task, tasks=PRIMARY_STAGE_E_TASKS):
    """Generalizes Criterion C's leave-one-out structure to ALL 4 tasks
    (Criterion C itself locks removed_task='lcc' only). Purely descriptive
    -- never substituted for Criterion C, never used to pick a different
    removed task after seeing results."""
    table = {}
    for removed in tasks:
        table[removed] = evaluate_criterion_c(raw_rho_by_task, removed_task=removed, tasks=tasks)
    return table


def per_task_valid_step_count_distribution(records):
    from collections import Counter

    out = {}
    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r["valid_step_count"])
    for task, counts in by_task.items():
        out[task] = dict(sorted(Counter(counts).items()))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_analysis(
    primary_root=DEFAULT_PRIMARY_ROOT,
    diagnostic_root=DEFAULT_DIAGNOSTIC_ROOT,
    stage_e_csv=DEFAULT_STAGE_E_TASK_LAYER_CSV,
    f0_csv=DEFAULT_F0_TASK_RESULTS_CSV,
    threshold=DEFAULT_EFFECT_SIZE_THRESHOLD,
):
    checksum_result = verify_frozen_checksums()

    primary_records, diagnostic_records = load_all_records(primary_root, diagnostic_root)
    if len(primary_records) != 256:
        raise FrozenInputError(f"expected exactly 256 primary records, got {len(primary_records)}")
    if len(diagnostic_records) != 16:
        raise FrozenInputError(f"expected exactly 16 diagnostic records, got {len(diagnostic_records)}")

    agg_primary = aggregate_task_layer_axis(primary_records)
    agg_diagnostic = aggregate_diagnostic_layer0(diagnostic_records)
    if len(agg_primary) != 64:
        raise FrozenInputError(f"expected exactly 64 primary (task,layer,axis) groups, got {len(agg_primary)}")

    stage_e = load_stage_e_task_layer_sensitivity(stage_e_csv)
    f0 = load_f0_layer0_task_deltas(f0_csv)

    axis_results = {}
    for axis in ("key", "value"):
        raw_rho_by_task = {t: r["raw_rho"] for t, r in compute_primary_task_rho(agg_primary, stage_e, axis).items()}
        per_task_detail = compute_primary_task_rho(agg_primary, stage_e, axis)

        a = evaluate_criterion_a(raw_rho_by_task)
        b = evaluate_criterion_b(raw_rho_by_task, threshold=threshold)
        c = evaluate_criterion_c(raw_rho_by_task, removed_task=LOTO_REMOVED_TASK)

        x3, y3 = compute_layer0_diagnostic_inputs(agg_primary, agg_diagnostic, stage_e, f0, axis)
        layer0_raw_rho = compute_layer0_diagnostic_rho(x3, y3)
        d = evaluate_criterion_d(layer0_raw_rho)

        axis_go = a["pass"] and b["pass"] and c["pass"] and d["pass"]

        axis_results[axis] = {
            "per_task_detail": per_task_detail,
            "raw_rho_by_task": raw_rho_by_task,
            "criterion_a": a,
            "criterion_b": b,
            "criterion_c": c,
            "criterion_d": d,
            "layer0_x_by_task": x3,
            "layer0_y_by_task": y3,
            "axis_go": axis_go,
            "secondary_full_loto": secondary_full_loto_table(raw_rho_by_task),
        }

    stage_decision = "GO" if (axis_results["key"]["axis_go"] or axis_results["value"]["axis_go"]) else "NO_GO"

    # Cross-check against the reused generic gate evaluator, to prove this
    # script's own A/B/C/D wiring agrees with feature_pilot_gate's combined
    # evaluate_feature_pilot_gate (never diverges from the shared library).
    cross_check = evaluate_feature_pilot_gate(
        axis_results["key"]["raw_rho_by_task"], axis_results["key"]["criterion_d"]["layer0_raw_rho"],
        axis_results["value"]["raw_rho_by_task"], axis_results["value"]["criterion_d"]["layer0_raw_rho"],
        threshold=threshold,
    )
    if cross_check["FEATURE_PILOT_STAGE"] != stage_decision:
        raise FrozenInputError("internal inconsistency: local gate combination disagrees with feature_pilot_gate.evaluate_feature_pilot_gate")

    all_records = primary_records + diagnostic_records
    early_eos_audit = {
        "valid_step_count_distribution": dict(sorted(__import__("collections").Counter(r["valid_step_count"] for r in all_records).items())),
        "zero_exposure_count": sum(1 for r in all_records if r["no_decode_exposure"]),
        "per_task_valid_step_count_distribution": per_task_valid_step_count_distribution(all_records),
    }

    return {
        "collection_head": COLLECTION_HEAD,
        "checksum_verification": checksum_result,
        "n_primary_records": len(primary_records),
        "n_diagnostic_records": len(diagnostic_records),
        "n_primary_groups": len(agg_primary),
        "axis_results": axis_results,
        "STAGE_H_ATTENTION_FEATURE": stage_decision,
        "early_eos_audit": early_eos_audit,
        "agg_primary": agg_primary,
        "agg_diagnostic": agg_diagnostic,
    }


def _clean_for_json(obj):
    # Convert tuple-keyed dicts to a JSON-safe form.
    if isinstance(obj, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


def write_outputs(result, primary_records, diagnostic_records, output_root=DEFAULT_OUTPUT_ROOT):
    """Writes every Part-15 output file from an already-computed `result`
    (run_full_analysis()'s return value) plus the raw record lists. Kept
    separate from run_full_analysis so the pure computation is testable
    without touching disk."""
    os.makedirs(output_root, exist_ok=True)

    with open(os.path.join(output_root, "collection_checksum_verification.json"), "w", encoding="utf-8") as f:
        json.dump({"collection_head": result["collection_head"], "checksums": result["checksum_verification"]}, f, indent=2)

    field_order = (
        "scope", "task", "dataset_index", "layer_idx", "tensor_axis", "policy", "k_bits", "v_bits",
        "model", "git_commit", "seed", "prompt_input_tokens", "actual_generated_token_count",
        "requested_decode_steps", "sampled_decode_steps", "valid_step_count", "no_decode_exposure",
        "sample_attention_distortion", "per_step_distortions",
    )
    with open(os.path.join(output_root, "sample_features.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(field_order)
        for r in primary_records + diagnostic_records:
            row = [json.dumps(r[k]) if isinstance(r[k], (list, dict)) else r[k] for k in field_order]
            w.writerow(row)

    with open(os.path.join(output_root, "aggregated_task_layer_axis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "layer_idx", "tensor_axis", "mean_distortion", "median_distortion_secondary", "n"])
        for (task, layer, axis), stats in sorted(result["agg_primary"].items()):
            w.writerow([task, layer, axis, stats["mean"], stats["median"], stats["n"]])

    with open(os.path.join(output_root, "primary_spearman.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tensor_axis", "task", "raw_rho", "gate_rho", "n", "feature_has_variance", "target_has_variance"])
        for axis in ("key", "value"):
            for task, detail in result["axis_results"][axis]["per_task_detail"].items():
                w.writerow([axis, task, detail["raw_rho"], detail["gate_rho"], detail["n"], detail["feature_has_variance"], detail["target_has_variance"]])

    with open(os.path.join(output_root, "diagnostic_layer0.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tensor_axis", "task", "layer0_mean_distortion", "signed_sensitivity", "layer0_raw_rho_n3", "criterion_d_pass"])
        for axis in ("key", "value"):
            ar = result["axis_results"][axis]
            for task in LAYER0_DIAGNOSTIC_TASKS:
                w.writerow([axis, task, ar["layer0_x_by_task"][task], ar["layer0_y_by_task"][task],
                            ar["criterion_d"]["layer0_raw_rho"], ar["criterion_d"]["pass"]])

    with open(os.path.join(output_root, "loto.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tensor_axis", "removed_task", "negative_count", "required_negative_count", "median_remaining_gate_rho", "pass", "is_locked_criterion_c"])
        for axis in ("key", "value"):
            for removed_task, entry in result["axis_results"][axis]["secondary_full_loto"].items():
                w.writerow([axis, removed_task, entry["negative_count"], entry["required_negative_count"],
                            entry["median_remaining_gate_rho"], entry["pass"], removed_task == LOTO_REMOVED_TASK])

    gate_results = {
        "threshold_description": "a preregistered pragmatic magnitude threshold (not a universal Spearman/Cohen large-effect standard)",
        "threshold_value": DEFAULT_EFFECT_SIZE_THRESHOLD,
        "key": {
            "criterion_a": result["axis_results"]["key"]["criterion_a"],
            "criterion_b": result["axis_results"]["key"]["criterion_b"],
            "criterion_c": result["axis_results"]["key"]["criterion_c"],
            "criterion_d": result["axis_results"]["key"]["criterion_d"],
            "axis_go": result["axis_results"]["key"]["axis_go"],
        },
        "value": {
            "criterion_a": result["axis_results"]["value"]["criterion_a"],
            "criterion_b": result["axis_results"]["value"]["criterion_b"],
            "criterion_c": result["axis_results"]["value"]["criterion_c"],
            "criterion_d": result["axis_results"]["value"]["criterion_d"],
            "axis_go": result["axis_results"]["value"]["axis_go"],
        },
        "STAGE_H_ATTENTION_FEATURE": result["STAGE_H_ATTENTION_FEATURE"],
    }
    with open(os.path.join(output_root, "gate_results.json"), "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(gate_results), f, indent=2, default=str)

    with open(os.path.join(output_root, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(result), f, indent=2, default=str)


def main():
    result = run_full_analysis()
    primary_records, diagnostic_records = load_all_records()
    write_outputs(result, primary_records, diagnostic_records)
    print(f"STAGE_H_ATTENTION_FEATURE = {result['STAGE_H_ATTENTION_FEATURE']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
