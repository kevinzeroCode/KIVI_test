"""Stage F0: task-generalization analysis for the Layer-0 K/V asymmetry
found in Stage E's 4-task pilot.

CPU-only, read-only with respect to prediction data: never loads a model,
never touches the GPU, never re-runs generation. Reuses Phase-1/Stage-E
scoring and bootstrap functions directly (imported, never reimplemented):
  - analysis.analyze_kv_ablation: compute_task_sample_scores, sample_score,
    derive_seed, summarize_bootstrap
  - analysis.analyze_layer_sensitivity_pilot: load_task_rows,
    recompute_scores, validate_pairing, bootstrap_condition_scores,
    apply_contrast, summarize_contrasts_from_points, fmt, write_csv

Does NOT rewrite analysis/results/layer_sensitivity_pilot/ (Stage E's
finalized 4-task results) or any Stage-D/F0 prediction file. Writes
exclusively under --output-dir (default analysis/results/layer_sensitivity_f0/).

Usage:
    ./.venv/bin/python analysis/analyze_layer_sensitivity_f0.py \\
        --bootstrap 10000 --seed 42 \\
        --output-dir analysis/results/layer_sensitivity_f0
"""
import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.analyze_kv_ablation import (  # noqa: E402 -- reuse, do not reimplement
    compute_task_sample_scores,
    derive_seed,
    sample_score,
    summarize_bootstrap,
)
from analysis.analyze_layer_sensitivity_pilot import (  # noqa: E402 -- reuse, do not reimplement
    FP16,
    apply_contrast,
    bootstrap_condition_scores,
    fmt,
    load_task_rows,
    recompute_scores,
    summarize_contrasts_from_points,
    write_csv,
)
from utils.jsonl_integrity import inspect_jsonl  # noqa: E402

LAYER00_KEY = "layer00_key"
LAYER00_VALUE = "layer00_value"
CONFIGS = [FP16, LAYER00_KEY, LAYER00_VALUE]

ORIGINAL_4_TASKS = ["trec", "lcc", "passage_retrieval_en", "2wikimqa"]
NEW_2_TASKS = ["multifieldqa_en", "samsum"]
COMBINED_6_TASKS = ORIGINAL_4_TASKS + NEW_2_TASKS

TASK_COUNTS = OrderedDict(
    [("trec", 200), ("lcc", 500), ("passage_retrieval_en", 200), ("2wikimqa", 200), ("multifieldqa_en", 150), ("samsum", 200)]
)

DEFAULT_FP16_DIR = REPO_ROOT / "pred" / "longchat-7b-v1.5-32k_31500_16bits_group32_residual128"
DEFAULT_STAGE_D_ROOT = REPO_ROOT / "outputs" / "layer_sensitivity_pilot"
DEFAULT_F0_ROOT = REPO_ROOT / "outputs" / "layer_sensitivity_f0"

STAGE_D_DIR_NAMES = {LAYER00_KEY: "layer00_key_4c4b21298993", LAYER00_VALUE: "layer00_value_11453b64892d"}
F0_DIR_NAMES = {LAYER00_KEY: "layer00_key_4c4b21298993", LAYER00_VALUE: "layer00_value_11453b64892d"}

MASTER_LOG_LIMITATION = (
    "The F0 master nohup log is unusable due to a shell/job-control launch "
    "artifact (it contains only the line '[1]+: command not found', not "
    "actual generation output). This is NOT claimed to be a clean log scan. "
    "Independent evidence supporting generation integrity instead: exact "
    "700/700 rows, pairing PASS, exit_code=0 for both conditions, "
    "nonfinite_logits_seen=false for both, stable boot IDs, and continuous "
    "host-monitor telemetry (~110 samples/condition at 30s cadence). Fixing "
    "the launcher's logging is a separate engineering task, not attempted here."
)


class F0AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------

def task_source_dir(config, task, fp16_dir=DEFAULT_FP16_DIR, stage_d_root=DEFAULT_STAGE_D_ROOT, f0_root=DEFAULT_F0_ROOT):
    """Resolves which directory holds `task`'s predictions for `config`.
    FP16 always comes from the formal pred/ baseline. layer00_key/value come
    from Stage D's output root for the original 4 tasks, and from Stage
    F0's output root for the 2 new tasks (each condition keeps the same
    policy-hash-suffixed directory name in both roots)."""
    if config == FP16:
        return Path(fp16_dir)
    root = stage_d_root if task in ORIGINAL_4_TASKS else f0_root
    dir_names = STAGE_D_DIR_NAMES if task in ORIGINAL_4_TASKS else F0_DIR_NAMES
    return Path(root) / dir_names[config]


def validate_pairing(all_data, tasks=COMBINED_6_TASKS):
    """Row-order pairing: answers/all_classes/length must match FP16 exactly
    for every config, every task, every row. Fails closed, never reorders
    rows to force a match. Same algorithm as
    analysis.analyze_layer_sensitivity_pilot.validate_pairing, adapted for
    this script's plain-string config keys ("FP16"/"layer00_key"/
    "layer00_value") instead of Stage E's (layer_idx, axis) tuple keys,
    which that version's config_label() cannot unpack."""
    configs = list(all_data)
    for task in tasks:
        n_rows = len(all_data[FP16][task])
        for cfg in configs:
            if cfg == FP16:
                continue
            rows = all_data[cfg][task]
            if len(rows) != n_rows:
                raise F0AnalysisError(f"{cfg}/{task}: row count {len(rows)} != FP16's {n_rows}")
            for i in range(n_rows):
                for field in ("answers", "all_classes", "length"):
                    if rows[i][field] != all_data[FP16][task][i][field]:
                        raise F0AnalysisError(f"{cfg}/{task}: row {i} field {field!r} mismatch vs FP16")
    return {"status": "PASS", "configs_checked": configs, "tasks_checked": tasks}


def validate_and_load_all(fp16_dir=DEFAULT_FP16_DIR, stage_d_root=DEFAULT_STAGE_D_ROOT, f0_root=DEFAULT_F0_ROOT, tasks=COMBINED_6_TASKS):
    problems = []
    for config in CONFIGS:
        for task in tasks:
            d = task_source_dir(config, task, fp16_dir, stage_d_root, f0_root)
            path = d / f"{task}.jsonl"
            info = inspect_jsonl(str(path))
            if not info.exists:
                problems.append(f"{config}/{task}: missing {path}")
                continue
            if info.invalid_rows:
                problems.append(f"{config}/{task}: {info.invalid_rows} invalid row(s) in {path}")
            if info.valid_rows != TASK_COUNTS[task]:
                problems.append(f"{config}/{task}: {info.valid_rows} rows in {path}, expected exactly {TASK_COUNTS[task]}")
    if problems:
        raise F0AnalysisError("Integrity validation failed:\n" + "\n".join(problems))

    all_data = {}
    for config in CONFIGS:
        all_data[config] = {}
        for task in tasks:
            d = task_source_dir(config, task, fp16_dir, stage_d_root, f0_root)
            all_data[config][task] = load_task_rows(d, task)
    return all_data


# ---------------------------------------------------------------------------
# Aggregation (equal-task-weighted, never pooled-sample)
# ---------------------------------------------------------------------------

def per_task_deltas_from_raw(task_mean_raw, tasks):
    key_deltas = {t: task_mean_raw[LAYER00_KEY][t] - task_mean_raw[FP16][t] for t in tasks}
    value_deltas = {t: task_mean_raw[LAYER00_VALUE][t] - task_mean_raw[FP16][t] for t in tasks}
    return key_deltas, value_deltas


def aggregate(key_deltas, value_deltas):
    signed_key = float(np.mean(list(key_deltas.values())))
    signed_value = float(np.mean(list(value_deltas.values())))
    abs_key = float(np.mean([abs(v) for v in key_deltas.values()]))
    abs_value = float(np.mean([abs(v) for v in value_deltas.values()]))
    axis_difference = signed_key - signed_value
    return {
        "signed_key_sensitivity": signed_key,
        "signed_value_sensitivity": signed_value,
        "abs_key_sensitivity": abs_key,
        "abs_value_sensitivity": abs_value,
        "axis_difference": axis_difference,
    }


def leave_one_task_out(key_deltas, value_deltas, tasks):
    """For the given task set, remove each task one at a time and recompute
    SignedKeySensitivity/SignedValueSensitivity/AxisDifference over the
    remainder. Same algorithm as Stage E's leave_one_task_out_per_layer,
    generalized to an explicit key_deltas/value_deltas pair instead of the
    (layer, axis)-keyed per_task_deltas structure."""
    results = {}
    for left_out in tasks:
        remaining = [t for t in tasks if t != left_out]
        k = {t: key_deltas[t] for t in remaining}
        v = {t: value_deltas[t] for t in remaining}
        agg = aggregate(k, v)
        results[left_out] = {
            "signed_key_sensitivity": agg["signed_key_sensitivity"],
            "signed_value_sensitivity": agg["signed_value_sensitivity"],
            "axis_difference": agg["axis_difference"],
            "key_sign": "positive" if agg["signed_key_sensitivity"] > 0 else ("negative" if agg["signed_key_sensitivity"] < 0 else "zero"),
            "value_sign": "positive" if agg["signed_value_sensitivity"] > 0 else ("negative" if agg["signed_value_sensitivity"] < 0 else "zero"),
            "axis_sign": "positive" if agg["axis_difference"] > 0 else ("negative" if agg["axis_difference"] < 0 else "zero"),
        }
    return results


# ---------------------------------------------------------------------------
# Pre-registered gate (Part 7) -- exact criteria, no effect-size/significance
# thresholds introduced here.
# ---------------------------------------------------------------------------

def apply_gate(combined6_agg, combined6_minus_lcc_agg, task_table):
    criteria = OrderedDict()
    criteria["criterion_1_combined6_signed_key_gt_0"] = combined6_agg["signed_key_sensitivity"] > 0
    criteria["criterion_2_combined6_signed_value_lt_0"] = combined6_agg["signed_value_sensitivity"] < 0
    criteria["criterion_3_combined6_axis_difference_gt_0"] = combined6_agg["axis_difference"] > 0
    criteria["criterion_4_lcc_removed_all_three_hold"] = (
        combined6_minus_lcc_agg["signed_key_sensitivity"] > 0
        and combined6_minus_lcc_agg["signed_value_sensitivity"] < 0
        and combined6_minus_lcc_agg["axis_difference"] > 0
    )
    criteria["criterion_5_at_least_one_new_task_axis_difference_gt_0"] = any(
        task_table[t]["axis_difference"] > 0 for t in NEW_2_TASKS
    )
    all_pass = all(criteria.values())
    return criteria, all_pass


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_report_md(path, ctx):
    lines = []
    lines.append("# Stage F0: Layer-0 K/V Asymmetry Task-Generalization Analysis")
    lines.append("")
    lines.append(f"- Generated: {ctx['timestamp']}")
    lines.append(f"- Bootstrap: {ctx['iterations']} iterations, seed={ctx['seed']}")
    lines.append("")
    lines.append(
        "One-layer-at-a-time perturbation measures LOCAL sensitivity around an "
        "otherwise-FP16 operating point, on a fixed screening task set. This "
        "analysis tests whether the Layer-0 Key/Value asymmetry found in "
        "Stage E's 4-task pilot is more task-general than that pilot alone "
        "could show -- it does not test multi-layer joint compression, and "
        "it does not establish that Layer 0 is universally special."
    )
    lines.append("")

    lines.append("## A. Original Stage-E observation (4 tasks, restated here for comparison)")
    lines.append("")
    a = ctx["agg_original4"]
    lines.append(f"- SignedKeySensitivity = {fmt(a['signed_key_sensitivity'])}, SignedValueSensitivity = {fmt(a['signed_value_sensitivity'])}, AxisDifference = {fmt(a['axis_difference'])}")
    lines.append("")

    lines.append("## B. New-task-only replication (2 tasks)")
    lines.append("")
    b = ctx["agg_new2"]
    lines.append(f"- SignedKeySensitivity = {fmt(b['signed_key_sensitivity'])}, SignedValueSensitivity = {fmt(b['signed_value_sensitivity'])}, AxisDifference = {fmt(b['axis_difference'])}")
    for t in NEW_2_TASKS:
        r = ctx["task_table"][t]
        lines.append(f"  - {t}: KeyDelta={fmt(r['key_delta'])}, ValueDelta={fmt(r['value_delta'])}, AxisDifference={fmt(r['axis_difference'])}")
    lines.append("")

    lines.append("## C. Combined six-task result")
    lines.append("")
    c = ctx["agg_combined6"]
    lines.append(f"- SignedKeySensitivity = {fmt(c['signed_key_sensitivity'])}, SignedValueSensitivity = {fmt(c['signed_value_sensitivity'])}, AxisDifference = {fmt(c['axis_difference'])}")
    lines.append("")

    lines.append("## D. lcc-removed robustness")
    lines.append("")
    o4 = ctx["orig4_minus_lcc"]
    c6 = ctx["combined6_minus_lcc"]
    lines.append(f"- Original 4 minus lcc (3 tasks): SignedKey={fmt(o4['signed_key_sensitivity'])}, SignedValue={fmt(o4['signed_value_sensitivity'])}, AxisDifference={fmt(o4['axis_difference'])}")
    lines.append(f"- Combined 6 minus lcc (5 tasks): SignedKey={fmt(c6['signed_key_sensitivity'])}, SignedValue={fmt(c6['signed_value_sensitivity'])}, AxisDifference={fmt(c6['axis_difference'])}")
    lines.append("")

    lines.append("## E. Bootstrap uncertainty (95% paired CI, not a significance gate)")
    lines.append("")
    for name, boot in [("original-4", ctx["boot_original4"]), ("new-2", ctx["boot_new2"]), ("combined-6", ctx["boot_combined6"])]:
        lines.append(f"**{name}**")
        for k in ("SignedKeySensitivity", "SignedValueSensitivity", "AxisDifference"):
            s = boot[k]
            lines.append(f"- {k}: {fmt(s['observed'])}, 95% CI [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}]" + (" (crosses zero)" if s["ci_low"] <= 0 <= s["ci_high"] else ""))
        lines.append("")

    lines.append("## Pre-registered gate")
    lines.append("")
    for name, val in ctx["gate_criteria"].items():
        lines.append(f"- {name}: {'PASS' if val else 'FAIL'}")
    lines.append("")
    lines.append(f"**EARLY_LAYER_EXPANSION = {'GO' if ctx['gate_pass'] else 'NO_GO'}**")
    lines.append("")

    lines.append("## What this analysis does NOT claim")
    lines.append("")
    lines.append("- Does not claim Layer 0 is universally special.")
    lines.append("- Does not claim Layer-0 quantization improves model quality in general.")
    lines.append("- Does not claim statistical significance proves mechanism.")
    lines.append("- Does not claim single-layer perturbation predicts multi-layer compression.")
    lines.append("- Does not treat a bootstrap CI crossing zero as proof of equivalence.")
    lines.append("")

    lines.append("## Master-log limitation")
    lines.append("")
    lines.append(MASTER_LOG_LIMITATION)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp16-dir", type=str, default=str(DEFAULT_FP16_DIR))
    parser.add_argument("--stage-d-root", type=str, default=str(DEFAULT_STAGE_D_ROOT))
    parser.add_argument("--f0-root", type=str, default=str(DEFAULT_F0_ROOT))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "analysis" / "results" / "layer_sensitivity_f0"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading + validating integrity (6 tasks x 3 configs)...")
    all_data = validate_and_load_all(args.fp16_dir, args.stage_d_root, args.f0_root)

    print("Validating row-order pairing vs FP16...")
    pairing_result = validate_pairing(all_data, tasks=COMBINED_6_TASKS)
    print(f"  pairing: {pairing_result['status']}")

    print("Recomputing scores...")
    sample_scores, task_mean_raw, task_mean_official = recompute_scores(all_data, tasks=COMBINED_6_TASKS)

    # --- Step 2: six-task table ---
    task_table = {}
    for t in COMBINED_6_TASKS:
        key_delta = task_mean_raw[LAYER00_KEY][t] - task_mean_raw[FP16][t]
        value_delta = task_mean_raw[LAYER00_VALUE][t] - task_mean_raw[FP16][t]
        task_table[t] = {
            "baseline_score": task_mean_official[FP16][t],
            "key_score": task_mean_official[LAYER00_KEY][t],
            "value_score": task_mean_official[LAYER00_VALUE][t],
            "key_delta": key_delta,
            "value_delta": value_delta,
            "axis_difference": key_delta - value_delta,
        }

    key_deltas_6, value_deltas_6 = per_task_deltas_from_raw(task_mean_raw, COMBINED_6_TASKS)
    key_deltas_4, value_deltas_4 = per_task_deltas_from_raw(task_mean_raw, ORIGINAL_4_TASKS)
    key_deltas_2, value_deltas_2 = per_task_deltas_from_raw(task_mean_raw, NEW_2_TASKS)

    print("Step 3: three aggregations...")
    agg_original4 = aggregate(key_deltas_4, value_deltas_4)
    agg_new2 = aggregate(key_deltas_2, value_deltas_2)
    agg_combined6 = aggregate(key_deltas_6, value_deltas_6)

    print(f"Step 4: paired bootstrap ({args.bootstrap} iterations)...")
    boot_original4_raw = bootstrap_condition_scores(sample_scores, ORIGINAL_4_TASKS, CONFIGS, args.bootstrap, args.seed)
    boot_new2_raw = bootstrap_condition_scores(sample_scores, NEW_2_TASKS, CONFIGS, args.bootstrap, args.seed)
    boot_combined6_raw = bootstrap_condition_scores(sample_scores, COMBINED_6_TASKS, CONFIGS, args.bootstrap, args.seed)

    def summarize_axis_contrasts(boot_raw, point_key_deltas, point_value_deltas):
        point_scores = {
            FP16: 0.0,
            LAYER00_KEY: float(np.mean(list(point_key_deltas.values()))),
            LAYER00_VALUE: float(np.mean(list(point_value_deltas.values()))),
        }
        # boot_raw already gives the per-config equal-weighted mean SCORE
        # (not delta) per replicate; convert to deltas vs FP16 per replicate
        # so contrasts are paired exactly like Stage E's.
        boot_deltas = {
            LAYER00_KEY: boot_raw[LAYER00_KEY] - boot_raw[FP16],
            LAYER00_VALUE: boot_raw[LAYER00_VALUE] - boot_raw[FP16],
        }
        key_summary = summarize_bootstrap(point_scores[LAYER00_KEY], boot_deltas[LAYER00_KEY])
        value_summary = summarize_bootstrap(point_scores[LAYER00_VALUE], boot_deltas[LAYER00_VALUE])
        axis_boot = boot_deltas[LAYER00_KEY] - boot_deltas[LAYER00_VALUE]
        axis_summary = summarize_bootstrap(point_scores[LAYER00_KEY] - point_scores[LAYER00_VALUE], axis_boot)
        return {"SignedKeySensitivity": key_summary, "SignedValueSensitivity": value_summary, "AxisDifference": axis_summary}

    boot_original4 = summarize_axis_contrasts(boot_original4_raw, key_deltas_4, value_deltas_4)
    boot_new2 = summarize_axis_contrasts(boot_new2_raw, key_deltas_2, value_deltas_2)
    boot_combined6 = summarize_axis_contrasts(boot_combined6_raw, key_deltas_6, value_deltas_6)

    print("Step 5: six-task leave-one-task-out...")
    loto_combined6 = leave_one_task_out(key_deltas_6, value_deltas_6, COMBINED_6_TASKS)
    orig4_minus_lcc_tasks = [t for t in ORIGINAL_4_TASKS if t != "lcc"]
    combined6_minus_lcc_tasks = [t for t in COMBINED_6_TASKS if t != "lcc"]
    orig4_minus_lcc = aggregate({t: key_deltas_4[t] for t in orig4_minus_lcc_tasks}, {t: value_deltas_4[t] for t in orig4_minus_lcc_tasks})
    combined6_minus_lcc = aggregate({t: key_deltas_6[t] for t in combined6_minus_lcc_tasks}, {t: value_deltas_6[t] for t in combined6_minus_lcc_tasks})

    print("Step 7: pre-registered gate...")
    gate_criteria, gate_pass = apply_gate(agg_combined6, combined6_minus_lcc, task_table)

    # --- Outputs ---
    write_csv(
        output_dir / "task_results.csv",
        ["task", "n_samples", "baseline_score", "key_score", "value_score", "key_delta", "value_delta", "axis_difference"],
        [[t, TASK_COUNTS[t], task_table[t]["baseline_score"], task_table[t]["key_score"], task_table[t]["value_score"],
          round(task_table[t]["key_delta"], 4), round(task_table[t]["value_delta"], 4), round(task_table[t]["axis_difference"], 4)]
         for t in COMBINED_6_TASKS],
    )

    write_csv(
        output_dir / "aggregation_results.csv",
        ["aggregation", "n_tasks", "signed_key_sensitivity", "signed_value_sensitivity", "abs_key_sensitivity", "abs_value_sensitivity", "axis_difference"],
        [
            ["original_4", 4, agg_original4["signed_key_sensitivity"], agg_original4["signed_value_sensitivity"], agg_original4["abs_key_sensitivity"], agg_original4["abs_value_sensitivity"], agg_original4["axis_difference"]],
            ["new_2", 2, agg_new2["signed_key_sensitivity"], agg_new2["signed_value_sensitivity"], agg_new2["abs_key_sensitivity"], agg_new2["abs_value_sensitivity"], agg_new2["axis_difference"]],
            ["combined_6", 6, agg_combined6["signed_key_sensitivity"], agg_combined6["signed_value_sensitivity"], agg_combined6["abs_key_sensitivity"], agg_combined6["abs_value_sensitivity"], agg_combined6["axis_difference"]],
        ],
    )

    bootstrap_rows = []
    for group_name, boot in [("original_4", boot_original4), ("new_2", boot_new2), ("combined_6", boot_combined6)]:
        for metric_name, s in boot.items():
            bootstrap_rows.append([group_name, metric_name, s["observed"], s["ci_low"], s["ci_high"], not (s["ci_low"] <= 0 <= s["ci_high"])])
    write_csv(output_dir / "bootstrap_results.csv", ["aggregation", "metric", "observed", "ci_low", "ci_high", "ci_excludes_zero"], bootstrap_rows)

    loto_rows = [
        [left_out, r["signed_key_sensitivity"], r["key_sign"], r["signed_value_sensitivity"], r["value_sign"], r["axis_difference"], r["axis_sign"]]
        for left_out, r in loto_combined6.items()
    ]
    write_csv(output_dir / "leave_one_task_out.csv", ["task_removed", "signed_key_sensitivity", "key_sign", "signed_value_sensitivity", "value_sign", "axis_difference", "axis_sign"], loto_rows)

    write_csv(
        output_dir / "gate_results.csv",
        ["criterion", "result"],
        [[name, "PASS" if val else "FAIL"] for name, val in gate_criteria.items()] + [["EARLY_LAYER_EXPANSION", "GO" if gate_pass else "NO_GO"]],
    )

    ctx = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "seed": args.seed,
        "iterations": args.bootstrap,
        "task_table": task_table,
        "agg_original4": agg_original4,
        "agg_new2": agg_new2,
        "agg_combined6": agg_combined6,
        "boot_original4": boot_original4,
        "boot_new2": boot_new2,
        "boot_combined6": boot_combined6,
        "loto_combined6": loto_combined6,
        "orig4_minus_lcc": orig4_minus_lcc,
        "combined6_minus_lcc": combined6_minus_lcc,
        "gate_criteria": gate_criteria,
        "gate_pass": gate_pass,
        "pairing_result": pairing_result,
    }

    summary = {
        "timestamp": ctx["timestamp"],
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap,
        "pairing": pairing_result,
        "task_results": task_table,
        "aggregation_original_4": agg_original4,
        "aggregation_new_2": agg_new2,
        "aggregation_combined_6": agg_combined6,
        "bootstrap_original_4": boot_original4,
        "bootstrap_new_2": boot_new2,
        "bootstrap_combined_6": boot_combined6,
        "leave_one_task_out_combined_6": loto_combined6,
        "original_4_minus_lcc": orig4_minus_lcc,
        "combined_6_minus_lcc": combined6_minus_lcc,
        "gate_criteria": gate_criteria,
        "gate_decision": "GO" if gate_pass else "NO_GO",
        "master_log_limitation": MASTER_LOG_LIMITATION,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    write_report_md(output_dir / "report.md", ctx)

    print("\nSix-task table:")
    for t in COMBINED_6_TASKS:
        r = task_table[t]
        print(f"  {t:22s} baseline={r['baseline_score']:.2f} key={r['key_score']:.2f} value={r['value_score']:.2f} KeyDelta={r['key_delta']:+.4f} ValueDelta={r['value_delta']:+.4f} AxisDiff={r['axis_difference']:+.4f}")

    print("\nGate criteria:")
    for name, val in gate_criteria.items():
        print(f"  {name}: {'PASS' if val else 'FAIL'}")
    print(f"\nEARLY_LAYER_EXPANSION = {'GO' if gate_pass else 'NO_GO'}")

    print(f"\nWrote outputs to {output_dir}")
    print("ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
