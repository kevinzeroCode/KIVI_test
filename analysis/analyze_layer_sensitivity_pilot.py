"""Stage E: scientific analysis of the 8-layer K/V sensitivity pilot.

CPU-only, read-only with respect to prediction data: never loads a model,
never touches the GPU, never re-runs generation. Re-derives per-sample
scores from the pilot's existing prediction JSONLs using the EXACT same
scoring functions as eval_long_bench.py / metrics.py -- imported directly
from analysis/analyze_kv_ablation.py (sample_score, compute_task_sample_scores,
derive_seed, summarize_bootstrap), never reimplemented.

This produces layer-sensitivity statistics only. It does NOT decide whether
to launch more GPU experiments by itself -- the decision gate in report.md
is this script's output, not an action.

Usage:
    ./.venv/bin/python analysis/analyze_layer_sensitivity_pilot.py \\
        --bootstrap 10000 --seed 42 \\
        --output-dir analysis/results/layer_sensitivity_pilot
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
from utils.jsonl_integrity import inspect_jsonl  # noqa: E402
from utils.pilot_policy import (  # noqa: E402
    PILOT_AXES,
    PILOT_LAYERS,
    PILOT_TASK_COUNTS,
    discover_and_validate_pilot_policies,
    output_dir_name,
)

PILOT_TASKS = list(PILOT_TASK_COUNTS)
FP16 = "FP16"

DEFAULT_PILOT_ROOT = REPO_ROOT / "outputs" / "layer_sensitivity_pilot"
DEFAULT_POLICIES_DIR = REPO_ROOT / "analysis" / "policies" / "layer_sensitivity_pilot"
DEFAULT_FP16_BASELINE_DIR = REPO_ROOT / "pred" / "longchat-7b-v1.5-32k_31500_16bits_group32_residual128"

EARLY_LAYERS = (0, 4, 9)
MID_LAYERS = (13, 18, 22)
LATE_LAYERS = (27, 31)

# Near-zero-denominator guard for the max/min absolute-sensitivity ratio:
# below this (percentage points), the ratio is reported but flagged as
# numerically unstable rather than a meaningful magnitude comparison.
MIN_ABS_SENSITIVITY_FOR_STABLE_RATIO = 0.05


class SensitivityAnalysisError(RuntimeError):
    pass


def config_label(cfg):
    if cfg == FP16:
        return "FP16"
    layer_idx, axis = cfg
    return f"layer{layer_idx:02d}_{axis}"


# ---------------------------------------------------------------------------
# Step 1: discovery + data loading
# ---------------------------------------------------------------------------

def discover_pilot_conditions(policies_dir, pilot_root):
    conditions = discover_and_validate_pilot_policies(policies_dir, PILOT_LAYERS, PILOT_AXES)
    by_key = {}
    for c in conditions:
        key = (c["layer_idx"], c["axis"])
        c["output_dir"] = pilot_root / output_dir_name(c)
        by_key[key] = c
    return by_key


def load_task_rows(dir_path, task):
    rows = []
    path = Path(dir_path) / f"{task}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {"pred": obj["pred"], "answers": obj["answers"], "all_classes": obj["all_classes"], "length": obj.get("length")}
            )
    return rows


def validate_integrity(dir_path, tasks=PILOT_TASKS, task_counts=PILOT_TASK_COUNTS):
    problems = []
    for task in tasks:
        path = Path(dir_path) / f"{task}.jsonl"
        info = inspect_jsonl(str(path))
        if not info.exists:
            problems.append(f"{dir_path}/{task}.jsonl: missing")
            continue
        if info.invalid_rows:
            problems.append(f"{dir_path}/{task}.jsonl: {info.invalid_rows} invalid row(s)")
        if info.valid_rows != task_counts[task]:
            problems.append(f"{dir_path}/{task}.jsonl: {info.valid_rows} rows, expected exactly {task_counts[task]}")
    return problems


def load_all_predictions(fp16_dir, conditions_by_key, tasks=PILOT_TASKS):
    """Returns {config: {task: [rows]}} for FP16 plus every pilot condition,
    after strict per-file integrity validation (fails closed)."""
    problems = validate_integrity(fp16_dir, tasks)
    for key, cond in conditions_by_key.items():
        problems += validate_integrity(cond["output_dir"], tasks)
    if problems:
        raise SensitivityAnalysisError("Integrity validation failed:\n" + "\n".join(problems))

    all_data = {FP16: {t: load_task_rows(fp16_dir, t) for t in tasks}}
    for key, cond in conditions_by_key.items():
        all_data[key] = {t: load_task_rows(cond["output_dir"], t) for t in tasks}
    return all_data


def validate_pairing(all_data, tasks=PILOT_TASKS):
    """Row-order pairing: answers/all_classes/length must match FP16 exactly
    for every condition, every task, every row. Fails closed. Never
    reorders rows to force a match."""
    configs = list(all_data)
    for task in tasks:
        n_rows = len(all_data[FP16][task])
        for cfg in configs:
            if cfg == FP16:
                continue
            rows = all_data[cfg][task]
            if len(rows) != n_rows:
                raise SensitivityAnalysisError(f"{config_label(cfg)}/{task}: row count {len(rows)} != FP16's {n_rows}")
            for i in range(n_rows):
                for field in ("answers", "all_classes", "length"):
                    if rows[i][field] != all_data[FP16][task][i][field]:
                        raise SensitivityAnalysisError(
                            f"{config_label(cfg)}/{task}: row {i} field {field!r} mismatch vs FP16"
                        )
    return {"status": "PASS", "configs_checked": [config_label(c) for c in configs], "tasks_checked": tasks}


# ---------------------------------------------------------------------------
# Step 2: score recomputation
# ---------------------------------------------------------------------------

def recompute_scores(all_data, tasks=PILOT_TASKS):
    """Returns (sample_scores, task_mean_raw, task_mean_official).
    sample_scores[cfg][task]: np.array in [0,100]. task_mean_raw: unrounded
    100*mean. task_mean_official: round(...,2), matching eval_long_bench.py."""
    sample_scores = {}
    task_mean_raw = {}
    task_mean_official = {}
    for cfg, data in all_data.items():
        sample_scores[cfg] = {}
        task_mean_raw[cfg] = {}
        task_mean_official[cfg] = {}
        for task in tasks:
            scores = compute_task_sample_scores(task, data[task])
            sample_scores[cfg][task] = scores
            raw = float(scores.mean())
            task_mean_raw[cfg][task] = raw
            task_mean_official[cfg][task] = round(raw, 2)
    return sample_scores, task_mean_raw, task_mean_official


def compare_against_existing_result_json(fp16_dir, task_mean_official, tasks=PILOT_TASKS):
    """The FP16 baseline has a result.json (from the formal pred_long_bench.py
    run); pilot conditions do not (the pilot driver never calls
    eval_long_bench.py). Fails closed on any disagreement for the tasks that
    do have a comparable artifact."""
    result_path = Path(fp16_dir) / "result.json"
    if not result_path.exists():
        return {"checked": False, "reason": "no result.json found"}
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    mismatches = []
    for task in tasks:
        if task not in stored:
            continue
        if task_mean_official[FP16][task] != stored[task]:
            mismatches.append({"task": task, "recomputed": task_mean_official[FP16][task], "stored": stored[task]})
    if mismatches:
        raise SensitivityAnalysisError(f"FP16 result.json disagreement: {mismatches}")
    return {"checked": True, "n_tasks_checked": len(tasks), "n_mismatches": 0}


# ---------------------------------------------------------------------------
# Step 3: primary sensitivity definitions (point estimates, raw/unrounded)
# ---------------------------------------------------------------------------

def compute_layer_sensitivities(task_mean_raw, layers=PILOT_LAYERS, tasks=PILOT_TASKS):
    """Returns per-layer dict with SignedKeySensitivity, SignedValueSensitivity,
    AbsKeySensitivity, AbsValueSensitivity, plus the raw per-task delta
    arrays used for LOTO. Equal-task-weighted mean -- never a pooled
    1100-example raw mean."""
    result = {}
    per_task_deltas = {}  # (layer, axis) -> {task: delta}
    for layer in layers:
        key_deltas = {t: task_mean_raw[(layer, "key")][t] - task_mean_raw[FP16][t] for t in tasks}
        value_deltas = {t: task_mean_raw[(layer, "value")][t] - task_mean_raw[FP16][t] for t in tasks}
        per_task_deltas[(layer, "key")] = key_deltas
        per_task_deltas[(layer, "value")] = value_deltas

        signed_key = float(np.mean(list(key_deltas.values())))
        signed_value = float(np.mean(list(value_deltas.values())))
        abs_key = float(np.mean([abs(v) for v in key_deltas.values()]))
        abs_value = float(np.mean([abs(v) for v in value_deltas.values()]))

        result[layer] = {
            "signed_key_sensitivity": signed_key,
            "signed_value_sensitivity": signed_value,
            "abs_key_sensitivity": abs_key,
            "abs_value_sensitivity": abs_value,
        }
    return result, per_task_deltas


# ---------------------------------------------------------------------------
# Step 4: paired sample-level bootstrap (task-stratified, shared indices
# across FP16 and all 16 conditions per replicate -- same principle as
# analysis/analyze_kv_ablation.py's bootstrap_overall_scores, reimplemented
# here only because the config/task universe differs (16 pilot conditions x
# 4 tasks vs the 3x3 matrix x 15 tasks); the resampling algorithm itself is
# identical and reuses derive_seed/summarize_bootstrap directly).
# ---------------------------------------------------------------------------

def bootstrap_condition_scores(sample_scores, tasks, configs, iterations, seed):
    """Returns {config: np.ndarray of shape (iterations,)} -- the paired,
    task-stratified bootstrap distribution of each config's equal-weighted
    mean score across `tasks`."""
    boot = {cfg: np.zeros(iterations, dtype=np.float64) for cfg in configs}
    for task in tasks:
        n_rows = len(sample_scores[configs[0]][task])
        task_seed = derive_seed(seed, "layer_pilot", task)
        rng = np.random.default_rng(task_seed)
        idx = rng.integers(0, n_rows, size=(iterations, n_rows))
        for cfg in configs:
            col = sample_scores[cfg][task]
            task_boot_means = col[idx].mean(axis=1)
            boot[cfg] += task_boot_means / len(tasks)
    return boot


def apply_contrast(score_map, contrast):
    total = None
    for cfg, coef in contrast.items():
        term = coef * score_map[cfg]
        total = term if total is None else total + term
    return total


def build_key_value_contrasts(layers=PILOT_LAYERS):
    key_contrasts = OrderedDict((f"KeySensitivity(L{l:02d})", {(l, "key"): 1.0, FP16: -1.0}) for l in layers)
    value_contrasts = OrderedDict((f"ValueSensitivity(L{l:02d})", {(l, "value"): 1.0, FP16: -1.0}) for l in layers)
    axis_diff_contrasts = OrderedDict((f"AxisDifference(L{l:02d})", {(l, "key"): 1.0, (l, "value"): -1.0}) for l in layers)
    return key_contrasts, value_contrasts, axis_diff_contrasts


def summarize_contrasts_from_points(contrasts, point_scores, boot):
    out = OrderedDict()
    for name, contrast in contrasts.items():
        observed = apply_contrast(point_scores, contrast)
        boot_values = apply_contrast(boot, contrast)
        out[name] = summarize_bootstrap(observed, boot_values)
    return out


# ---------------------------------------------------------------------------
# Step 5: layer heterogeneity
# ---------------------------------------------------------------------------

def layer_heterogeneity(layer_sensitivities, layers=PILOT_LAYERS):
    def stats_for(key):
        values = {l: layer_sensitivities[l][key] for l in layers}
        ranked = sorted(layers, key=lambda l: values[l])
        min_layer, max_layer = ranked[0], ranked[-1]
        return {
            "values": values,
            "ranking_ascending": ranked,
            "min_layer": min_layer,
            "min_value": values[min_layer],
            "max_layer": max_layer,
            "max_value": values[max_layer],
            "range": values[max_layer] - values[min_layer],
            "std_dev": float(np.std(list(values.values()), ddof=1)),
        }

    signed_key = stats_for("signed_key_sensitivity")
    signed_value = stats_for("signed_value_sensitivity")
    abs_key = stats_for("abs_key_sensitivity")
    abs_value = stats_for("abs_value_sensitivity")

    def abs_ratio(abs_stats):
        min_v, max_v = abs_stats["min_value"], abs_stats["max_value"]
        if min_v < MIN_ABS_SENSITIVITY_FOR_STABLE_RATIO:
            return {"ratio": (max_v / min_v) if min_v > 0 else None, "stable": False,
                    "note": f"min absolute sensitivity ({min_v:.4f}) is below {MIN_ABS_SENSITIVITY_FOR_STABLE_RATIO} pp -- ratio is not a meaningful magnitude comparison"}
        return {"ratio": max_v / min_v, "stable": True, "note": None}

    return {
        "signed_key": signed_key,
        "signed_value": signed_value,
        "abs_key": abs_key,
        "abs_value": abs_value,
        "abs_key_max_min_ratio": abs_ratio(abs_key),
        "abs_value_max_min_ratio": abs_ratio(abs_value),
    }


# ---------------------------------------------------------------------------
# Step 6: task dependence + rank correlation (no scipy -- pure numpy)
# ---------------------------------------------------------------------------

def _rank(values):
    """Average-tie ranking (1-indexed), matching scipy.stats.rankdata's
    default 'average' method, implemented without scipy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def task_dependence_table(per_task_deltas, axis, layers=PILOT_LAYERS, tasks=PILOT_TASKS):
    """8x4 table: per_task_deltas[(layer, axis)][task] -> delta."""
    table = {l: {t: per_task_deltas[(l, axis)][t] for t in tasks} for l in layers}

    per_task_summary = {}
    for t in tasks:
        col = {l: table[l][t] for l in layers}
        ranked = sorted(layers, key=lambda l: col[l])
        per_task_summary[t] = {
            "most_harmed_layer": ranked[0], "most_harmed_value": col[ranked[0]],
            "least_harmed_or_most_improved_layer": ranked[-1], "least_harmed_or_most_improved_value": col[ranked[-1]],
            "range": col[ranked[-1]] - col[ranked[0]],
            "ranking_ascending": ranked,
        }

    pairwise_spearman = {}
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            v1 = [table[l][t1] for l in layers]
            v2 = [table[l][t2] for l in layers]
            pairwise_spearman[f"{t1}_vs_{t2}"] = spearman_corr(v1, v2)

    return {"table": table, "per_task_summary": per_task_summary, "pairwise_spearman": pairwise_spearman}


# ---------------------------------------------------------------------------
# Step 7: depth pattern (descriptive only)
# ---------------------------------------------------------------------------

def depth_pattern(layer_sensitivities, layers=PILOT_LAYERS):
    layer_idx_arr = list(layers)

    def corr_for(key):
        vals = [layer_sensitivities[l][key] for l in layers]
        return spearman_corr(layer_idx_arr, vals)

    def bucket_means(key):
        def mean_of(bucket):
            vals = [layer_sensitivities[l][key] for l in bucket if l in layer_sensitivities]
            return float(np.mean(vals)) if vals else None
        return {"early_0_4_9": mean_of(EARLY_LAYERS), "mid_13_18_22": mean_of(MID_LAYERS), "late_27_31": mean_of(LATE_LAYERS)}

    return {
        "spearman_layer_idx_vs_signed_key": corr_for("signed_key_sensitivity"),
        "spearman_layer_idx_vs_signed_value": corr_for("signed_value_sensitivity"),
        "spearman_layer_idx_vs_abs_key": corr_for("abs_key_sensitivity"),
        "spearman_layer_idx_vs_abs_value": corr_for("abs_value_sensitivity"),
        "bucket_means_signed_key": bucket_means("signed_key_sensitivity"),
        "bucket_means_signed_value": bucket_means("signed_value_sensitivity"),
        "bucket_means_abs_key": bucket_means("abs_key_sensitivity"),
        "bucket_means_abs_value": bucket_means("abs_value_sensitivity"),
        "note": "Descriptive only -- 8 sampled layers out of 32 cannot establish a smooth depth trend.",
    }


# ---------------------------------------------------------------------------
# Step 8: Key vs Value relationship (descriptive, transparent median split)
# ---------------------------------------------------------------------------

def key_value_relationship(layer_sensitivities, layers=PILOT_LAYERS):
    signed_key = [layer_sensitivities[l]["signed_key_sensitivity"] for l in layers]
    signed_value = [layer_sensitivities[l]["signed_value_sensitivity"] for l in layers]
    abs_key = [layer_sensitivities[l]["abs_key_sensitivity"] for l in layers]
    abs_value = [layer_sensitivities[l]["abs_value_sensitivity"] for l in layers]

    median_abs_key = float(np.median(abs_key))
    median_abs_value = float(np.median(abs_value))

    quadrants = {}
    for i, l in enumerate(layers):
        high_key = abs_key[i] >= median_abs_key
        high_value = abs_value[i] >= median_abs_value
        if high_key and high_value:
            label = "high_key_high_value"
        elif high_key and not high_value:
            label = "high_key_low_value"
        elif not high_key and high_value:
            label = "low_key_high_value"
        else:
            label = "low_key_low_value"
        quadrants[l] = label

    return {
        "spearman_signed_key_vs_signed_value": spearman_corr(signed_key, signed_value),
        "spearman_abs_key_vs_abs_value": spearman_corr(abs_key, abs_value),
        "median_split_quadrants": {
            "definition": "exploratory: layer classified high/low per axis by whether its AbsKeySensitivity/AbsValueSensitivity is >= the median across the 8 pilot layers",
            "median_abs_key_sensitivity": median_abs_key,
            "median_abs_value_sensitivity": median_abs_value,
            "layer_labels": quadrants,
        },
    }


# ---------------------------------------------------------------------------
# Step 9: leave-one-task-out
# ---------------------------------------------------------------------------

def leave_one_task_out_per_layer(per_task_deltas, layers=PILOT_LAYERS, axes=PILOT_AXES, tasks=PILOT_TASKS):
    results = {}
    for layer in layers:
        for axis in axes:
            deltas = per_task_deltas[(layer, axis)]
            full_mean = float(np.mean(list(deltas.values())))
            loto = {}
            for left_out in tasks:
                remaining = [v for t, v in deltas.items() if t != left_out]
                loto[left_out] = float(np.mean(remaining))
            min_task = min(loto, key=loto.get)
            max_task = max(loto, key=loto.get)
            full_sign = np.sign(full_mean)
            sign_changes = [t for t, v in loto.items() if full_sign != 0 and np.sign(v) != full_sign]
            results[(layer, axis)] = {
                "full_4task_mean": full_mean,
                "loto_values": loto,
                "min": loto[min_task], "min_task": min_task,
                "max": loto[max_task], "max_task": max_task,
                "sign_changed_when_removing": sign_changes,
            }
    return results


def leave_one_task_out_contrast(per_task_deltas, layer_a, layer_b, axis, tasks=PILOT_TASKS):
    """LOTO for a fixed most-vs-least contrast: per-task value =
    delta(layer_a, axis, t) - delta(layer_b, axis, t)."""
    per_task_contrast = {t: per_task_deltas[(layer_a, axis)][t] - per_task_deltas[(layer_b, axis)][t] for t in tasks}
    full_mean = float(np.mean(list(per_task_contrast.values())))
    loto = {}
    for left_out in tasks:
        remaining = [v for t, v in per_task_contrast.items() if t != left_out]
        loto[left_out] = float(np.mean(remaining))
    min_task = min(loto, key=loto.get)
    max_task = max(loto, key=loto.get)
    full_sign = np.sign(full_mean)
    sign_changes = [t for t, v in loto.items() if full_sign != 0 and np.sign(v) != full_sign]
    return {
        "full_4task_mean": full_mean,
        "loto_values": loto,
        "min": loto[min_task], "min_task": min_task,
        "max": loto[max_task], "max_task": max_task,
        "sign_changed_when_removing": sign_changes,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def fmt(x, nd=4):
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x)


def write_csv(path, header, rows):
    import csv as csv_mod
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot-root", type=str, default=str(DEFAULT_PILOT_ROOT))
    parser.add_argument("--policies-dir", type=str, default=str(DEFAULT_POLICIES_DIR))
    parser.add_argument("--fp16-baseline-dir", type=str, default=str(DEFAULT_FP16_BASELINE_DIR))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "analysis" / "results" / "layer_sensitivity_pilot"))
    args = parser.parse_args()

    pilot_root = Path(args.pilot_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering 16 pilot conditions...")
    conditions_by_key = discover_pilot_conditions(args.policies_dir, pilot_root)
    if len(conditions_by_key) != 16:
        raise SensitivityAnalysisError(f"Expected 16 conditions, discovered {len(conditions_by_key)}")

    print("Loading + validating integrity (17,600 expected rows)...")
    all_data = load_all_predictions(args.fp16_baseline_dir, conditions_by_key)

    print("Validating row-order pairing vs FP16...")
    pairing_result = validate_pairing(all_data)
    print(f"  pairing: {pairing_result['status']}")

    print("Recomputing scores...")
    sample_scores, task_mean_raw, task_mean_official = recompute_scores(all_data)

    print("Comparing FP16 scores against pred/.../result.json...")
    result_json_check = compare_against_existing_result_json(args.fp16_baseline_dir, task_mean_official)
    print(f"  {result_json_check}")

    print("Computing per-layer sensitivities (point estimates)...")
    layer_sensitivities, per_task_deltas = compute_layer_sensitivities(task_mean_raw)

    print(f"Running paired bootstrap ({args.bootstrap} iterations)...")
    all_configs = [FP16] + list(conditions_by_key.keys())
    boot = bootstrap_condition_scores(sample_scores, PILOT_TASKS, all_configs, args.bootstrap, args.seed)

    key_contrasts, value_contrasts, axis_diff_contrasts = build_key_value_contrasts()
    point_scores = {cfg: float(np.mean([task_mean_raw[cfg][t] for t in PILOT_TASKS])) for cfg in all_configs}

    key_boot = summarize_contrasts_from_points(key_contrasts, point_scores, boot)
    value_boot = summarize_contrasts_from_points(value_contrasts, point_scores, boot)
    axis_diff_boot = summarize_contrasts_from_points(axis_diff_contrasts, point_scores, boot)

    print("Layer heterogeneity...")
    heterogeneity = layer_heterogeneity(layer_sensitivities)

    # Fixed signed-range contrasts -- selected by POINT ESTIMATE of SIGNED
    # sensitivity, then bootstrapped as a single fixed contrast (never
    # re-selected per replicate, which would introduce winner-selection
    # bias). "Most harmed" / "most improved" describe SIGNED direction
    # (most negative / most positive delta vs FP16), never magnitude --
    # magnitude-based ranking is AbsKeySensitivity/AbsValueSensitivity
    # elsewhere in this module and must not be conflated with these.
    most_harmed_key_layer = heterogeneity["signed_key"]["min_layer"]  # most negative signed delta
    most_improved_key_layer = heterogeneity["signed_key"]["max_layer"]  # most positive signed delta
    most_harmed_value_layer = heterogeneity["signed_value"]["min_layer"]
    most_improved_value_layer = heterogeneity["signed_value"]["max_layer"]

    signed_range_contrasts = OrderedDict(
        [
            (f"SignedKeyRangeContrast_L{most_harmed_key_layer:02d}_minus_L{most_improved_key_layer:02d}",
             {(most_harmed_key_layer, "key"): 1.0, (most_improved_key_layer, "key"): -1.0}),
            (f"SignedValueRangeContrast_L{most_harmed_value_layer:02d}_minus_L{most_improved_value_layer:02d}",
             {(most_harmed_value_layer, "value"): 1.0, (most_improved_value_layer, "value"): -1.0}),
        ]
    )
    signed_range_boot = summarize_contrasts_from_points(signed_range_contrasts, point_scores, boot)

    print("Task dependence...")
    key_task_dep = task_dependence_table(per_task_deltas, "key")
    value_task_dep = task_dependence_table(per_task_deltas, "value")

    print("Depth pattern...")
    depth = depth_pattern(layer_sensitivities)

    print("Key vs Value relationship...")
    kv_relationship = key_value_relationship(layer_sensitivities)

    print("Leave-one-task-out...")
    loto_per_layer = leave_one_task_out_per_layer(per_task_deltas)
    loto_signed_key_range = leave_one_task_out_contrast(per_task_deltas, most_harmed_key_layer, most_improved_key_layer, "key")
    loto_signed_value_range = leave_one_task_out_contrast(per_task_deltas, most_harmed_value_layer, most_improved_value_layer, "value")

    # --- Outputs ---
    def loto_key_str(k):
        return f"L{k[0]:02d}_{k[1]}" if isinstance(k, tuple) else str(k)

    layer_sensitivity_rows = [
        [l, layer_sensitivities[l]["signed_key_sensitivity"], key_boot[f"KeySensitivity(L{l:02d})"]["ci_low"], key_boot[f"KeySensitivity(L{l:02d})"]["ci_high"],
         layer_sensitivities[l]["signed_value_sensitivity"], value_boot[f"ValueSensitivity(L{l:02d})"]["ci_low"], value_boot[f"ValueSensitivity(L{l:02d})"]["ci_high"],
         layer_sensitivities[l]["abs_key_sensitivity"], layer_sensitivities[l]["abs_value_sensitivity"]]
        for l in PILOT_LAYERS
    ]
    write_csv(
        output_dir / "layer_sensitivity.csv",
        ["layer_idx", "signed_key_sensitivity", "key_ci_low", "key_ci_high",
         "signed_value_sensitivity", "value_ci_low", "value_ci_high",
         "abs_key_sensitivity", "abs_value_sensitivity"],
        layer_sensitivity_rows,
    )

    task_score_rows = []
    for l in PILOT_LAYERS:
        for axis in PILOT_AXES:
            for t in PILOT_TASKS:
                baseline = task_mean_official[FP16][t]
                perturbed = task_mean_official[(l, axis)][t]
                task_score_rows.append([l, axis, t, baseline, perturbed, round(perturbed - baseline, 4)])
    write_csv(
        output_dir / "task_layer_sensitivity.csv",
        ["layer_idx", "axis", "task", "baseline_score", "perturbed_score", "delta"],
        task_score_rows,
    )

    bootstrap_rows = []
    for name, s in list(key_boot.items()) + list(value_boot.items()) + list(axis_diff_boot.items()) + list(signed_range_boot.items()):
        bootstrap_rows.append([name, s["observed"], s["ci_low"], s["ci_high"], not (s["ci_low"] <= 0 <= s["ci_high"])])
    write_csv(
        output_dir / "bootstrap_layer_sensitivity.csv",
        ["contrast", "observed", "ci_low", "ci_high", "ci_excludes_zero"],
        bootstrap_rows,
    )

    write_csv(
        output_dir / "axis_comparison.csv",
        ["layer_idx", "spearman_key_vs_value_signed", "spearman_key_vs_value_abs", "median_split_label"],
        [[l, kv_relationship["spearman_signed_key_vs_signed_value"], kv_relationship["spearman_abs_key_vs_abs_value"],
          kv_relationship["median_split_quadrants"]["layer_labels"][l]] for l in PILOT_LAYERS],
    )

    corr_rows = []
    for pair, rho in key_task_dep["pairwise_spearman"].items():
        corr_rows.append(["key", pair, rho])
    for pair, rho in value_task_dep["pairwise_spearman"].items():
        corr_rows.append(["value", pair, rho])
    write_csv(output_dir / "task_rank_correlations.csv", ["axis", "task_pair", "spearman_rho"], corr_rows)

    loto_rows = []
    for (l, axis), r in loto_per_layer.items():
        loto_rows.append([f"L{l:02d}_{axis}", r["full_4task_mean"], r["min"], r["min_task"], r["max"], r["max_task"], bool(r["sign_changed_when_removing"])])
    loto_rows.append([f"SignedKeyRangeContrast_L{most_harmed_key_layer:02d}_minus_L{most_improved_key_layer:02d}", loto_signed_key_range["full_4task_mean"], loto_signed_key_range["min"], loto_signed_key_range["min_task"], loto_signed_key_range["max"], loto_signed_key_range["max_task"], bool(loto_signed_key_range["sign_changed_when_removing"])])
    loto_rows.append([f"SignedValueRangeContrast_L{most_harmed_value_layer:02d}_minus_L{most_improved_value_layer:02d}", loto_signed_value_range["full_4task_mean"], loto_signed_value_range["min"], loto_signed_value_range["min_task"], loto_signed_value_range["max"], loto_signed_value_range["max_task"], bool(loto_signed_value_range["sign_changed_when_removing"])])
    write_csv(
        output_dir / "leave_one_task_out.csv",
        ["contrast", "full_4task_mean", "min", "min_task", "max", "max_task", "sign_changed"],
        loto_rows,
    )

    ctx = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "seed": args.seed,
        "iterations": args.bootstrap,
        "pairing_result": pairing_result,
        "result_json_check": result_json_check,
        "layer_sensitivities": layer_sensitivities,
        "heterogeneity": heterogeneity,
        "key_boot": key_boot,
        "value_boot": value_boot,
        "axis_diff_boot": axis_diff_boot,
        "signed_range_boot": signed_range_boot,
        "most_harmed_key_layer": most_harmed_key_layer, "most_improved_key_layer": most_improved_key_layer,
        "most_harmed_value_layer": most_harmed_value_layer, "most_improved_value_layer": most_improved_value_layer,
        "key_task_dep": key_task_dep,
        "value_task_dep": value_task_dep,
        "depth": depth,
        "kv_relationship": kv_relationship,
        "loto_signed_key_range": loto_signed_key_range,
        "loto_signed_value_range": loto_signed_value_range,
        "baseline_scores": {t: task_mean_official[FP16][t] for t in PILOT_TASKS},
    }

    summary = {
        "timestamp": ctx["timestamp"],
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap,
        "pairing": pairing_result,
        "result_json_check": result_json_check,
        "baseline_scores": ctx["baseline_scores"],
        "layer_sensitivities": layer_sensitivities,
        "heterogeneity": heterogeneity,
        "key_bootstrap": key_boot,
        "value_bootstrap": value_boot,
        "axis_difference_bootstrap": axis_diff_boot,
        "signed_range_contrast_bootstrap": signed_range_boot,
        "signed_range_layers": {
            "most_harmed_key_layer": most_harmed_key_layer, "most_improved_key_layer": most_improved_key_layer,
            "most_harmed_value_layer": most_harmed_value_layer, "most_improved_value_layer": most_improved_value_layer,
        },
        "task_dependence_key": key_task_dep,
        "task_dependence_value": value_task_dep,
        "depth_pattern": depth,
        "key_value_relationship": kv_relationship,
        "loto_per_layer": {f"L{l:02d}_{axis}": r for (l, axis), r in loto_per_layer.items()},
        "loto_signed_range_contrasts": {"key": loto_signed_key_range, "value": loto_signed_value_range},
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    write_report_md(output_dir / "report.md", ctx)

    print("\nSignedKeySensitivity by layer:")
    for l in PILOT_LAYERS:
        s = key_boot[f"KeySensitivity(L{l:02d})"]
        print(f"  L{l:02d}: {s['observed']:+.4f}  95% CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]")
    print("\nSignedValueSensitivity by layer:")
    for l in PILOT_LAYERS:
        s = value_boot[f"ValueSensitivity(L{l:02d})"]
        print(f"  L{l:02d}: {s['observed']:+.4f}  95% CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]")

    print(f"\nWrote outputs to {output_dir}")
    print("ANALYSIS_COMPLETE")


def write_report_md(path, ctx):
    lines = []
    lines.append("# Stage E: 8-Layer K/V Sensitivity Pilot -- Analysis")
    lines.append("")
    lines.append(f"- Generated: {ctx['timestamp']}")
    lines.append(f"- Bootstrap: {ctx['iterations']} iterations, seed={ctx['seed']}")
    lines.append("")
    lines.append(
        "One-layer-at-a-time perturbation measures LOCAL sensitivity around an "
        "otherwise-FP16 operating point. It does not directly predict multi-layer "
        "joint compression behavior."
    )
    lines.append("")

    lines.append("## A. Point-estimate observations")
    lines.append("")
    lines.append(f"FP16 baseline scores: {ctx['baseline_scores']}")
    lines.append("")
    lines.append("| layer | SignedKey | SignedValue | AbsKey | AbsValue |")
    lines.append("|---|---|---|---|---|")
    for l, v in ctx["layer_sensitivities"].items():
        lines.append(f"| {l} | {fmt(v['signed_key_sensitivity'])} | {fmt(v['signed_value_sensitivity'])} | {fmt(v['abs_key_sensitivity'])} | {fmt(v['abs_value_sensitivity'])} |")
    lines.append("")
    h = ctx["heterogeneity"]
    lines.append(f"- Signed Key: min=L{h['signed_key']['min_layer']:02d} ({fmt(h['signed_key']['min_value'])}), max=L{h['signed_key']['max_layer']:02d} ({fmt(h['signed_key']['max_value'])}), range={fmt(h['signed_key']['range'])}, std={fmt(h['signed_key']['std_dev'])}")
    lines.append(f"- Signed Value: min=L{h['signed_value']['min_layer']:02d} ({fmt(h['signed_value']['min_value'])}), max=L{h['signed_value']['max_layer']:02d} ({fmt(h['signed_value']['max_value'])}), range={fmt(h['signed_value']['range'])}, std={fmt(h['signed_value']['std_dev'])}")
    lines.append(f"- Abs Key max/min ratio: {h['abs_key_max_min_ratio']}")
    lines.append(f"- Abs Value max/min ratio: {h['abs_value_max_min_ratio']}")
    lines.append("")

    lines.append("## B. Bootstrap-supported observations (95% CI excludes 0)")
    lines.append("")

    def excludes_zero(s):
        return not (s["ci_low"] <= 0 <= s["ci_high"])

    any_supported = False
    for group_name, group in [("Key", ctx["key_boot"]), ("Value", ctx["value_boot"]), ("AxisDifference", ctx["axis_diff_boot"]), ("Signed range contrast", ctx["signed_range_boot"])]:
        for name, s in group.items():
            if excludes_zero(s):
                any_supported = True
                lines.append(f"- [{group_name}] {name}: {fmt(s['observed'])}, 95% CI [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}]")
    if not any_supported:
        lines.append("- None. Every layer/axis contrast's 95% paired-bootstrap CI crosses zero -- this pilot found no individually bootstrap-supported single-layer effect.")
    lines.append("")
    lines.append(
        "Naming note: \"MostHarmed\"/\"MostImproved\" below describe SIGNED direction "
        "(most negative / most positive delta vs FP16) of the fixed point-estimate-selected "
        "layers, never magnitude. Magnitude ranking uses AbsKeySensitivity/AbsValueSensitivity "
        "(Section A) and must not be conflated with these signed-range contrasts."
    )
    lines.append("")

    lines.append("## C. Task-dependent / LOTO-unstable observations")
    lines.append("")
    lines.append(f"- SignedKeyRangeContrast (MostHarmedKey L{ctx['most_harmed_key_layer']:02d} minus MostImprovedKey L{ctx['most_improved_key_layer']:02d}) LOTO: full={fmt(ctx['loto_signed_key_range']['full_4task_mean'])}, min={fmt(ctx['loto_signed_key_range']['min'])} ({ctx['loto_signed_key_range']['min_task']}), max={fmt(ctx['loto_signed_key_range']['max'])} ({ctx['loto_signed_key_range']['max_task']}), sign_changed={ctx['loto_signed_key_range']['sign_changed_when_removing'] or 'no'}")
    lines.append(f"- SignedValueRangeContrast (MostHarmedValue L{ctx['most_harmed_value_layer']:02d} minus MostImprovedValue L{ctx['most_improved_value_layer']:02d}) LOTO: full={fmt(ctx['loto_signed_value_range']['full_4task_mean'])}, min={fmt(ctx['loto_signed_value_range']['min'])} ({ctx['loto_signed_value_range']['min_task']}), max={fmt(ctx['loto_signed_value_range']['max'])} ({ctx['loto_signed_value_range']['max_task']}), sign_changed={ctx['loto_signed_value_range']['sign_changed_when_removing'] or 'no'}")
    lines.append("")
    lines.append("Pairwise Spearman correlations between tasks' 8-layer sensitivity vectors (descriptive, n=8 per vector):")
    lines.append("")
    lines.append(f"- Key: {ctx['key_task_dep']['pairwise_spearman']}")
    lines.append(f"- Value: {ctx['value_task_dep']['pairwise_spearman']}")
    lines.append("")

    lines.append("## D. Exploratory depth / Key-vs-Value patterns")
    lines.append("")
    d = ctx["depth"]
    lines.append(f"- Spearman(layer_idx, SignedKey) = {fmt(d['spearman_layer_idx_vs_signed_key']) if d['spearman_layer_idx_vs_signed_key'] is not None else 'n/a'} (descriptive, 8 sampled layers only -- not a smooth-depth-trend claim)")
    lines.append(f"- Spearman(layer_idx, SignedValue) = {fmt(d['spearman_layer_idx_vs_signed_value']) if d['spearman_layer_idx_vs_signed_value'] is not None else 'n/a'}")
    lines.append(f"- early/mid/late bucket means (SignedKey): {d['bucket_means_signed_key']}")
    lines.append(f"- early/mid/late bucket means (SignedValue): {d['bucket_means_signed_value']}")
    kv = ctx["kv_relationship"]
    lines.append(f"- Spearman(SignedKey, SignedValue) across 8 layers = {fmt(kv['spearman_signed_key_vs_signed_value']) if kv['spearman_signed_key_vs_signed_value'] is not None else 'n/a'}")
    lines.append(f"- Median-split quadrants (exploratory): {kv['median_split_quadrants']['layer_labels']}")
    lines.append("")

    lines.append("## What this analysis does NOT claim")
    lines.append("")
    lines.append("- Does not claim a layer is universally sensitive based on only 4 screening tasks.")
    lines.append("- Does not claim correlation implies mechanism (Spearman correlations here are descriptive, n=4 or n=8).")
    lines.append("- Does not claim a CI crossing zero proves equivalence.")
    lines.append("- Does not claim one-layer-at-a-time perturbation predicts multi-layer joint compression.")
    lines.append(
        "- Does not claim Phase-1 (the global 3x3 K/V matrix) found a real K x V interaction. "
        "The correct Phase-1 result is: K4/V2 - FP16 was the only configuration-level delta "
        "whose paired-bootstrap 95% CI excluded zero; all five explicit K x V "
        "difference-in-differences interaction CIs crossed zero."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
