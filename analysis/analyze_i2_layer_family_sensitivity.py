"""Stage I2D: scientific analysis of the layer x quantizer-family
sensitivity experiment (Stage I2A/B/C).

CPU-only, read-only with respect to prediction data: never loads a model,
never touches the GPU, never re-runs generation. Re-derives per-sample
scores from the I2 experiment's prediction JSONLs using the EXACT same
scoring functions as eval_long_bench.py / metrics.py -- imported directly
from analysis/analyze_kv_ablation.py (sample_score, compute_task_sample_scores,
derive_seed, summarize_bootstrap), never reimplemented. The task-stratified,
shared-index-per-task bootstrap resampling loop mirrors
analysis/analyze_layer_sensitivity_pilot.py::bootstrap_condition_scores,
reimplemented only because the config universe differs ((layer, family)
cells, not (layer, axis) cells vs a fixed FP16 baseline).

Frozen scientific question (docs/stage_i2_pre_registration.md):
"At a fixed K2/V16 KV-cache precision, does the relative downstream effect
of Rotation-KIVI versus standard KIVI depend on decoder layer and workload?"

    Delta_family(layer, task) = score(rotation_kivi, layer, task) - score(kivi, layer, task)
    Delta_family(layer)       = UNWEIGHTED mean of Delta_family(layer, task) over the 6 tasks

C1 (Stage I1) observed, on lcc/sample 122/layer 0/K2-V16 only, that
Rotation-KIVI had higher Key decode distortion than standard KIVI. This
module never reads, reuses, or conditions on that observation -- it is not
an I2 endpoint, does not select any task/layer here, and does not bias any
computation in this file (see docs/stage_i2_pre_registration.md Section 12).

Usage:
    ./.venv/bin/python analysis/analyze_i2_layer_family_sensitivity.py \\
        --bootstrap 10000 --seed 42 \\
        --output-dir analysis/results/i2_layer_family_sensitivity
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
from utils.i2_layer_family_conditions import (  # noqa: E402
    I2_FAMILIES,
    I2_LAYERS,
    I2_TASK_COUNTS,
    discover_and_validate_i2_policies,
    output_dir_name,
)
from utils.jsonl_integrity import inspect_jsonl  # noqa: E402

I2_TASKS = list(I2_TASK_COUNTS)
KIVI = "kivi"
ROTATION = "rotation_kivi"
assert set(I2_FAMILIES) == {KIVI, ROTATION}

DEFAULT_I2_ROOT = REPO_ROOT / "outputs" / "i2_layer_family_sensitivity"
DEFAULT_POLICIES_DIR = REPO_ROOT / "analysis" / "policies" / "i2_layer_family_sensitivity"

INTERPRETATION_CATEGORIES = (
    "UNIFORM_ROTATION_ADVANTAGE",
    "UNIFORM_KIVI_ADVANTAGE",
    "MIXED_POINT_ESTIMATE_DIRECTION",
    "STRONG_CROSSOVER",
    "NO_CLEAR_FAMILY_DIFFERENCE",
)

# Recorded verbatim for provenance only -- read by no computation in this
# module. See docs/stage_i2_pre_registration.md Section 12.
C1_NON_SELECTION_NOTE = (
    "C1 (Stage I1) observed on lcc/dataset_index=122/layer=0/K2-V16 that "
    "Rotation-KIVI had higher Key decode distortion than standard KIVI. "
    "C1 distortion is NOT an I2 endpoint. C1 sample 122 does not determine "
    "I2 task selection. C1 direction does not determine the expected I2 "
    "downstream direction. No layer was removed because of C1."
)


class I2AnalysisError(RuntimeError):
    pass


def config_label(cfg):
    layer_idx, family = cfg
    return f"layer{layer_idx:02d}_{family}"


# ---------------------------------------------------------------------------
# Step 1: discovery + data loading
# ---------------------------------------------------------------------------

def discover_i2_conditions(policies_dir, i2_root, layers=I2_LAYERS, families=I2_FAMILIES):
    conditions = discover_and_validate_i2_policies(policies_dir, layers, families)
    by_key = {}
    for c in conditions:
        key = (c["layer_idx"], c["family"])
        c["output_dir"] = Path(i2_root) / output_dir_name(c)
        by_key[key] = c
    return by_key


def load_task_rows(dir_path, task):
    """Returns an ordered list of rows, each carrying an explicit
    `dataset_index` (this task's position in `range(expected_count)`, the
    order run_i2_layer_family_sensitivity.run_condition always writes in).
    An explicit field -- rather than only implicit row order, as
    analysis/analyze_layer_sensitivity_pilot.py's load_task_rows relies on
    -- so pairing/duplicate/missing-index integrity can be checked directly
    (Section 9's "pairing integrity ... task, dataset index, layer")."""
    rows = []
    path = Path(dir_path) / f"{task}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            dataset_index = obj.get("dataset_index", i)
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "pred": obj["pred"],
                    "answers": obj["answers"],
                    "all_classes": obj["all_classes"],
                    "length": obj.get("length"),
                }
            )
    return rows


def validate_integrity(dir_path, tasks=I2_TASKS, task_counts=I2_TASK_COUNTS):
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


def validate_index_integrity(rows, expected, where):
    """No duplicate dataset_index, no missing dataset_index -- the exact
    set {0, ..., expected-1} must appear, each exactly once. Fails closed
    (I2AnalysisError). A partial/short row list (len(rows) != expected) is
    caught here too (as a set of missing indices), so a truncated/partial
    file can never be silently classified complete by this module even if
    somehow not already rejected by validate_integrity."""
    indices = [r["dataset_index"] for r in rows]
    if len(rows) != expected:
        raise I2AnalysisError(f"{where}: {len(rows)} rows, expected exactly {expected} (partial output is never complete).")
    seen = set()
    duplicates = set()
    for idx in indices:
        if idx in seen:
            duplicates.add(idx)
        seen.add(idx)
    if duplicates:
        raise I2AnalysisError(f"{where}: duplicate dataset_index value(s) {sorted(duplicates)}.")
    missing = set(range(expected)) - seen
    if missing:
        raise I2AnalysisError(f"{where}: missing dataset_index value(s) {sorted(missing)}.")
    return {"status": "PASS", "expected": expected, "n_rows": len(rows), "n_unique_indices": len(seen)}


def load_all_predictions(conditions_by_key, tasks=I2_TASKS, task_counts=I2_TASK_COUNTS):
    """Returns {(layer,family): {task: [rows]}} after strict per-file
    integrity validation (fails closed) plus per-(condition, task)
    duplicate/missing dataset_index validation."""
    problems = []
    for key, cond in conditions_by_key.items():
        problems += validate_integrity(cond["output_dir"], tasks, task_counts)
    if problems:
        raise I2AnalysisError("Integrity validation failed:\n" + "\n".join(problems))

    all_data = {}
    for key, cond in conditions_by_key.items():
        all_data[key] = {}
        for t in tasks:
            rows = load_task_rows(cond["output_dir"], t)
            validate_index_integrity(rows, task_counts[t], f"{config_label(key)}/{t}")
            all_data[key][t] = rows
    return all_data


def validate_pairing(all_data, layers=I2_LAYERS, tasks=I2_TASKS):
    """Pairing integrity (Section 9): for every layer and task, the kivi
    and rotation_kivi conditions must carry the SAME set of dataset_index
    values, and at each shared dataset_index the answers/all_classes/length
    fields must match exactly. Fails closed. Never reorders or drops rows
    to force a match -- a pairing failure must surface, not be silently
    patched. Required to pass before any bootstrap/CI calculation
    (Section 9's "Require pairing integrity before any CI calculation")."""
    for layer in layers:
        for task in tasks:
            kivi_rows = {r["dataset_index"]: r for r in all_data[(layer, KIVI)][task]}
            rot_rows = {r["dataset_index"]: r for r in all_data[(layer, ROTATION)][task]}
            if set(kivi_rows) != set(rot_rows):
                raise I2AnalysisError(
                    f"layer{layer:02d}/{task}: dataset_index sets differ between kivi and rotation_kivi "
                    f"(kivi-only: {sorted(set(kivi_rows) - set(rot_rows))}, "
                    f"rotation-only: {sorted(set(rot_rows) - set(kivi_rows))})"
                )
            for idx, krow in kivi_rows.items():
                rrow = rot_rows[idx]
                for field in ("answers", "all_classes", "length"):
                    if krow[field] != rrow[field]:
                        raise I2AnalysisError(
                            f"layer{layer:02d}/{task}/dataset_index={idx}: field {field!r} mismatch "
                            "between kivi and rotation_kivi (same dataset_index must be the same sample)."
                        )
    return {"status": "PASS", "layers_checked": list(layers), "tasks_checked": list(tasks), "families_checked": [KIVI, ROTATION]}


# ---------------------------------------------------------------------------
# Step 2: score recomputation
# ---------------------------------------------------------------------------

def recompute_scores(all_data, tasks=I2_TASKS):
    """Returns (sample_scores, task_mean_raw, task_mean_official).
    sample_scores[cfg][task]: np.array in [0,100], ordered by dataset_index
    ascending (so the SAME positional index always means the SAME sample
    across every (layer, family) cell -- required for the paired
    bootstrap's shared per-task index arrays to actually pair correctly).
    task_mean_raw: unrounded 100*mean. task_mean_official: round(...,2),
    matching eval_long_bench.py."""
    sample_scores = {}
    task_mean_raw = {}
    task_mean_official = {}
    for cfg, data in all_data.items():
        sample_scores[cfg] = {}
        task_mean_raw[cfg] = {}
        task_mean_official[cfg] = {}
        for task in tasks:
            rows_sorted = sorted(data[task], key=lambda r: r["dataset_index"])
            scores = compute_task_sample_scores(task, rows_sorted)
            sample_scores[cfg][task] = scores
            raw = float(scores.mean())
            task_mean_raw[cfg][task] = raw
            task_mean_official[cfg][task] = round(raw, 2)
    return sample_scores, task_mean_raw, task_mean_official


# ---------------------------------------------------------------------------
# Step 3: Delta_family point estimates (Section 2/7/8 -- LOCKED definitions)
# ---------------------------------------------------------------------------

def compute_task_deltas(task_mean_raw, layers=I2_LAYERS, tasks=I2_TASKS):
    """Delta_family(layer, task) = rotation_kivi score - kivi score.
    Positive = Rotation-KIVI has the higher LongBench score."""
    return {(layer, task): task_mean_raw[(layer, ROTATION)][task] - task_mean_raw[(layer, KIVI)][task] for layer in layers for task in tasks}


def compute_layer_aggregate_deltas(task_deltas, layers=I2_LAYERS, tasks=I2_TASKS):
    """Delta_family(layer) = UNWEIGHTED mean across the six task-level
    deltas -- each task gets equal weight regardless of its sample count
    (lcc's 500 samples must not receive 2.5x the weight of a 200-sample
    task)."""
    return {layer: float(np.mean([task_deltas[(layer, t)] for t in tasks])) for layer in layers}


# ---------------------------------------------------------------------------
# Step 4: paired, task-stratified bootstrap (Section 9). Same principle as
# analysis/analyze_kv_ablation.py's bootstrap_overall_scores /
# analysis/analyze_layer_sensitivity_pilot.py's bootstrap_condition_scores:
# ONE shared resampled-index array per task per replicate, applied
# identically to every (layer, family) cell -- this is what keeps the
# bootstrap paired. Reimplemented (not imported) only because the config
# universe differs; derive_seed/summarize_bootstrap are imported unchanged.
# ---------------------------------------------------------------------------

def bootstrap_task_condition_scores(sample_scores, tasks, conditions, iterations, seed):
    """Returns boot[cfg][task] = np.ndarray shape (iterations,): the
    task-level paired bootstrap mean score for each (layer, family) cell,
    NOT yet aggregated across tasks (equal-task aggregation happens in a
    separate step so per-task deltas and the layer-aggregate delta share
    the exact same underlying resampled task means)."""
    boot = {cfg: {} for cfg in conditions}
    for task in tasks:
        n_rows = len(sample_scores[conditions[0]][task])
        task_seed = derive_seed(seed, "i2_layer_family", task)
        rng = np.random.default_rng(task_seed)
        idx = rng.integers(0, n_rows, size=(iterations, n_rows))
        for cfg in conditions:
            col = sample_scores[cfg][task]
            boot[cfg][task] = col[idx].mean(axis=1)
    return boot


def bootstrap_task_deltas(boot_task_scores, layers=I2_LAYERS, tasks=I2_TASKS):
    """delta_boot[(layer, task)] = rotation_kivi task-boot-mean - kivi
    task-boot-mean, per replicate -- paired because both families' arrays
    for this (layer, task) were resampled with the SAME index array."""
    return {
        (layer, task): boot_task_scores[(layer, ROTATION)][task] - boot_task_scores[(layer, KIVI)][task]
        for layer in layers
        for task in tasks
    }


def bootstrap_layer_aggregate_deltas(task_delta_boot, layers=I2_LAYERS, tasks=I2_TASKS):
    """layer_boot[layer] = equal-task-weighted mean, per replicate, of the
    six task_delta_boot[(layer, task)] arrays -- preserves task boundaries
    (Section 9: "combine the six task-level bootstrap statistics with equal
    task weighting"), never a pooled per-sample bootstrap."""
    return {layer: np.mean([task_delta_boot[(layer, t)] for t in tasks], axis=0) for layer in layers}


def bootstrap_family_effect_range(layer_boot, layers=I2_LAYERS):
    """Bootstrap distribution of max_layer Delta_family(layer) - min_layer
    Delta_family(layer), letting the argmax/argmin layer vary PER
    REPLICATE (the only way to validly bootstrap a range/extremum
    statistic -- a fixed pair of layers selected by the observed point
    estimate would reintroduce winner-selection bias). Implemented and
    reported unconditionally, before any results are inspected -- Section
    10D: "Do not invent a range threshold after generation," satisfied by
    reporting the CI with no threshold/gate attached to it."""
    stacked = np.stack([layer_boot[l] for l in layers], axis=0)  # (n_layers, iterations)
    return stacked.max(axis=0) - stacked.min(axis=0)


# ---------------------------------------------------------------------------
# Step 5: layer-dependence / crossover analysis (Section 10, 11 -- LOCKED)
# ---------------------------------------------------------------------------

def strong_crossover_layers(layer_point, layer_ci, layers=I2_LAYERS):
    """Section 10B: STRONG_CROSSOVER requires >=1 layer with a 95% CI
    entirely > 0 AND >=1 layer with a 95% CI entirely < 0."""
    positive_layers = [l for l in layers if layer_ci[l][0] > 0]
    negative_layers = [l for l in layers if layer_ci[l][1] < 0]
    return positive_layers, negative_layers


def classify_interpretation(layer_point, layer_ci, layers=I2_LAYERS):
    """Freezes the exact vocabulary/priority order of Section 11. Checked
    in this order because STRONG_CROSSOVER's defining condition (>=1 CI
    entirely >0 and >=1 CI entirely <0) always implies mixed point-estimate
    direction too -- STRONG_CROSSOVER is the more specific claim and must
    be reported instead of the weaker MIXED_POINT_ESTIMATE_DIRECTION
    whenever it holds."""
    positive_layers, negative_layers = strong_crossover_layers(layer_point, layer_ci, layers)
    strong_crossover = bool(positive_layers) and bool(negative_layers)

    points = [layer_point[l] for l in layers]
    mixed_point_direction = any(p > 0 for p in points) and any(p < 0 for p in points)
    all_positive = all(p > 0 for p in points)
    all_negative = all(p < 0 for p in points)

    if strong_crossover:
        category = "STRONG_CROSSOVER"
    elif mixed_point_direction:
        category = "MIXED_POINT_ESTIMATE_DIRECTION"
    elif all_positive:
        category = "UNIFORM_ROTATION_ADVANTAGE"
    elif all_negative:
        category = "UNIFORM_KIVI_ADVANTAGE"
    else:
        category = "NO_CLEAR_FAMILY_DIFFERENCE"

    return {
        "category": category,
        "strong_crossover": strong_crossover,
        "strong_crossover_positive_layers": positive_layers,
        "strong_crossover_negative_layers": negative_layers,
        "mixed_point_estimate_direction": mixed_point_direction,
        "all_layers_positive_point_estimate": all_positive,
        "all_layers_negative_point_estimate": all_negative,
    }


def task_conditioned_matrix(task_deltas, layers=I2_LAYERS, tasks=I2_TASKS):
    """8x6 matrix: rows=layers, columns=tasks, values = Rotation - KIVI
    score (Section 10E)."""
    return {layer: {task: task_deltas[(layer, task)] for task in tasks} for layer in layers}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(path, header, rows):
    import csv as csv_mod
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--i2-root", type=str, default=str(DEFAULT_I2_ROOT))
    parser.add_argument("--policies-dir", type=str, default=str(DEFAULT_POLICIES_DIR))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "analysis" / "results" / "i2_layer_family_sensitivity"))
    args = parser.parse_args()

    i2_root = Path(args.i2_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering 16 I2A conditions...")
    conditions_by_key = discover_i2_conditions(args.policies_dir, i2_root)
    if len(conditions_by_key) != 16:
        raise I2AnalysisError(f"Expected 16 conditions, discovered {len(conditions_by_key)}")

    print("Loading + validating integrity (23,200 expected rows) + dataset_index integrity...")
    all_data = load_all_predictions(conditions_by_key)

    print("Validating pairing (task, dataset_index, layer) between kivi and rotation_kivi...")
    pairing_result = validate_pairing(all_data)
    print(f"  pairing: {pairing_result['status']}")

    print("Recomputing scores...")
    sample_scores, task_mean_raw, task_mean_official = recompute_scores(all_data)

    print("Computing Delta_family point estimates...")
    task_deltas = compute_task_deltas(task_mean_raw)
    layer_point = compute_layer_aggregate_deltas(task_deltas)

    print(f"Running paired, task-stratified bootstrap ({args.bootstrap} iterations)...")
    all_conditions = list(conditions_by_key.keys())
    boot_task_scores = bootstrap_task_condition_scores(sample_scores, I2_TASKS, all_conditions, args.bootstrap, args.seed)
    task_delta_boot = bootstrap_task_deltas(boot_task_scores)
    layer_boot = bootstrap_layer_aggregate_deltas(task_delta_boot)
    range_boot = bootstrap_family_effect_range(layer_boot)

    task_delta_ci = {k: summarize_bootstrap(task_deltas[k], v) for k, v in task_delta_boot.items()}
    layer_ci_summary = {l: summarize_bootstrap(layer_point[l], v) for l, v in layer_boot.items()}
    layer_ci = {l: (layer_ci_summary[l]["ci_low"], layer_ci_summary[l]["ci_high"]) for l in I2_LAYERS}
    range_observed = max(layer_point.values()) - min(layer_point.values())
    range_ci_summary = summarize_bootstrap(range_observed, range_boot)

    print("Classifying interpretation category...")
    interpretation = classify_interpretation(layer_point, layer_ci)

    print("Building task-conditioned matrix...")
    matrix = task_conditioned_matrix(task_deltas)

    # --- Outputs ---
    task_score_rows = []
    for layer in I2_LAYERS:
        for family in I2_FAMILIES:
            for task in I2_TASKS:
                task_score_rows.append([layer, family, task, task_mean_official[(layer, family)][task]])
    write_csv(output_dir / "i2_task_score_matrix.csv", ["layer", "family", "task", "score"], task_score_rows)

    family_delta_rows = []
    for layer in I2_LAYERS:
        for task in I2_TASKS:
            family_delta_rows.append(
                [layer, task, task_mean_official[(layer, KIVI)][task], task_mean_official[(layer, ROTATION)][task], round(task_deltas[(layer, task)], 4)]
            )
    write_csv(output_dir / "i2_family_delta_matrix.csv", ["layer", "task", "kivi_score", "rotation_score", "delta"], family_delta_rows)

    layer_aggregate_rows = [[l, layer_point[l], layer_ci[l][0], layer_ci[l][1]] for l in I2_LAYERS]
    write_csv(output_dir / "i2_layer_aggregate.csv", ["layer", "delta_family_layer", "ci_low", "ci_high"], layer_aggregate_rows)

    bootstrap_rows = []
    for (layer, task), s in task_delta_ci.items():
        bootstrap_rows.append(["task_delta", layer, task, s["observed"], s["ci_low"], s["ci_high"], not (s["ci_low"] <= 0 <= s["ci_high"])])
    for l in I2_LAYERS:
        s = layer_ci_summary[l]
        bootstrap_rows.append(["layer_aggregate", l, "", s["observed"], s["ci_low"], s["ci_high"], not (s["ci_low"] <= 0 <= s["ci_high"])])
    bootstrap_rows.append(
        ["family_effect_range", "", "", range_ci_summary["observed"], range_ci_summary["ci_low"], range_ci_summary["ci_high"],
         not (range_ci_summary["ci_low"] <= 0 <= range_ci_summary["ci_high"])]
    )
    write_csv(output_dir / "i2_bootstrap_ci.csv", ["contrast", "layer", "task", "observed", "ci_low", "ci_high", "ci_excludes_zero"], bootstrap_rows)

    with open(output_dir / "i2_pairing_audit.json", "w", encoding="utf-8") as f:
        json.dump(pairing_result, f, indent=2)

    interpretation_summary = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "seed": args.seed,
        "bootstrap_iterations": args.bootstrap,
        "scientific_question": (
            "At a fixed K2/V16 KV-cache precision, does the relative downstream effect of "
            "Rotation-KIVI versus standard KIVI depend on decoder layer and workload?"
        ),
        "c1_non_selection_note": C1_NON_SELECTION_NOTE,
        "layers": list(I2_LAYERS),
        "tasks": I2_TASKS,
        "layer_point_estimates": layer_point,
        "layer_bootstrap_ci": {l: {"ci_low": layer_ci[l][0], "ci_high": layer_ci[l][1]} for l in I2_LAYERS},
        "task_conditioned_matrix": matrix,
        "family_effect_range": range_ci_summary,
        "interpretation": interpretation,
    }
    with open(output_dir / "i2_interpretation.json", "w", encoding="utf-8") as f:
        json.dump(interpretation_summary, f, indent=2, default=str)

    print(f"\nDelta_family(layer) point estimates + 95% CI:")
    for l in I2_LAYERS:
        s = layer_ci_summary[l]
        print(f"  L{l:02d}: {s['observed']:+.4f}  95% CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]")
    print(f"\nInterpretation category: {interpretation['category']}")
    print(f"Family-effect range: {range_ci_summary['observed']:+.4f}  95% CI [{range_ci_summary['ci_low']:+.4f}, {range_ci_summary['ci_high']:+.4f}]")
    print(f"\nWrote outputs to {output_dir}")
    print("ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
