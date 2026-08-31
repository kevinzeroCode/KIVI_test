"""Stage G4: pre-registered scientific feature-response analysis for the
Stage G3B layer feature pilot.

CPU-only, read-only with respect to the frozen scientific collection
(outputs/layer_feature_pilot/{primary,diagnostic}/*/features.jsonl) -- never
loads a model, never touches the GPU, never re-runs collection. Does not
modify features.jsonl, sample_selection.json, or manifest.json.

Reuses, never reimplements:
  - analysis/feature_pilot_gate.py for the EXACT pre-registered A/B/C/D gate
    (evaluate_feature_pilot_gate / evaluate_axis_gate / evaluate_criterion_a..d
    / compute_task_specific_rho / compute_layer0_diagnostic_rho / to_gate_rho),
    including its locked undefined-Spearman convention.
  - analysis.analyze_layer_sensitivity_pilot.spearman_corr indirectly, via
    feature_pilot_gate's compute_*_rho wrappers (never called independently
    here).

Sensitivity labels are read directly from the already-finalized Stage-E /
Stage-F0 CSV outputs (never recomputed, only cross-referenced):
  - analysis/results/layer_sensitivity_pilot/task_layer_sensitivity.csv
    (primary 4 tasks x 8 layers x 2 axes; also the source for lcc's Layer-0
    value used in the diagnostic table)
  - analysis/results/layer_sensitivity_f0/task_results.csv
    (multifieldqa_en / samsum Layer-0-only key_delta/value_delta)

Outputs (CSV column order and JSON structure are stable/deterministic) go
to analysis/results/layer_feature_pilot/ -- analysis/results/layer_sensitivity_pilot/
and analysis/results/layer_sensitivity_f0/ are never written to.

Usage:
    ./.venv/bin/python analysis/analyze_layer_feature_pilot.py \\
        --output-dir analysis/results/layer_feature_pilot
"""
import argparse
import csv
import glob
import json
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

from analysis.feature_pilot_gate import (  # noqa: E402 -- reuse, do not reimplement
    DEFAULT_EFFECT_SIZE_THRESHOLD,
    LAYER0_DIAGNOSTIC_TASKS,
    LOTO_REMOVED_TASK,
    PRIMARY_STAGE_E_TASKS,
    compute_layer0_diagnostic_rho,
    compute_task_specific_rho,
    evaluate_criterion_c,
    evaluate_feature_pilot_gate,
    to_gate_rho,
)
from scripts.collect_layer_features import (  # noqa: E402 -- reuse constants, no duplication
    F0_DIAGNOSTIC_LAYERS,
    F0_DIAGNOSTIC_TASKS,
    PRIMARY_STAGE_E_LAYERS,
)

DEFAULT_PRIMARY_ROOT = str(Path(REPO_ROOT) / "outputs" / "layer_feature_pilot" / "primary")
DEFAULT_DIAGNOSTIC_ROOT = str(Path(REPO_ROOT) / "outputs" / "layer_feature_pilot" / "diagnostic")
DEFAULT_STAGE_E_CSV = str(Path(REPO_ROOT) / "analysis" / "results" / "layer_sensitivity_pilot" / "task_layer_sensitivity.csv")
DEFAULT_F0_CSV = str(Path(REPO_ROOT) / "analysis" / "results" / "layer_sensitivity_f0" / "task_results.csv")
DEFAULT_OUTPUT_DIR = str(Path(REPO_ROOT) / "analysis" / "results" / "layer_feature_pilot")

CALIBRATION_SAMPLES_PER_OBSERVATION = 4
AXES = ("key", "value")

PRIMARY_RECON_FIELD = "relative_l2"
EXPLORATORY_DIST_FIELDS = ("p99_abs", "outlier_fraction")
SECONDARY_FIELDS = ("mean", "std", "variance", "max_abs", "p50_abs", "p95_abs", "mse", "max_abs_error")
ALL_AGGREGATED_FIELDS = (PRIMARY_RECON_FIELD,) + EXPLORATORY_DIST_FIELDS + SECONDARY_FIELDS


class AnalysisError(RuntimeError):
    """Any malformed/unexpected input -- fails closed rather than guessing
    a missing observation or silently tolerating a wrong sample count."""


# ---------------------------------------------------------------------------
# Pure helpers (CPU-testable without any real feature/sensitivity file)
# ---------------------------------------------------------------------------

def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def discover_single_features_jsonl(root):
    """Fails closed if zero or more than one features.jsonl is found under
    a scope's output root -- scientific analysis must read one unambiguous
    frozen artifact, never guess among several run labels.
    """
    matches = sorted(glob.glob(str(Path(root) / "*" / "features.jsonl")))
    if len(matches) != 1:
        raise AnalysisError(f"expected exactly one features.jsonl under {root}, found {len(matches)}: {matches}")
    return matches[0]


def load_feature_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_task_layer_axis(rows, samples_per_observation=CALIBRATION_SAMPLES_PER_OBSERVATION):
    """Groups feature rows by (task, layer_idx, tensor_axis) and computes
    the PRE-REGISTERED equal-sample arithmetic mean (primary) and median
    (secondary robustness) across exactly `samples_per_observation`
    calibration samples -- NEVER weighted by input_tokens/
    distribution_tokens/quantized_tokens/selection_length. Fails closed if
    any group does not have exactly the expected sample count (catches a
    missing or duplicated sample rather than silently averaging over the
    wrong population).

    Returns dict[(task, layer_idx, tensor_axis)] -> aggregated record with
    mean_<field>/median_<field> for every field in ALL_AGGREGATED_FIELDS,
    plus n_samples and the constituent sample_idx list (sorted) for
    provenance.
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r["task"], r["layer_idx"], r["tensor_axis"])
        groups[key].append(r)

    aggregated = OrderedDict()
    for key in sorted(groups):
        group_rows = groups[key]
        if len(group_rows) != samples_per_observation:
            raise AnalysisError(
                f"{key}: expected exactly {samples_per_observation} calibration samples, got {len(group_rows)}"
            )
        sample_ids = sorted(r["sample_idx"] for r in group_rows)
        if len(set(sample_ids)) != samples_per_observation:
            raise AnalysisError(f"{key}: duplicate sample_idx among calibration samples: {sample_ids}")

        entry = OrderedDict(
            [("task", key[0]), ("layer_idx", key[1]), ("tensor_axis", key[2]), ("n_samples", samples_per_observation), ("sample_indices", sample_ids)]
        )
        for field in ALL_AGGREGATED_FIELDS:
            values = [r[field] for r in group_rows]
            entry[f"mean_{field}"] = sum(values) / len(values)
            entry[f"median_{field}"] = _median(values)
        aggregated[key] = entry
    return aggregated


def load_stage_e_sensitivity(csv_path, tasks=PRIMARY_STAGE_E_TASKS, layers=PRIMARY_STAGE_E_LAYERS):
    """Reads the already-finalized Stage-E per-(layer,axis,task) delta
    directly from task_layer_sensitivity.csv -- never recomputed. delta
    IS SignedKeySensitivity/SignedValueSensitivity by this project's
    established convention (positive = quantizing improved the score vs
    FP16 / less harm, negative = harmed).

    Returns dict[(layer_idx, axis, task)] -> delta (float), restricted to
    and validated against the exact expected tasks/layers/axes grid (fails
    closed on missing or unexpected rows).
    """
    expected_keys = {(l, axis, t) for l in layers for axis in AXES for t in tasks}
    sensitivity = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["layer_idx"]), row["axis"], row["task"])
            if key in expected_keys:
                sensitivity[key] = float(row["delta"])
    missing = expected_keys - set(sensitivity)
    if missing:
        raise AnalysisError(f"task_layer_sensitivity.csv missing expected (layer,axis,task) rows: {sorted(missing)}")
    return sensitivity


def load_f0_layer0_sensitivity(csv_path, tasks=F0_DIAGNOSTIC_TASKS):
    """Reads Stage F0's Layer-0-only key_delta/value_delta for the
    diagnostic-only tasks (multifieldqa_en, samsum) directly from
    task_results.csv -- never recomputed.

    Returns dict[(axis, task)] -> delta.
    """
    sensitivity = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["task"] in tasks:
                sensitivity[("key", row["task"])] = float(row["key_delta"])
                sensitivity[("value", row["task"])] = float(row["value_delta"])
    missing = {(axis, t) for axis in AXES for t in tasks} - set(sensitivity)
    if missing:
        raise AnalysisError(f"task_results.csv missing expected (axis,task) rows: {sorted(missing)}")
    return sensitivity


def build_task_layer_matrices(aggregated, sensitivity, axis, feature_field, tasks=PRIMARY_STAGE_E_TASKS, layers=PRIMARY_STAGE_E_LAYERS):
    """Builds the x_by_task_layer / y_by_task_layer dicts
    compute_task_specific_rho expects, for ONE axis and ONE feature field
    (e.g. "mean_relative_l2"). Fails closed if an expected (task, layer)
    aggregated observation is missing.
    """
    x_by_task_layer = {t: {} for t in tasks}
    y_by_task_layer = {t: {} for t in tasks}
    for t in tasks:
        for l in layers:
            key = (t, l, axis)
            if key not in aggregated:
                raise AnalysisError(f"missing aggregated observation for {key}")
            x_by_task_layer[t][l] = aggregated[key][feature_field]
            y_by_task_layer[t][l] = sensitivity[(l, axis, t)]
    return x_by_task_layer, y_by_task_layer


def variance_nonzero(values):
    return len(set(values)) > 1


def build_rho_records(x_by_task_layer, y_by_task_layer, tasks=PRIMARY_STAGE_E_TASKS):
    """Computes raw_rho via the shared compute_task_specific_rho (never an
    independent Spearman implementation), then attaches gate_rho (via
    to_gate_rho) and the two variance-presence diagnostic flags per task.
    """
    raw_rho_by_task = compute_task_specific_rho(x_by_task_layer, y_by_task_layer, tasks=tasks)
    records = OrderedDict()
    for t in tasks:
        raw_rho = raw_rho_by_task[t]
        records[t] = OrderedDict(
            [
                ("task", t),
                ("raw_rho", raw_rho),
                ("gate_rho", to_gate_rho(raw_rho)),
                ("feature_variance_nonzero", variance_nonzero(list(x_by_task_layer[t].values()))),
                ("target_variance_nonzero", variance_nonzero(list(y_by_task_layer[t].values()))),
            ]
        )
    return records, raw_rho_by_task


def rho_summary_stats(rho_records, tasks=PRIMARY_STAGE_E_TASKS):
    gate_rhos = [rho_records[t]["gate_rho"] for t in tasks]
    negative_count = sum(1 for g in gate_rhos if g < 0)
    return OrderedDict(
        [
            ("median_gate_rho", _median(gate_rhos)),
            ("median_abs_gate_rho", _median([abs(g) for g in gate_rhos])),
            ("negative_count", negative_count),
            ("sign_consistent_negative", negative_count == len(tasks)),
            ("min_gate_rho", min(gate_rhos)),
            ("max_gate_rho", max(gate_rhos)),
        ]
    )


def build_layer0_diagnostic_table(primary_aggregated, diagnostic_aggregated, stage_e_sensitivity, f0_sensitivity, axis):
    """Three-row diagnostic table (lcc, multifieldqa_en, samsum) for ONE
    axis: lcc's row comes from the PRIMARY aggregation's Layer-0 entry
    (Stage-E-sourced sensitivity); multifieldqa_en/samsum come from the
    DIAGNOSTIC aggregation (Stage-F0-sourced sensitivity). Never mixes a
    task's feature values with the wrong scope's aggregation.
    """
    rows = []
    for task in LAYER0_DIAGNOSTIC_TASKS:
        if task == "lcc":
            entry = primary_aggregated[(task, 0, axis)]
            sensitivity = stage_e_sensitivity[(0, axis, task)]
        else:
            entry = diagnostic_aggregated[(task, 0, axis)]
            sensitivity = f0_sensitivity[(axis, task)]
        rows.append(
            OrderedDict(
                [
                    ("task", task),
                    ("axis", axis),
                    ("SignedSensitivity", sensitivity),
                    ("mean_relative_l2", entry["mean_relative_l2"]),
                    ("median_relative_l2", entry["median_relative_l2"]),
                    ("mean_p99_abs", entry["mean_p99_abs"]),
                    ("mean_outlier_fraction", entry["mean_outlier_fraction"]),
                ]
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_analysis(primary_root, diagnostic_root, stage_e_csv, f0_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_path = discover_single_features_jsonl(primary_root)
    diagnostic_path = discover_single_features_jsonl(diagnostic_root)

    primary_rows = load_feature_rows(primary_path)
    diagnostic_rows = load_feature_rows(diagnostic_path)

    if len(primary_rows) != 256:
        raise AnalysisError(f"expected exactly 256 primary feature rows, got {len(primary_rows)}")
    if len(diagnostic_rows) != 16:
        raise AnalysisError(f"expected exactly 16 diagnostic feature rows, got {len(diagnostic_rows)}")

    primary_aggregated = aggregate_task_layer_axis(primary_rows)
    diagnostic_aggregated = aggregate_task_layer_axis(diagnostic_rows)

    expected_primary_groups = len(PRIMARY_STAGE_E_TASKS) * len(PRIMARY_STAGE_E_LAYERS) * len(AXES)
    if len(primary_aggregated) != expected_primary_groups:
        raise AnalysisError(f"expected {expected_primary_groups} primary (task,layer,axis) groups, got {len(primary_aggregated)}")
    per_axis_primary_observations = len(PRIMARY_STAGE_E_TASKS) * len(PRIMARY_STAGE_E_LAYERS)  # 32
    expected_diagnostic_groups = len(F0_DIAGNOSTIC_TASKS) * len(F0_DIAGNOSTIC_LAYERS) * len(AXES)
    if len(diagnostic_aggregated) != expected_diagnostic_groups:
        raise AnalysisError(f"expected {expected_diagnostic_groups} diagnostic (task,layer,axis) groups, got {len(diagnostic_aggregated)}")

    stage_e_sensitivity = load_stage_e_sensitivity(stage_e_csv)
    f0_sensitivity = load_f0_layer0_sensitivity(f0_csv)

    # --- Primary relative_l2 task-specific Spearman (the gate-relevant table) ---
    task_spearman_rows = []
    rho_by_axis = {}
    raw_rho_by_axis = {}
    for axis in AXES:
        x, y = build_task_layer_matrices(primary_aggregated, stage_e_sensitivity, axis, "mean_relative_l2")
        records, raw_rho_by_task = build_rho_records(x, y)
        rho_by_axis[axis] = records
        raw_rho_by_axis[axis] = raw_rho_by_task
        for t in PRIMARY_STAGE_E_TASKS:
            r = records[t]
            task_spearman_rows.append(
                OrderedDict(
                    [
                        ("axis", axis),
                        ("task", t),
                        ("raw_rho", r["raw_rho"]),
                        ("gate_rho", r["gate_rho"]),
                        ("feature_variance_nonzero", r["feature_variance_nonzero"]),
                        ("target_variance_nonzero", r["target_variance_nonzero"]),
                    ]
                )
            )

    rho_summary_by_axis = {axis: rho_summary_stats(rho_by_axis[axis]) for axis in AXES}

    # --- Layer-0 diagnostic (Criterion D input) ---
    diagnostic_layer0_rows = []
    layer0_raw_rho_by_axis = {}
    for axis in AXES:
        rows3 = build_layer0_diagnostic_table(primary_aggregated, diagnostic_aggregated, stage_e_sensitivity, f0_sensitivity, axis)
        diagnostic_layer0_rows.extend(rows3)
        relative_l2_by_task = {r["task"]: r["mean_relative_l2"] for r in rows3}
        sensitivity_by_task = {r["task"]: r["SignedSensitivity"] for r in rows3}
        layer0_raw_rho_by_axis[axis] = compute_layer0_diagnostic_rho(relative_l2_by_task, sensitivity_by_task, tasks=LAYER0_DIAGNOSTIC_TASKS)

    # --- The exact pre-registered gate (reused, not reimplemented) ---
    gate_result = evaluate_feature_pilot_gate(
        raw_rho_by_axis["key"], layer0_raw_rho_by_axis["key"],
        raw_rho_by_axis["value"], layer0_raw_rho_by_axis["value"],
    )

    gate_results_rows = []
    for axis in AXES:
        ax = gate_result[axis]
        gate_results_rows.append(
            OrderedDict(
                [
                    ("axis", axis),
                    ("criterion_a_pass", ax["criterion_a"]["pass"]),
                    ("criterion_a_negative_count", ax["criterion_a"]["negative_count"]),
                    ("criterion_b_pass", ax["criterion_b"]["pass"]),
                    ("criterion_b_median_abs_gate_rho", ax["criterion_b"]["median_abs_gate_rho"]),
                    ("criterion_c_pass", ax["criterion_c"]["pass"]),
                    ("criterion_c_negative_count", ax["criterion_c"]["negative_count"]),
                    ("criterion_c_median_remaining_gate_rho", ax["criterion_c"]["median_remaining_gate_rho"]),
                    ("criterion_d_pass", ax["criterion_d"]["pass"]),
                    ("criterion_d_gate_rho", ax["criterion_d"]["layer0_gate_rho"]),
                    ("axis_go", ax["axis_go"]),
                ]
            )
        )

    # --- Exploratory family-agnostic distribution features (NEVER gate-affecting) ---
    distribution_feature_correlations_rows = []
    exploratory_summary = {}
    for axis in AXES:
        exploratory_summary[axis] = {}
        for field in EXPLORATORY_DIST_FIELDS:
            x, y = build_task_layer_matrices(primary_aggregated, stage_e_sensitivity, axis, f"mean_{field}")
            records, _ = build_rho_records(x, y)
            summary = rho_summary_stats(records)
            exploratory_summary[axis][field] = summary
            for t in PRIMARY_STAGE_E_TASKS:
                r = records[t]
                distribution_feature_correlations_rows.append(
                    OrderedDict(
                        [
                            ("axis", axis),
                            ("feature", field),
                            ("task", t),
                            ("raw_rho", r["raw_rho"]),
                            ("gate_rho", r["gate_rho"]),
                            ("label", "EXPLORATORY_NON_GATE"),
                        ]
                    )
                )

    # --- Secondary/descriptive features: aggregated values only, no leaderboard ---
    secondary_rows = []
    for key in sorted(primary_aggregated):
        entry = primary_aggregated[key]
        row = OrderedDict([("scope", "primary"), ("task", entry["task"]), ("layer_idx", entry["layer_idx"]), ("tensor_axis", entry["tensor_axis"])])
        for field in SECONDARY_FIELDS:
            row[f"mean_{field}"] = entry[f"mean_{field}"]
            row[f"median_{field}"] = entry[f"median_{field}"]
        secondary_rows.append(row)
    for key in sorted(diagnostic_aggregated):
        entry = diagnostic_aggregated[key]
        row = OrderedDict([("scope", "diagnostic"), ("task", entry["task"]), ("layer_idx", entry["layer_idx"]), ("tensor_axis", entry["tensor_axis"])])
        for field in SECONDARY_FIELDS:
            row[f"mean_{field}"] = entry[f"mean_{field}"]
            row[f"median_{field}"] = entry[f"median_{field}"]
        secondary_rows.append(row)

    # --- aggregated_features.csv: full combined table (all fields, both scopes) ---
    aggregated_rows = []
    for scope, agg in (("primary", primary_aggregated), ("diagnostic", diagnostic_aggregated)):
        for key in sorted(agg):
            entry = agg[key]
            row = OrderedDict([("scope", scope), ("task", entry["task"]), ("layer_idx", entry["layer_idx"]), ("tensor_axis", entry["tensor_axis"]), ("n_samples", entry["n_samples"]), ("sample_indices", " ".join(str(i) for i in entry["sample_indices"]))])
            for field in ALL_AGGREGATED_FIELDS:
                row[f"mean_{field}"] = entry[f"mean_{field}"]
                row[f"median_{field}"] = entry[f"median_{field}"]
            aggregated_rows.append(row)

    # --- LOTO: pre-registered Criterion C (lcc) + descriptive single-task-removal table ---
    loto_rows = []
    for axis in AXES:
        for removed in PRIMARY_STAGE_E_TASKS:
            c = evaluate_criterion_c(raw_rho_by_axis[axis], removed_task=removed)
            loto_rows.append(
                OrderedDict(
                    [
                        ("axis", axis),
                        ("removed_task", removed),
                        ("is_preregistered_criterion_c", removed == LOTO_REMOVED_TASK),
                        ("negative_count", c["negative_count"]),
                        ("median_remaining_gate_rho", c["median_remaining_gate_rho"]),
                        ("pass", c["pass"]),
                    ]
                )
            )

    # --- Write outputs ---
    write_csv(
        output_dir / "aggregated_features.csv",
        aggregated_rows,
        list(aggregated_rows[0].keys()) if aggregated_rows else [],
    )
    write_csv(output_dir / "task_spearman.csv", task_spearman_rows, list(task_spearman_rows[0].keys()))
    write_csv(output_dir / "diagnostic_layer0.csv", diagnostic_layer0_rows, list(diagnostic_layer0_rows[0].keys()))
    write_csv(
        output_dir / "distribution_feature_correlations.csv",
        distribution_feature_correlations_rows,
        list(distribution_feature_correlations_rows[0].keys()),
    )
    write_csv(output_dir / "secondary_features.csv", secondary_rows, list(secondary_rows[0].keys()))
    write_csv(output_dir / "gate_results.csv", gate_results_rows, list(gate_results_rows[0].keys()))
    write_csv(output_dir / "loto.csv", loto_rows, list(loto_rows[0].keys()))

    summary = OrderedDict(
        [
            ("generated_at", datetime.now(timezone.utc).isoformat()),
            ("primary_features_path", primary_path),
            ("diagnostic_features_path", diagnostic_path),
            ("primary_row_count", len(primary_rows)),
            ("diagnostic_row_count", len(diagnostic_rows)),
            ("primary_observations_per_axis", per_axis_primary_observations),
            ("effect_size_threshold", DEFAULT_EFFECT_SIZE_THRESHOLD),
            ("task_spearman_relative_l2", {axis: {t: rho_by_axis[axis][t] for t in PRIMARY_STAGE_E_TASKS} for axis in AXES}),
            ("rho_summary_relative_l2", rho_summary_by_axis),
            ("layer0_diagnostic_raw_rho", layer0_raw_rho_by_axis),
            ("gate_result", gate_result),
            ("exploratory_distribution_summary", exploratory_summary),
            ("loto_descriptive", loto_rows),
        ]
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

def format_rho(r):
    return "undefined" if r is None else f"{r:+.4f}"


def render_report(summary, output_dir):
    lines = []
    lines.append("# Stage G4: Pre-Registered Scientific Feature-Response Analysis\n")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Primary features: `{summary['primary_features_path']}` ({summary['primary_row_count']} rows)")
    lines.append(f"- Diagnostic features: `{summary['diagnostic_features_path']}` ({summary['diagnostic_row_count']} rows)")
    lines.append(
        "- Aggregation: equal-sample arithmetic mean across exactly 4 pre-registered calibration "
        "samples per (task, layer, axis) -- never weighted by token counts. Median reported as the "
        "pre-registered secondary robustness summary only."
    )
    lines.append(f"- {summary['primary_observations_per_axis']} primary task-layer observations per axis (4 tasks x 8 layers).\n")

    lines.append("## A. Preregistered primary reconstruction result (relative_l2)\n")
    for axis in AXES:
        lines.append(f"**{axis.capitalize()} axis, task-specific Spearman(mean relative_l2, Signed{axis.capitalize()}Sensitivity), n=8 layers each:**\n")
        lines.append("| task | raw_rho | gate_rho |")
        lines.append("|---|---|---|")
        for t in PRIMARY_STAGE_E_TASKS:
            rec = summary["task_spearman_relative_l2"][axis][t]
            lines.append(f"| {t} | {format_rho(rec['raw_rho'])} | {rec['gate_rho']:+.4f} |")
        lines.append("")
        rs = summary["rho_summary_relative_l2"][axis]
        lines.append(
            f"- median gate_rho = {rs['median_gate_rho']:+.4f}, median(abs(gate_rho)) = {rs['median_abs_gate_rho']:.4f}, "
            f"negative_count = {rs['negative_count']}/4, range [{rs['min_gate_rho']:+.4f}, {rs['max_gate_rho']:+.4f}]\n"
        )

    lines.append("## B. Exact A/B/C/D gate (analysis/feature_pilot_gate.py, unmodified)\n")
    for axis in AXES:
        g = summary["gate_result"][axis]
        lines.append(f"**{axis.capitalize()} axis:**")
        lines.append(f"- A (>=3/4 gate_rho<0): {'PASS' if g['criterion_a']['pass'] else 'FAIL'} (negative_count={g['criterion_a']['negative_count']}/4)")
        lines.append(f"- B (median(abs(gate_rho))>=0.5): {'PASS' if g['criterion_b']['pass'] else 'FAIL'} (median_abs_gate_rho={g['criterion_b']['median_abs_gate_rho']:.4f})")
        lines.append(f"- C (lcc-removed, >=2/3 negative AND median<0): {'PASS' if g['criterion_c']['pass'] else 'FAIL'} (negative_count={g['criterion_c']['negative_count']}/3, median={g['criterion_c']['median_remaining_gate_rho']:+.4f})")
        lines.append(f"- D (Layer-0 n=3 gate_rho<0): {'PASS' if g['criterion_d']['pass'] else 'FAIL'} (gate_rho={g['criterion_d']['layer0_gate_rho']:+.4f})")
        lines.append(f"- **axis_go = {g['axis_go']}**\n")
    lines.append(f"**FEATURE_PILOT_STAGE = {summary['gate_result']['FEATURE_PILOT_STAGE']}**\n")

    lines.append("## C. Layer-0 diagnostic (lcc / multifieldqa_en / samsum)\n")
    for axis in AXES:
        lines.append(f"**{axis.capitalize()} axis:** Criterion-D raw_rho = {format_rho(summary['layer0_diagnostic_raw_rho'][axis])}\n")
    lines.append("(Full 3-row-per-axis table: `diagnostic_layer0.csv`.)\n")

    lines.append("## D. Family-agnostic exploratory distribution results (EXPLORATORY / NON-GATE)\n")
    lines.append("These never affect FEATURE_PILOT_STAGE; reported for exploratory purposes only.\n")
    for axis in AXES:
        for field in EXPLORATORY_DIST_FIELDS:
            s = summary["exploratory_distribution_summary"][axis][field]
            lines.append(
                f"- {axis}/{field}: median gate_rho = {s['median_gate_rho']:+.4f}, "
                f"negative_count = {s['negative_count']}/4, range [{s['min_gate_rho']:+.4f}, {s['max_gate_rho']:+.4f}]"
            )
    lines.append("")

    lines.append("## E. Secondary/descriptive analyses\n")
    lines.append(
        "Secondary features (mean, std, variance, max_abs, p50_abs, p95_abs, mse, max_abs_error) are "
        "tabulated as aggregated values only in `secondary_features.csv` -- no correlation leaderboard "
        "was computed for them, and none can override the preregistered gate.\n"
    )
    lines.append("**LOTO (descriptive):** the preregistered Criterion C already tests lcc-removal specifically. "
                  "Additional single-task-removal summaries for all 4 tasks are in `loto.csv`; none of them "
                  "constitutes a new decision criterion.\n")

    lines.append("## F. Limitations\n")
    lines.append(
        "- Correlation does not prove mechanism. Reconstruction error is not claimed to universally predict "
        "sensitivity. Layer index is not claimed irrelevant. Task-conditioned features are not claimed to "
        "solve quantizer routing. n=8 layers per task is not claimed sufficient for a learned predictor. "
        "An undefined correlation was never treated as rho=0 scientifically -- only gate_rho=0.0 for gate "
        "evaluation. A failed gate is not claimed to prove no relationship exists at all -- only that this "
        "preregistered relative_l2 hypothesis did not clear this small pilot's preregistered bar.\n"
    )
    (Path(output_dir) / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary-root", default=DEFAULT_PRIMARY_ROOT)
    p.add_argument("--diagnostic-root", default=DEFAULT_DIAGNOSTIC_ROOT)
    p.add_argument("--stage-e-csv", default=DEFAULT_STAGE_E_CSV)
    p.add_argument("--f0-csv", default=DEFAULT_F0_CSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args(argv)


def main():
    args = parse_args()
    summary = run_analysis(args.primary_root, args.diagnostic_root, args.stage_e_csv, args.f0_csv, args.output_dir)
    render_report(summary, args.output_dir)
    print(json.dumps({"FEATURE_PILOT_STAGE": summary["gate_result"]["FEATURE_PILOT_STAGE"]}, indent=2))
    print(f"\nOutputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
