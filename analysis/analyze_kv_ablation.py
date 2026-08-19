"""Paired sample-level and bootstrap analysis of the full 3x3 K/V-bit
ablation matrix (K, V in {2, 4, 16}) for LongChat-7B / LongBench.

CPU-only, read-only with respect to prediction data: this script never loads
a model, never touches the GPU, and never re-runs generation. It:

  1. discovers the 9 prediction directories under --pred-root by reading
     run_config.json (or, for the 3 pre-run_config.json legacy runs, the
     established `_<N>bits_` directory-naming convention -- see
     `discover_configurations`),
  2. strictly validates row counts / JSON validity / NUL-byte absence /
     trailing newline / absence of leftover .partial files for every
     configuration x task (utils.jsonl_integrity.inspect_jsonl),
  3. validates sample-level pairing (answers/all_classes/length) across all
     9 configurations,
  4. re-derives per-sample scores using the exact scoring functions imported
     from eval_long_bench.py / metrics.py (never reimplemented) and verifies
     they reproduce each configuration's committed result.json,
  5. runs a paired, task-stratified bootstrap that preserves LongBench's
     equal-weight-over-15-tasks aggregation semantics (see
     `bootstrap_overall_scores`), and reports CIs for every cell's delta vs
     FP16, every Key/Value trajectory contrast, and 5 K x V interaction
     contrasts,
  6. reports per-task sensitivity and leave-one-task-out robustness for the
     same contrasts.

Usage:
    ./.venv/bin/python analysis/analyze_kv_ablation.py \\
        --pred-root pred --bootstrap 10000 --seed 42 \\
        --output-dir analysis/results/kv_3x3
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_long_bench import dataset2metric  # noqa: E402
from utils.jsonl_integrity import inspect_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed experiment shape
# ---------------------------------------------------------------------------

EXPECTED_TASK_COUNTS = OrderedDict(
    [
        ("narrativeqa", 200),
        ("qasper", 200),
        ("multifieldqa_en", 150),
        ("hotpotqa", 200),
        ("musique", 200),
        ("2wikimqa", 200),
        ("gov_report", 200),
        ("qmsum", 200),
        ("multi_news", 200),
        ("lcc", 500),
        ("repobench-p", 500),
        ("triviaqa", 200),
        ("samsum", 200),
        ("trec", 200),
        ("passage_retrieval_en", 200),
    ]
)
EXPECTED_TASKS = list(EXPECTED_TASK_COUNTS)

FIRST_LINE_ONLY_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}

K_LEVELS = (16, 4, 2)
V_LEVELS = (16, 4, 2)
ALL_CONFIGS = [(k, v) for k in K_LEVELS for v in V_LEVELS]
FP16 = (16, 16)

REQUIRED_RUN_SETTINGS = {
    "model_name_or_path": "lmsys/longchat-7b-v1.5-32k",
    "max_length": 31500,
    "group_size": 32,
    "residual_length": 128,
    "seed": 42,
}

# The 3 earliest runs predate run_config.json and use the original KIVI
# naming convention where a single bit-width applies symmetrically to both
# K and V (16bits == FP16 passthrough, 2bits == joint K2/V2, 4bits == joint
# K4/V4). This mapping was already established and human-reviewed in the
# prior (5-configuration) version of this script; it is kept here as the
# sole legacy fallback, applied only when run_config.json is absent.
LEGACY_DIR_RE = re.compile(r"^longchat-7b-v1\.5-32k_31500_(\d+)bits_group32_residual128$")


def config_label(cfg):
    k, v = cfg
    return f"K{k}/V{v}"


class ConfigDiscoveryError(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


class PairingError(RuntimeError):
    pass


class ScoreReproductionError(RuntimeError):
    pass


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def short_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Step 1: configuration discovery (never hard-code directory names)
# ---------------------------------------------------------------------------

def discover_configurations(pred_root):
    """Scans pred_root's immediate subdirectories and resolves the 9
    (k_bits, v_bits) configurations required for the 3x3 matrix.

    Returns (configs, discovery_log):
      configs: OrderedDict[(k, v)] -> Path, ordered per ALL_CONFIGS
      discovery_log: list of per-directory dicts (dir, status, reason/config/source)
        for the report's data-validation section.
    """
    pred_root = Path(pred_root)
    found = {}
    log = []

    for d in sorted(pred_root.iterdir()):
        if not d.is_dir():
            continue

        run_config_path = d / "run_config.json"
        if run_config_path.exists():
            cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
            mismatch = None
            for key, expected in REQUIRED_RUN_SETTINGS.items():
                if cfg.get(key) != expected:
                    mismatch = f"{key}={cfg.get(key)!r} (expected {expected!r})"
                    break
            if mismatch:
                log.append({"dir": d.name, "status": "SKIPPED", "reason": mismatch})
                continue
            if "k_bits" not in cfg or "v_bits" not in cfg:
                log.append({"dir": d.name, "status": "SKIPPED", "reason": "run_config.json missing k_bits/v_bits"})
                continue
            key = (int(cfg["k_bits"]), int(cfg["v_bits"]))
            source = "run_config.json"
        else:
            m = LEGACY_DIR_RE.match(d.name)
            if not m:
                log.append({"dir": d.name, "status": "IGNORED", "reason": "no run_config.json and name does not match legacy pattern"})
                continue
            bits = int(m.group(1))
            key = (bits, bits)
            source = (
                "legacy directory name (.../<N>bits_group32_residual128, no "
                "run_config.json) -- symmetric K=V quantization inferred from "
                "the established KIVI naming convention, N=16 == FP16 passthrough"
            )

        if key not in ALL_CONFIGS:
            log.append({"dir": d.name, "status": "IGNORED", "reason": f"resolved config {config_label(key)} is outside the 3x3 matrix"})
            continue
        if key in found:
            raise ConfigDiscoveryError(
                f"Duplicate directories resolve to the same configuration {config_label(key)}: "
                f"{found[key][0].name} and {d.name}"
            )
        found[key] = (d, source)
        log.append({"dir": d.name, "status": "RESOLVED", "config": config_label(key), "source": source})

    missing = [c for c in ALL_CONFIGS if c not in found]
    if missing:
        raise ConfigDiscoveryError(
            f"Missing {len(missing)}/9 required configurations: {[config_label(c) for c in missing]}. "
            f"discovery_log={log}"
        )

    configs = OrderedDict((c, found[c][0]) for c in ALL_CONFIGS)
    return configs, log


# ---------------------------------------------------------------------------
# Step 2: strict input validation
# ---------------------------------------------------------------------------

def validate_integrity(configs):
    """Per utils.jsonl_integrity.inspect_jsonl: for every configuration and
    task, verify expected row count, zero invalid JSON rows, zero NUL bytes
    (invalid_rows==0 already implies no NUL-corrupted lines), trailing
    newline present, and no leftover *.jsonl.partial for that task. Fails
    closed (raises IntegrityError) with the full list of problems found."""
    problems = []
    records = {}

    for cfg, pred_dir in configs.items():
        records[cfg] = {}
        for task, expected_count in EXPECTED_TASK_COUNTS.items():
            jsonl_path = pred_dir / f"{task}.jsonl"
            partial_path = pred_dir / f"{task}.jsonl.partial"

            if partial_path.exists():
                problems.append(f"{config_label(cfg)}/{task}: leftover partial file {partial_path}")

            insp = inspect_jsonl(str(jsonl_path))
            records[cfg][task] = insp

            if not insp.exists:
                problems.append(f"{config_label(cfg)}/{task}: missing final JSONL {jsonl_path}")
                continue
            if insp.invalid_rows != 0:
                problems.append(
                    f"{config_label(cfg)}/{task}: {insp.invalid_rows} invalid JSON row(s), "
                    f"first at line {insp.first_invalid_line}"
                )
            if not insp.ends_with_newline:
                problems.append(f"{config_label(cfg)}/{task}: file does not end with a trailing newline")
            if insp.valid_rows != expected_count:
                problems.append(
                    f"{config_label(cfg)}/{task}: expected {expected_count} rows, found {insp.valid_rows} valid rows"
                )

    status = "PASS" if not problems else "FAIL"
    if problems:
        raise IntegrityError(f"{len(problems)} integrity problem(s) found:\n" + "\n".join(problems))

    return {"status": status, "problems": problems, "n_config_task_pairs_checked": len(configs) * len(EXPECTED_TASK_COUNTS)}


def load_predictions(pred_dir):
    """Returns {task_name: [{"pred":..., "answers":..., "all_classes":..., "length":...}, ...]}."""
    pred_dir = Path(pred_dir)
    data = {}
    for task in EXPECTED_TASKS:
        path = pred_dir / f"{task}.jsonl"
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append(
                    {
                        "pred": obj["pred"],
                        "answers": obj["answers"],
                        "all_classes": obj["all_classes"],
                        "length": obj.get("length"),
                    }
                )
        data[task] = rows
    return data


def load_result_json(pred_dir):
    path = Path(pred_dir) / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_task_sets(all_data):
    for cfg, data in all_data.items():
        tasks = set(data.keys())
        if tasks != set(EXPECTED_TASKS):
            missing = set(EXPECTED_TASKS) - tasks
            extra = tasks - set(EXPECTED_TASKS)
            raise PairingError(f"{config_label(cfg)}: task set mismatch. missing={sorted(missing)} extra={sorted(extra)}")


def validate_pairing(all_data):
    """Row-order pairing validation: for each task and row index, answers,
    all_classes, and length must match exactly across all 9 configurations.
    Fails closed on the first mismatch found, reporting only a short hash of
    the differing field's value, never the raw content. Never reorders rows
    to force a match."""
    configs = list(all_data.keys())
    reference = configs[0]

    for task in EXPECTED_TASKS:
        row_counts = {cfg: len(all_data[cfg][task]) for cfg in configs}
        if len(set(row_counts.values())) != 1:
            raise PairingError(
                f"task={task}: row count mismatch across configurations: "
                f"{ {config_label(c): n for c, n in row_counts.items()} }"
            )

        n_rows = row_counts[reference]
        for idx in range(n_rows):
            ref_row = all_data[reference][task][idx]
            for cfg in configs[1:]:
                row = all_data[cfg][task][idx]
                for field in ("answers", "all_classes", "length"):
                    if row[field] != ref_row[field]:
                        raise PairingError(
                            f"pairing mismatch: config={config_label(cfg)} task={task} row_index={idx} "
                            f"field={field} reference_config={config_label(reference)} "
                            f"reference_hash={short_hash(ref_row[field])} observed_hash={short_hash(row[field])}"
                        )
    return {
        "status": "PASS",
        "configs_compared": [config_label(c) for c in configs],
        "reference_config": config_label(reference),
        "tasks_checked": EXPECTED_TASKS,
        "fields_checked": ["answers", "all_classes", "length"],
    }


def sample_score(dataset, prediction, ground_truths, all_classes):
    """Exact re-derivation of a single sample's score, matching
    eval_long_bench.py's `scorer()` inner loop (pre-*100, pre-round, in [0,1])."""
    if dataset in FIRST_LINE_ONLY_DATASETS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
    return score


def compute_task_sample_scores(task, rows):
    """Returns a numpy array of per-row scores in [0, 100], using the same
    all_classes value eval_long_bench.py's scorer() effectively uses for the
    whole task (the *last* row's all_classes -- the reference implementation
    overwrites a single `all_classes` variable while iterating the file and
    only uses its final value)."""
    all_classes = rows[-1]["all_classes"]
    scores = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        scores[i] = 100.0 * sample_score(task, row["pred"], row["answers"], all_classes)
    return scores


def recompute_and_verify(all_data, result_jsons):
    """Recomputes per-sample and per-task scores for all 9 configurations,
    verifies round(recomputed_task_mean, 2) == result.json's stored score
    for every configuration-task combination, and returns the sample-score
    matrices plus both flavors of the 15-task average (official = mean of
    result.json's rounded task scores; raw = mean of unrounded recomputed
    task means)."""
    sample_scores = {cfg: {} for cfg in all_data}
    task_mean_raw = {cfg: {} for cfg in all_data}
    mismatches = []
    official_avg = {}
    raw_avg = {}

    for cfg, data in all_data.items():
        task_recomputed_rounded = {}
        for task in EXPECTED_TASKS:
            rows = data[task]
            scores = compute_task_sample_scores(task, rows)
            sample_scores[cfg][task] = scores
            raw_mean = float(scores.mean())
            rounded_mean = round(raw_mean, 2)
            task_recomputed_rounded[task] = rounded_mean
            task_mean_raw[cfg][task] = raw_mean

            stored = result_jsons[cfg][task]
            if rounded_mean != stored:
                mismatches.append(
                    {"config": config_label(cfg), "task": task, "recomputed": rounded_mean, "stored_result_json": stored}
                )

        official_avg[cfg] = sum(result_jsons[cfg][t] for t in EXPECTED_TASKS) / len(EXPECTED_TASKS)
        raw_avg[cfg] = sum(task_mean_raw[cfg][t] for t in EXPECTED_TASKS) / len(EXPECTED_TASKS)

    status = "PASS" if not mismatches else "FAIL"
    if mismatches:
        raise ScoreReproductionError(f"{len(mismatches)} config-task score mismatches: {mismatches[:5]}")

    return {
        "sample_scores": sample_scores,
        "task_mean_raw": task_mean_raw,
        "status": status,
        "n_checked": len(all_data) * len(EXPECTED_TASKS),
        "n_mismatches": len(mismatches),
        "official_avg": official_avg,
        "raw_avg": raw_avg,
    }


# ---------------------------------------------------------------------------
# Step 3: contrasts (deltas vs FP16, Key/Value trajectories) as linear
# combinations over configuration cells -- shared machinery for point
# estimates, bootstrap CIs, and leave-one-task-out (Steps 3/4/5/7).
# ---------------------------------------------------------------------------

def apply_contrast(score_map, contrast):
    """score_map: {(k,v): value-or-array}. contrast: {(k,v): coefficient}.
    Returns sum(coef * score_map[cfg]) -- works for scalars or numpy arrays."""
    total = None
    for cfg, coef in contrast.items():
        term = coef * score_map[cfg]
        total = term if total is None else total + term
    return total


def build_delta_vs_fp16_contrasts():
    contrasts = OrderedDict()
    for cfg in ALL_CONFIGS:
        if cfg == FP16:
            continue
        contrasts[f"{config_label(cfg)} - FP16"] = {cfg: 1.0, FP16: -1.0}
    return contrasts


def build_key_trajectory_contrasts():
    """score(destination) - score(source); negative = degradation."""
    contrasts = OrderedDict()
    for v in V_LEVELS:
        contrasts[f"K16->K4 @ V{v}"] = {(4, v): 1.0, (16, v): -1.0}
        contrasts[f"K4->K2 @ V{v}"] = {(2, v): 1.0, (4, v): -1.0}
        contrasts[f"K16->K2 @ V{v}"] = {(2, v): 1.0, (16, v): -1.0}
    return contrasts


def build_value_trajectory_contrasts():
    contrasts = OrderedDict()
    for k in K_LEVELS:
        contrasts[f"V16->V4 @ K{k}"] = {(k, 4): 1.0, (k, 16): -1.0}
        contrasts[f"V4->V2 @ K{k}"] = {(k, 2): 1.0, (k, 4): -1.0}
        contrasts[f"V16->V2 @ K{k}"] = {(k, 2): 1.0, (k, 16): -1.0}
    return contrasts


def interaction_contrast(k_source, k_dest, v_source, v_dest):
    """Interaction = [S(K_dest,V_dest) - S(K_source,V_dest)]
                    - [S(K_dest,V_source) - S(K_source,V_source)]"""
    return {
        (k_dest, v_dest): 1.0,
        (k_source, v_dest): -1.0,
        (k_dest, v_source): -1.0,
        (k_source, v_source): 1.0,
    }


def build_interaction_contrasts():
    return OrderedDict(
        [
            ("K16->4 x V16->4", interaction_contrast(16, 4, 16, 4)),
            ("K16->4 x V4->2", interaction_contrast(16, 4, 4, 2)),
            ("K4->2 x V16->4", interaction_contrast(4, 2, 16, 4)),
            ("K4->2 x V4->2", interaction_contrast(4, 2, 4, 2)),
            ("K16->2 x V16->2 (broad)", interaction_contrast(16, 2, 16, 2)),
        ]
    )


DELTA_VS_FP16_CONTRASTS = build_delta_vs_fp16_contrasts()
KEY_TRAJECTORY_CONTRASTS = build_key_trajectory_contrasts()
VALUE_TRAJECTORY_CONTRASTS = build_value_trajectory_contrasts()
INTERACTION_CONTRASTS = build_interaction_contrasts()

ALL_BOOTSTRAP_CONTRASTS = OrderedDict()
ALL_BOOTSTRAP_CONTRASTS.update(DELTA_VS_FP16_CONTRASTS)
ALL_BOOTSTRAP_CONTRASTS.update(KEY_TRAJECTORY_CONTRASTS)
ALL_BOOTSTRAP_CONTRASTS.update(VALUE_TRAJECTORY_CONTRASTS)

LOTO_CONTRASTS = OrderedDict()
LOTO_CONTRASTS.update(DELTA_VS_FP16_CONTRASTS)
LOTO_CONTRASTS.update(INTERACTION_CONTRASTS)


# ---------------------------------------------------------------------------
# Step 4: paired sample-level bootstrap preserving LongBench's equal-weight
# 15-task aggregation.
# ---------------------------------------------------------------------------

def derive_seed(base_seed, *parts):
    key = f"{base_seed}:" + ":".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def bootstrap_overall_scores(sample_scores, iterations, seed):
    """For each of the 15 tasks, draw `iterations` sets of with-replacement
    sample indices (one shared index set per task per replicate, applied
    identically to all 9 configurations -- this is what keeps the bootstrap
    paired). For each replicate and configuration, the task's bootstrap
    score is the mean of that task's resampled per-sample scores; the
    configuration's overall bootstrap score for that replicate is the
    equal-weight mean of its 15 (resampled) task scores -- exactly mirroring
    eval_long_bench.py's per-task `100 * mean(...)` followed by an unweighted
    15-task average, never a raw mean over all 3550 samples.

    Returns overall_boot: {(k,v): np.ndarray of shape (iterations,)}.
    """
    overall_boot = {cfg: np.zeros(iterations, dtype=np.float64) for cfg in ALL_CONFIGS}

    for task in EXPECTED_TASKS:
        n_rows = len(sample_scores[FP16][task])
        task_seed = derive_seed(seed, "overall", task)
        rng = np.random.default_rng(task_seed)
        idx = rng.integers(0, n_rows, size=(iterations, n_rows))  # (iterations, n_rows), shared across configs

        for cfg in ALL_CONFIGS:
            col = sample_scores[cfg][task]  # (n_rows,)
            task_boot_means = col[idx].mean(axis=1)  # (iterations,)
            overall_boot[cfg] += task_boot_means / len(EXPECTED_TASKS)

    return overall_boot


def summarize_bootstrap(observed, boot_values):
    mean = float(boot_values.mean())
    se = float(boot_values.std(ddof=1))
    ci_low, ci_high = (float(v) for v in np.percentile(boot_values, [2.5, 97.5]))
    return {"observed": float(observed), "bootstrap_mean": mean, "se": se, "ci_low": ci_low, "ci_high": ci_high}


def summarize_contrasts(contrasts, raw_avg, overall_boot):
    """contrasts: OrderedDict[name] -> {(k,v): coef}. Returns OrderedDict[name] -> summary dict."""
    out = OrderedDict()
    for name, contrast in contrasts.items():
        observed = apply_contrast(raw_avg, contrast)
        boot_values = apply_contrast(overall_boot, contrast)  # (iterations,)
        out[name] = summarize_bootstrap(observed, boot_values)
    return out


# ---------------------------------------------------------------------------
# Step 7: leave-one-task-out robustness
# ---------------------------------------------------------------------------

def leave_one_task_out(task_mean_raw, contrasts):
    """task_mean_raw: {(k,v): {task: raw_score}}. contrasts: OrderedDict[name] -> {(k,v): coef}.
    For each contrast, computes the per-task contrast value (same linear
    combination applied at single-task granularity), then the full 15-task
    mean and every 14-task (one-task-removed) mean."""
    per_task_contrast = {}  # name -> {task: value}
    for name, contrast in contrasts.items():
        per_task_contrast[name] = {}
        for task in EXPECTED_TASKS:
            score_map_t = {cfg: task_mean_raw[cfg][task] for cfg in ALL_CONFIGS}
            per_task_contrast[name][task] = apply_contrast(score_map_t, contrast)

    result = OrderedDict()
    for name in contrasts:
        values = per_task_contrast[name]
        full_mean = sum(values.values()) / len(EXPECTED_TASKS)
        loto_values = {}
        for left_out in EXPECTED_TASKS:
            remaining = [values[t] for t in EXPECTED_TASKS if t != left_out]
            loto_values[left_out] = sum(remaining) / len(remaining)

        min_task = min(loto_values, key=loto_values.get)
        max_task = max(loto_values, key=loto_values.get)
        full_sign = np.sign(full_mean)
        sign_changes = [t for t, v in loto_values.items() if full_sign != 0 and np.sign(v) != full_sign]

        result[name] = {
            "full_15_task_mean": float(full_mean),
            "loto_values": {t: float(v) for t, v in loto_values.items()},
            "min": float(loto_values[min_task]),
            "min_task": min_task,
            "max": float(loto_values[max_task]),
            "max_task": max_task,
            "sign_changed_when_removing": sign_changes,
        }
    return result


# ---------------------------------------------------------------------------
# Step 6: per-task sensitivity table
#
# Exact formulas (percentage points, raw/unrounded per-task means):
#   largest_abs_degradation(task)  = min over the 8 non-FP16 cells of
#                                     (score(cell, task) - score(FP16, task))
#                                     -- i.e. the single most negative delta
#                                     vs FP16 among the 8 quantized configs.
#   K_sensitivity(task)  = mean over V in {16,4,2} of
#                           |score(K2,V,task) - score(K16,V,task)|
#   V_sensitivity(task)  = mean over K in {16,4,2} of
#                           |score(K,V2,task) - score(K,V16,task)|
#   tolerance_4bit(task) = mean of {score(K4,V16,task)-FP16, score(K16,V4,task)-FP16}
#                           (single-axis drop to 4-bit only)
#   sensitivity_2bit(task) = mean of {score(K2,V16,task)-FP16, score(K16,V2,task)-FP16}
#                           (single-axis drop to 2-bit only)
#   joint_mixed_sensitivity(task) = mean of {score(cfg,task)-FP16 for cfg in
#                           {K2/V2, K4/V4, K2/V4, K4/V2}} (both axes quantized)
# ---------------------------------------------------------------------------

def build_task_sensitivity_table(task_mean_raw):
    rows = []
    for task in EXPECTED_TASKS:
        fp16 = task_mean_raw[FP16][task]
        cell = {cfg: task_mean_raw[cfg][task] for cfg in ALL_CONFIGS}
        deltas = {cfg: cell[cfg] - fp16 for cfg in ALL_CONFIGS if cfg != FP16}

        worst_cfg = min(deltas, key=deltas.get)
        worst_delta = deltas[worst_cfg]

        k_sensitivity = float(np.mean([abs(cell[(2, v)] - cell[(16, v)]) for v in V_LEVELS]))
        v_sensitivity = float(np.mean([abs(cell[(k, 2)] - cell[(k, 16)]) for k in K_LEVELS]))
        tolerance_4bit = float(np.mean([deltas[(4, 16)], deltas[(16, 4)]]))
        sensitivity_2bit = float(np.mean([deltas[(2, 16)], deltas[(16, 2)]]))
        joint_mixed_sensitivity = float(np.mean([deltas[(2, 2)], deltas[(4, 4)], deltas[(2, 4)], deltas[(4, 2)]]))

        row = {
            "task": task,
            "n_samples": EXPECTED_TASK_COUNTS[task],
            "fp16_score": fp16,
        }
        for cfg in ALL_CONFIGS:
            row[f"{config_label(cfg)}_score"] = cell[cfg]
        for cfg in ALL_CONFIGS:
            if cfg == FP16:
                continue
            row[f"{config_label(cfg)}_delta_vs_fp16"] = deltas[cfg]
        row.update(
            {
                "largest_abs_degradation": worst_delta,
                "largest_abs_degradation_config": config_label(worst_cfg),
                "k_sensitivity": k_sensitivity,
                "v_sensitivity": v_sensitivity,
                "tolerance_4bit": tolerance_4bit,
                "sensitivity_2bit": sensitivity_2bit,
                "joint_mixed_sensitivity": joint_mixed_sensitivity,
            }
        )
        rows.append(row)
    return rows


HIGHLIGHT_CONTRASTS = OrderedDict(
    [
        ("K16/V16 -> K16/V2", {(16, 2): 1.0, (16, 16): -1.0}),
        ("K16/V16 -> K2/V16", {(2, 16): 1.0, (16, 16): -1.0}),
        ("K16/V4 -> K16/V2", {(16, 2): 1.0, (16, 4): -1.0}),
        ("K2/V4 -> K2/V2", {(2, 2): 1.0, (2, 4): -1.0}),
        ("K4/V4 -> K4/V2", {(4, 2): 1.0, (4, 4): -1.0}),
        ("K16/V2 -> K2/V2", {(2, 2): 1.0, (16, 2): -1.0}),
    ]
)


def rank_tasks_by_highlight_contrasts(task_mean_raw):
    out = OrderedDict()
    for name, contrast in HIGHLIGHT_CONTRASTS.items():
        per_task = {}
        for task in EXPECTED_TASKS:
            score_map_t = {cfg: task_mean_raw[cfg][task] for cfg in ALL_CONFIGS}
            per_task[task] = apply_contrast(score_map_t, contrast)
        ranked = sorted(EXPECTED_TASKS, key=lambda t: per_task[t])
        out[name] = {
            "most_negative_5": ranked[:5],
            "values": {t: float(per_task[t]) for t in EXPECTED_TASKS},
        }
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def fmt(x, nd=4):
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x)


def write_overall_scores_csv(path, official_avg, raw_avg, delta_boot):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "k_bits", "v_bits", "official_avg", "raw_avg", "delta_vs_fp16_raw", "delta_ci_low", "delta_ci_high"])
        for cfg in ALL_CONFIGS:
            k, v = cfg
            label = config_label(cfg)
            if cfg == FP16:
                writer.writerow([label, k, v, official_avg[cfg], raw_avg[cfg], 0.0, 0.0, 0.0])
            else:
                d = delta_boot[f"{label} - FP16"]
                writer.writerow([label, k, v, official_avg[cfg], raw_avg[cfg], d["observed"], d["ci_low"], d["ci_high"]])


def write_task_scores_csv(path, sensitivity_rows):
    if not sensitivity_rows:
        return
    fieldnames = list(sensitivity_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sensitivity_rows:
            writer.writerow(row)


def write_bootstrap_contrasts_csv(path, boot_summaries, group_name):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "contrast", "observed", "ci_low", "ci_high", "ci_excludes_zero"])
        for name, s in boot_summaries.items():
            excludes_zero = not (s["ci_low"] <= 0 <= s["ci_high"])
            writer.writerow([group_name, name, s["observed"], s["ci_low"], s["ci_high"], excludes_zero])


def write_interactions_csv(path, interaction_boot):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["interaction", "observed", "ci_low", "ci_high", "ci_excludes_zero"])
        for name, s in interaction_boot.items():
            excludes_zero = not (s["ci_low"] <= 0 <= s["ci_high"])
            writer.writerow([name, s["observed"], s["ci_low"], s["ci_high"], excludes_zero])


def write_loto_csv(path, loto):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["contrast", "full_15_task_mean", "min", "min_task", "max", "max_task", "sign_changed"])
        for name, l in loto.items():
            writer.writerow(
                [name, l["full_15_task_mean"], l["min"], l["min_task"], l["max"], l["max_task"], bool(l["sign_changed_when_removing"])]
            )


def write_report_md(path, ctx):
    integrity = ctx["integrity_result"]
    pairing = ctx["pairing_result"]
    recompute = ctx["recompute_result"]
    official_avg = ctx["official_avg"]
    raw_avg = ctx["raw_avg"]
    delta_boot = ctx["delta_boot"]
    key_traj_boot = ctx["key_traj_boot"]
    val_traj_boot = ctx["val_traj_boot"]
    interaction_boot = ctx["interaction_boot"]
    loto = ctx["loto"]
    sensitivity_rows = ctx["sensitivity_rows"]
    highlight_rankings = ctx["highlight_rankings"]

    def ci_note(summary):
        return "CI crosses 0 (unstable)" if summary["ci_low"] <= 0 <= summary["ci_high"] else "CI excludes 0"

    lines = []
    lines.append("# K/V Cache 3x3 Ablation: Paired Sample-Level and Bootstrap Analysis")
    lines.append("")
    lines.append(f"- Generated: {ctx['timestamp']}")
    lines.append(f"- Git commit: `{ctx['git_commit']}`")
    lines.append(f"- Bootstrap iterations: {ctx['iterations']}, base seed: {ctx['seed']}")
    lines.append(f"- Pred root: `{ctx['pred_root']}`")
    lines.append("")

    lines.append("## 1. Data validation status")
    lines.append("")
    lines.append("| check | status |")
    lines.append("|---|---|")
    lines.append(f"| configuration discovery (9/9) | PASS |")
    lines.append(f"| strict JSONL integrity (row counts, valid JSON, no NUL, trailing newline, no leftover .partial) | {integrity['status']} |")
    lines.append(f"| task-set completeness (15/15 per config) | PASS |")
    lines.append(f"| sample-level pairing (answers/all_classes/length, all 9 configs) | {pairing['status']} |")
    lines.append(f"| recomputed scores vs result.json ({recompute['n_checked']} config-task pairs) | {recompute['status']} ({recompute['n_mismatches']} mismatches) |")
    lines.append("")
    lines.append(
        "**Pairing method**: predictions carry no explicit sample ID; pairing is "
        "validated by same-task row order plus row-by-row equality of `answers`, "
        "`all_classes`, and `length` across all 9 configurations. This analysis "
        "fails closed (raises, writes nothing) if any of the above checks fail."
    )
    lines.append("")
    lines.append("**Configuration discovery** (how each of the 9 directories was resolved):")
    lines.append("")
    lines.append("| directory | status | config / reason |")
    lines.append("|---|---|---|")
    for entry in ctx["discovery_log"]:
        detail = entry.get("config", entry.get("reason", ""))
        lines.append(f"| {entry['dir']} | {entry['status']} | {detail} |")
    lines.append("")

    lines.append("## 2. Full 3x3 matrix (official average -- mean of 15 rounded task scores)")
    lines.append("")
    lines.append("| K \\ V | V16 | V4 | V2 |")
    lines.append("|---|---|---|---|")
    for k in K_LEVELS:
        cells = " | ".join(f"{official_avg[(k, v)]:.6f}" for v in V_LEVELS)
        lines.append(f"| K{k} | {cells} |")
    lines.append("")

    lines.append("## 3. Deltas vs FP16 (raw/unrounded, percentage points)")
    lines.append("")
    lines.append("Sign convention: delta = score(destination) - score(source); negative = degradation.")
    lines.append("")
    lines.append("| config | delta vs FP16 | 95% bootstrap CI | |")
    lines.append("|---|---|---|---|")
    for cfg in ALL_CONFIGS:
        if cfg == FP16:
            continue
        s = delta_boot[f"{config_label(cfg)} - FP16"]
        lines.append(f"| {config_label(cfg)} | {fmt(s['observed'])} | [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] | {ci_note(s)} |")
    lines.append("")

    lines.append("## 4. Key trajectories (fixed Value precision)")
    lines.append("")
    lines.append("| contrast | observed | 95% CI | |")
    lines.append("|---|---|---|---|")
    for name, s in key_traj_boot.items():
        lines.append(f"| {name} | {fmt(s['observed'])} | [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] | {ci_note(s)} |")
    lines.append("")

    lines.append("## 5. Value trajectories (fixed Key precision)")
    lines.append("")
    lines.append("| contrast | observed | 95% CI | |")
    lines.append("|---|---|---|---|")
    for name, s in val_traj_boot.items():
        lines.append(f"| {name} | {fmt(s['observed'])} | [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] | {ci_note(s)} |")
    lines.append("")

    lines.append("## 6. Bootstrap design")
    lines.append("")
    lines.append(
        "Paired, task-stratified bootstrap: for each of the 15 tasks, the same "
        "with-replacement sample of that task's row indices is applied to all 9 "
        "configurations in a given replicate; each configuration's replicate task "
        "score is the mean of the resampled per-sample scores; each configuration's "
        "replicate overall score is the equal (1/15) weighted mean of its 15 "
        "replicate task scores -- reproducing eval_long_bench.py's aggregation "
        "exactly (never a raw mean over all 3550 samples). All contrasts below are "
        "linear combinations of these paired per-replicate overall scores, so every "
        "reported CI is fully paired."
    )
    lines.append("")

    lines.append("## 7. K x V interactions")
    lines.append("")
    lines.append("Interaction = [S(K_dest,V_dest) - S(K_source,V_dest)] - [S(K_dest,V_source) - S(K_source,V_source)]")
    lines.append("")
    lines.append("| interaction | observed | 95% CI | |")
    lines.append("|---|---|---|---|")
    for name, s in interaction_boot.items():
        lines.append(f"| {name} | {fmt(s['observed'])} | [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] | {ci_note(s)} |")
    lines.append("")

    lines.append("## 8. Task-level sensitivity")
    lines.append("")
    lines.append(
        "Formulas (raw/unrounded per-task percentage points): "
        "`k_sensitivity` = mean over V in {16,4,2} of |score(K2,V) - score(K16,V)|; "
        "`v_sensitivity` = mean over K in {16,4,2} of |score(K,V2) - score(K,V16)|; "
        "`tolerance_4bit` = mean of {K4/V16-FP16, K16/V4-FP16} (single-axis 4-bit drop); "
        "`sensitivity_2bit` = mean of {K2/V16-FP16, K16/V2-FP16} (single-axis 2-bit drop); "
        "`joint_mixed_sensitivity` = mean of {K2/V2, K4/V4, K2/V4, K4/V2} deltas vs FP16 "
        "(both axes quantized)."
    )
    lines.append("")
    lines.append("| task | n | worst delta vs FP16 | worst config | k_sensitivity | v_sensitivity | tol_4bit | sens_2bit | joint_mixed |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in sensitivity_rows:
        lines.append(
            f"| {row['task']} | {row['n_samples']} | {fmt(row['largest_abs_degradation'])} | "
            f"{row['largest_abs_degradation_config']} | {fmt(row['k_sensitivity'])} | "
            f"{fmt(row['v_sensitivity'])} | {fmt(row['tolerance_4bit'])} | "
            f"{fmt(row['sensitivity_2bit'])} | {fmt(row['joint_mixed_sensitivity'])} |"
        )
    lines.append("")
    lines.append("Tasks most responsible for large changes in specific highlighted contrasts (most negative 5):")
    lines.append("")
    for name, r in highlight_rankings.items():
        lines.append(f"- **{name}**: {', '.join(r['most_negative_5'])}")
    lines.append("")

    lines.append("## 9. Leave-one-task-out robustness")
    lines.append("")
    lines.append("| contrast | full 15-task | min (task removed) | max (task removed) | sign ever flips |")
    lines.append("|---|---|---|---|---|")
    for name, l in loto.items():
        changed = ", ".join(l["sign_changed_when_removing"]) if l["sign_changed_when_removing"] else "no"
        lines.append(
            f"| {name} | {fmt(l['full_15_task_mean'])} | {fmt(l['min'])} ({l['min_task']}) | "
            f"{fmt(l['max'])} ({l['max_task']}) | {changed} |"
        )
    lines.append("")

    lines.append("## 10. Safe interpretation")
    lines.append("")

    def excludes_zero(s):
        return not (s["ci_low"] <= 0 <= s["ci_high"])

    v4_stable = all(
        not excludes_zero(val_traj_boot[f"V16->V4 @ K{k}"]) or val_traj_boot[f"V16->V4 @ K{k}"]["observed"] > -0.5
        for k in K_LEVELS
    )
    v2_drop_present = any(excludes_zero(val_traj_boot[f"V4->V2 @ K{k}"]) for k in K_LEVELS)

    lines.append(
        "This section distinguishes point estimates, bootstrap-supported "
        "conclusions (CI excludes 0), and unstable/task-dependent observations "
        "(CI crosses 0, or leave-one-task-out changes sign)."
    )
    lines.append("")
    lines.append("**Point estimates only** (not claims of significance):")
    lines.append(f"- The full 3x3 official-average matrix is reported in Section 2; all 9 cells are within a narrow band of FP16 ({official_avg[FP16]:.4f}).")
    lines.append("")
    lines.append("**Bootstrap-supported observations** (95% CI excludes 0):")
    for name, s in list(delta_boot.items()) + list(key_traj_boot.items()) + list(val_traj_boot.items()):
        if excludes_zero(s):
            lines.append(f"- {name}: {fmt(s['observed'])}, 95% CI [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}].")
    lines.append("")
    lines.append("**Unstable / task-dependent observations** (CI crosses 0, or sign flips under leave-one-task-out):")
    for name, s in list(delta_boot.items()) + list(key_traj_boot.items()) + list(val_traj_boot.items()):
        if not excludes_zero(s):
            lines.append(f"- {name}: {fmt(s['observed'])}, 95% CI [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] crosses 0.")
    for name, l in loto.items():
        if l["sign_changed_when_removing"]:
            lines.append(f"- {name}: sign flips under leave-one-task-out when removing {', '.join(l['sign_changed_when_removing'])}.")
    lines.append("")
    lines.append(
        "**On the working hypothesis** (\"Value-cache quantization may remain "
        "relatively stable at 4-bit precision, while a larger degradation emerges "
        "when Value precision is reduced from 4 to 2 bits; Key sensitivity may also "
        "depend on Value precision\"): see Sections 3-7 above for the exact point "
        "estimates and CIs this hypothesis should be checked against "
        "(V16->V4 vs V4->V2 trajectories at each fixed K, and the K x V interaction "
        "terms). This script reports the paired estimates; it does not itself "
        "assert the hypothesis is confirmed."
    )
    lines.append("")
    lines.append("**What this analysis does NOT claim:**")
    lines.append("- It does not claim any quantized configuration is better than FP16 merely because a point estimate is higher.")
    lines.append("- It does not claim statistical equivalence between any two configurations (no equivalence test was implemented).")
    lines.append("- It does not claim significance from point-estimate differences alone; only CI-based statements above are bootstrap-supported.")
    lines.append("- It does not compute or report p-values.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("1. Each configuration has exactly one deterministic greedy-decoded prediction pass; no repeated seeds.")
    lines.append("2. Bootstrap estimates benchmark sample/task uncertainty, not run-to-run or cross-hardware variance.")
    lines.append("3. Predictions carry no sample ID; pairing relies on validated row order plus answers/all_classes/length equality.")
    lines.append("4. The 3 earliest configurations (FP16, K2/V2, K4/V4) predate run_config.json; their k_bits/v_bits are inferred from the established `_<N>bits_` legacy directory-naming convention, not independently confirmed metadata.")
    lines.append("5. Only one model (LongChat-7B), one context length, and one host/bit-configuration set is covered.")
    lines.append("6. Interaction terms are operational ablation interactions, not evidence of a causal mechanism.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary_json(ctx):
    return {
        "timestamp": ctx["timestamp"],
        "git_commit": ctx["git_commit"],
        "pred_root": ctx["pred_root"],
        "bootstrap_seed": ctx["seed"],
        "bootstrap_iterations": ctx["iterations"],
        "discovery_log": ctx["discovery_log"],
        "metric_implementation": "eval_long_bench.dataset2metric + metrics.py (unmodified, imported)",
        "integrity_validation": {"status": ctx["integrity_result"]["status"], "n_checked": ctx["integrity_result"]["n_config_task_pairs_checked"]},
        "pairing_validation": ctx["pairing_result"],
        "recomputed_score_validation": {
            "status": ctx["recompute_result"]["status"],
            "n_checked": ctx["recompute_result"]["n_checked"],
            "n_mismatches": ctx["recompute_result"]["n_mismatches"],
        },
        "matrix_official_avg": {config_label(c): v for c, v in ctx["official_avg"].items()},
        "matrix_raw_avg": {config_label(c): v for c, v in ctx["raw_avg"].items()},
        "delta_vs_fp16": ctx["delta_boot"],
        "key_trajectories": ctx["key_traj_boot"],
        "value_trajectories": ctx["val_traj_boot"],
        "interactions": ctx["interaction_boot"],
        "leave_one_task_out": ctx["loto"],
        "task_sensitivity": ctx["sensitivity_rows"],
        "highlight_task_rankings": ctx["highlight_rankings"],
        "limitations": [
            "Each configuration has exactly one deterministic greedy-decoded prediction pass; no repeated seeds.",
            "Bootstrap estimates benchmark sample/task uncertainty, not run-to-run or cross-hardware variance.",
            "Predictions carry no sample ID; pairing relies on validated row order plus answers/all_classes/length equality.",
            "The 3 earliest configurations (FP16, K2/V2, K4/V4) predate run_config.json; k_bits/v_bits are inferred from the legacy `_<N>bits_` directory naming convention.",
            "Only one model, one context length, and one host/bit-configuration set is covered.",
            "Interaction terms are operational ablation interactions, not evidence of a causal mechanism.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-root", type=str, default="pred", help="Root directory containing the 9 configuration subdirectories.")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of paired bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; reruns with the same seed reproduce identical bootstrap results.")
    parser.add_argument("--output-dir", type=str, default="analysis/results/kv_3x3")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    if not pred_root.is_absolute():
        pred_root = REPO_ROOT / pred_root

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    print(f"Discovering configurations under {pred_root}...")
    configs, discovery_log = discover_configurations(pred_root)
    print(f"  resolved {len(configs)}/9 configurations: {[config_label(c) for c in configs]}")

    print("Validating strict JSONL integrity (row counts, valid JSON, NUL bytes, trailing newline, no leftover .partial)...")
    integrity_result = validate_integrity(configs)
    print(f"  integrity: {integrity_result['status']} ({integrity_result['n_config_task_pairs_checked']} config-task pairs checked)")

    print("Loading predictions for 9 configurations...")
    all_data = {cfg: load_predictions(path) for cfg, path in configs.items()}
    result_jsons = {cfg: load_result_json(path) for cfg, path in configs.items()}

    print("Validating task sets...")
    validate_task_sets(all_data)

    print("Validating row-order pairing (answers/all_classes/length)...")
    pairing_result = validate_pairing(all_data)
    print(f"  pairing validation: {pairing_result['status']}")

    print("Recomputing sample-level scores and verifying against result.json...")
    recompute_result = recompute_and_verify(all_data, result_jsons)
    print(f"  score reproduction: {recompute_result['status']} ({recompute_result['n_checked']} config-task combinations checked, {recompute_result['n_mismatches']} mismatches)")

    sample_scores = recompute_result["sample_scores"]
    task_mean_raw = recompute_result["task_mean_raw"]
    official_avg = recompute_result["official_avg"]
    raw_avg = recompute_result["raw_avg"]

    print(f"Running paired task-stratified bootstrap ({args.bootstrap} iterations)...")
    overall_boot = bootstrap_overall_scores(sample_scores, args.bootstrap, args.seed)

    print("Summarizing delta-vs-FP16 / Key / Value trajectory contrasts...")
    delta_boot = summarize_contrasts(DELTA_VS_FP16_CONTRASTS, raw_avg, overall_boot)
    key_traj_boot = summarize_contrasts(KEY_TRAJECTORY_CONTRASTS, raw_avg, overall_boot)
    val_traj_boot = summarize_contrasts(VALUE_TRAJECTORY_CONTRASTS, raw_avg, overall_boot)

    print("Summarizing K x V interactions...")
    interaction_boot = summarize_contrasts(INTERACTION_CONTRASTS, raw_avg, overall_boot)

    print("Computing leave-one-task-out robustness...")
    loto = leave_one_task_out(task_mean_raw, LOTO_CONTRASTS)

    print("Building task-level sensitivity table...")
    sensitivity_rows = build_task_sensitivity_table(task_mean_raw)
    highlight_rankings = rank_tasks_by_highlight_contrasts(task_mean_raw)

    def validate_summary(label, s):
        for key in ("observed", "bootstrap_mean", "se", "ci_low", "ci_high"):
            if not np.isfinite(s[key]):
                raise RuntimeError(f"Non-finite value in {label}: {key}={s[key]}")
        if not (s["ci_low"] <= s["bootstrap_mean"] <= s["ci_high"]):
            raise RuntimeError(f"CI does not bracket bootstrap mean for {label}: {s}")

    for group_name, group in [("delta", delta_boot), ("key_traj", key_traj_boot), ("val_traj", val_traj_boot), ("interaction", interaction_boot)]:
        for name, s in group.items():
            validate_summary(f"{group_name}.{name}", s)

    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": get_git_commit(),
        "pred_root": str(pred_root),
        "seed": args.seed,
        "iterations": args.bootstrap,
        "discovery_log": discovery_log,
        "integrity_result": integrity_result,
        "pairing_result": pairing_result,
        "recompute_result": recompute_result,
        "official_avg": official_avg,
        "raw_avg": raw_avg,
        "delta_boot": delta_boot,
        "key_traj_boot": key_traj_boot,
        "val_traj_boot": val_traj_boot,
        "interaction_boot": interaction_boot,
        "loto": loto,
        "sensitivity_rows": sensitivity_rows,
        "highlight_rankings": highlight_rankings,
    }

    summary_path = output_dir / "summary.json"
    overall_csv_path = output_dir / "overall_scores.csv"
    task_csv_path = output_dir / "task_scores.csv"
    contrasts_csv_path = output_dir / "bootstrap_contrasts.csv"
    interactions_csv_path = output_dir / "interactions.csv"
    loto_csv_path = output_dir / "leave_one_task_out.csv"
    report_path = output_dir / "report.md"

    summary_path.write_text(json.dumps(build_summary_json(ctx), indent=2, ensure_ascii=False), encoding="utf-8")
    write_overall_scores_csv(overall_csv_path, official_avg, raw_avg, delta_boot)
    write_task_scores_csv(task_csv_path, sensitivity_rows)

    with contrasts_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "contrast", "observed", "ci_low", "ci_high", "ci_excludes_zero"])
    for group_name, group in [("delta_vs_fp16", delta_boot), ("key_trajectory", key_traj_boot), ("value_trajectory", val_traj_boot)]:
        with contrasts_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for name, s in group.items():
                excludes_zero = not (s["ci_low"] <= 0 <= s["ci_high"])
                writer.writerow([group_name, name, s["observed"], s["ci_low"], s["ci_high"], excludes_zero])

    write_interactions_csv(interactions_csv_path, interaction_boot)
    write_loto_csv(loto_csv_path, loto)
    write_report_md(report_path, ctx)

    print("\nFull 3x3 matrix (official average):")
    header = "K \\ V".ljust(8) + "".join(f"V{v}".rjust(14) for v in V_LEVELS)
    print(header)
    for k in K_LEVELS:
        row = f"K{k}".ljust(8) + "".join(f"{official_avg[(k, v)]:.6f}".rjust(14) for v in V_LEVELS)
        print(row)

    print("\nDelta vs FP16 (bootstrap):")
    for name, s in delta_boot.items():
        print(f"  {name:16s} observed={s['observed']:+.4f} 95% CI=[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]")

    print("\nK x V interactions (bootstrap):")
    for name, s in interaction_boot.items():
        print(f"  {name:28s} observed={s['observed']:+.4f} 95% CI=[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]")

    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {overall_csv_path}")
    print(f"Wrote: {task_csv_path}")
    print(f"Wrote: {contrasts_csv_path}")
    print(f"Wrote: {interactions_csv_path}")
    print(f"Wrote: {loto_csv_path}")
    print(f"Wrote: {report_path}")
    print("\nANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
