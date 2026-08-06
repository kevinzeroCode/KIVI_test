"""Paired sample-level and bootstrap analysis of the five completed KV-cache
ablation runs (FP16, K2/V16, K16/V2, K2/V2, K4/V4) on LongChat-7B / LongBench.

CPU-only, read-only with respect to prediction data: this script never loads
a model, never touches the GPU, and never re-runs generation. It re-derives
per-sample scores from the existing prediction JSONLs using the exact same
scoring functions as eval_long_bench.py / metrics.py (imported, not
reimplemented), verifies those recomputed scores reproduce the committed
result.json task scores, and then runs paired bootstrap resampling over the
already-fixed set of samples/tasks to characterize benchmark-level
uncertainty (NOT run-to-run or hardware variance -- see the "limitations"
section this script writes into its own report).

Usage:
    python analysis/analyze_kv_ablation.py \
        --bootstrap-iterations 10000 --seed 42 --output-dir analysis/results
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_long_bench import dataset2metric  # noqa: E402

METHODS = OrderedDict(
    [
        ("FP16", "pred/longchat-7b-v1.5-32k_31500_16bits_group32_residual128"),
        ("K2/V16", "pred/longchat-7b-v1.5-32k_31500_k2_v16_group32_residual128"),
        ("K16/V2", "pred/longchat-7b-v1.5-32k_31500_k16_v2_group32_residual128"),
        ("K2/V2", "pred/longchat-7b-v1.5-32k_31500_2bits_group32_residual128"),
        ("K4/V4", "pred/longchat-7b-v1.5-32k_31500_4bits_group32_residual128"),
    ]
)

EXPECTED_TASKS = sorted(
    [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "musique",
        "2wikimqa", "gov_report", "qmsum", "multi_news", "triviaqa",
        "samsum", "trec", "passage_retrieval_en", "lcc", "repobench-p",
    ]
)

FIRST_LINE_ONLY_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}

# EFFECT definitions operate on percentage-point sample scores (score * 100),
# matching eval_long_bench.py's `100 * total_score / len(predictions)` scale.
EFFECT_NAMES = [
    "key_only",       # K2/V16 - FP16
    "value_only",     # K16/V2 - FP16
    "joint",          # K2/V2  - FP16
    "kivi4",          # K4/V4  - FP16
    "key_minus_value",  # K2/V16 - K16/V2
    "interaction",    # K2/V2 - K2/V16 - K16/V2 + FP16
]

EFFECT_LABELS = {
    "key_only": "Key-only effect (K2/V16 - FP16)",
    "value_only": "Value-only effect (K16/V2 - FP16)",
    "joint": "Joint effect (K2/V2 - FP16)",
    "kivi4": "KIVI-4 effect (K4/V4 - FP16)",
    "key_minus_value": "Key-only minus Value-only (K2/V16 - K16/V2)",
    "interaction": "Interaction (K2/V2 - K2/V16 - K16/V2 + FP16)",
}

# Effects that represent "quantized method vs FP16" and therefore support a
# meaningful win/tie/loss breakdown at the sample level.
VS_FP16_EFFECTS = {"key_only", "value_only", "joint", "kivi4"}

HIGHLIGHT_TASKS = ["lcc", "repobench-p", "passage_retrieval_en", "2wikimqa", "multifieldqa_en"]


class PairingError(RuntimeError):
    pass


class ScoreReproductionError(RuntimeError):
    pass


def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()


def short_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_method_predictions(pred_dir):
    """Returns {task_name: [{"pred":..., "answers":..., "all_classes":..., "length":...}, ...]}."""
    pred_dir = REPO_ROOT / pred_dir
    data = {}
    for path in sorted(pred_dir.glob("*.jsonl")):
        task = path.stem
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
    path = REPO_ROOT / pred_dir / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_task_sets(all_data):
    for method, data in all_data.items():
        tasks = set(data.keys())
        if tasks != set(EXPECTED_TASKS):
            missing = set(EXPECTED_TASKS) - tasks
            extra = tasks - set(EXPECTED_TASKS)
            raise PairingError(
                f"{method}: task set mismatch. missing={sorted(missing)} extra={sorted(extra)}"
            )


def validate_pairing(all_data):
    """Row-order pairing validation: for each task and row index, answers,
    all_classes, and length must match exactly across all five methods.
    Fails closed (raises) on the first mismatch found, reporting only a
    short hash of the differing field's value, never the raw content."""
    methods = list(all_data.keys())
    reference_method = methods[0]

    for task in EXPECTED_TASKS:
        row_counts = {m: len(all_data[m][task]) for m in methods}
        if len(set(row_counts.values())) != 1:
            raise PairingError(f"task={task}: row count mismatch across methods: {row_counts}")

        n_rows = row_counts[reference_method]
        for idx in range(n_rows):
            ref_row = all_data[reference_method][task][idx]
            for method in methods[1:]:
                row = all_data[method][task][idx]
                for field in ("answers", "all_classes", "length"):
                    if row[field] != ref_row[field]:
                        raise PairingError(
                            f"pairing mismatch: method={method} task={task} row_index={idx} "
                            f"field={field} reference_method={reference_method} "
                            f"reference_hash={short_hash(ref_row[field])} "
                            f"observed_hash={short_hash(row[field])}"
                        )
    return {
        "status": "PASS",
        "methods_compared": methods,
        "reference_method": reference_method,
        "tasks_checked": EXPECTED_TASKS,
        "fields_checked": ["answers", "all_classes", "length"],
        "note": (
            "因 prediction 不包含原始 sample ID，本分析以相同 task 內的 row order 配對，"
            "並以 answers、all_classes、length 的逐列一致性作為配對驗證。"
        ),
    }


def sample_score(dataset, prediction, ground_truths, all_classes):
    """Exact re-derivation of a single sample's score, matching
    eval_long_bench.py's `scorer()` inner loop (pre-*100, pre-round,
    in [0, 1])."""
    if dataset in FIRST_LINE_ONLY_DATASETS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
    return score


def compute_task_sample_scores(task, rows):
    """Returns a numpy array of per-row scores in [0, 100], using the same
    all_classes value eval_long_bench.py's scorer() effectively uses for the
    whole task (the *last* row's all_classes, since the reference
    implementation overwrites a single `all_classes` variable while
    iterating the file and only uses its final value)."""
    all_classes = rows[-1]["all_classes"]
    scores = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        scores[i] = 100.0 * sample_score(task, row["pred"], row["answers"], all_classes)
    return scores


def recompute_and_verify(all_data, result_jsons):
    """Recomputes per-sample and per-task scores for all methods/tasks,
    verifies round(recomputed_task_mean, 2) == result.json's stored score
    for all method-task combinations, and returns the sample-score matrices
    plus both flavors of the 15-task average."""
    sample_scores = {method: {} for method in all_data}
    mismatches = []
    official_avg = {}
    raw_avg = {}

    for method, data in all_data.items():
        task_recomputed_rounded = {}
        task_recomputed_raw = {}
        for task in EXPECTED_TASKS:
            rows = data[task]
            scores = compute_task_sample_scores(task, rows)
            sample_scores[method][task] = scores
            raw_mean = float(scores.mean())
            rounded_mean = round(raw_mean, 2)
            task_recomputed_rounded[task] = rounded_mean
            task_recomputed_raw[task] = raw_mean

            stored = result_jsons[method][task]
            if rounded_mean != stored:
                mismatches.append(
                    {
                        "method": method,
                        "task": task,
                        "recomputed": rounded_mean,
                        "stored_result_json": stored,
                    }
                )

        official_avg[method] = sum(result_jsons[method][t] for t in EXPECTED_TASKS) / len(EXPECTED_TASKS)
        raw_avg[method] = sum(task_recomputed_raw[t] for t in EXPECTED_TASKS) / len(EXPECTED_TASKS)

    status = "PASS" if not mismatches else "FAIL"
    if mismatches:
        raise ScoreReproductionError(
            f"{len(mismatches)} method-task score mismatches: {mismatches[:5]}"
        )

    return {
        "sample_scores": sample_scores,
        "status": status,
        "n_checked": len(all_data) * len(EXPECTED_TASKS),
        "n_mismatches": len(mismatches),
        "official_avg": official_avg,
        "raw_avg": raw_avg,
    }


def compute_task_effect_matrix(sample_scores, task):
    """Returns an (n_rows, 6) array: columns in EFFECT_NAMES order, each a
    per-row (paired, same row index) effect in percentage points."""
    fp16 = sample_scores["FP16"][task]
    k2v16 = sample_scores["K2/V16"][task]
    k16v2 = sample_scores["K16/V2"][task]
    k2v2 = sample_scores["K2/V2"][task]
    k4v4 = sample_scores["K4/V4"][task]

    key_only = k2v16 - fp16
    value_only = k16v2 - fp16
    joint = k2v2 - fp16
    kivi4 = k4v4 - fp16
    key_minus_value = k2v16 - k16v2
    interaction = k2v2 - k2v16 - k16v2 + fp16

    return np.stack([key_only, value_only, joint, kivi4, key_minus_value, interaction], axis=1)


def derive_seed(base_seed, *parts):
    key = f"{base_seed}:" + ":".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def bootstrap_task_replicate_means(effect_matrix, iterations, seed):
    """effect_matrix: (n_rows, 6). Returns (iterations, 6): for each bootstrap
    iteration, the mean effect over a with-replacement resample of the same
    number of rows."""
    n_rows = effect_matrix.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=(iterations, n_rows))
    resampled = effect_matrix[idx]  # (iterations, n_rows, 6)
    return resampled.mean(axis=1)  # (iterations, 6)


def summarize_bootstrap(observed, boot_values):
    """boot_values: 1-D array of bootstrap replicate estimates for one effect."""
    mean = float(boot_values.mean())
    se = float(boot_values.std(ddof=1))
    ci_low, ci_high = (float(v) for v in np.percentile(boot_values, [2.5, 97.5]))
    p_gt0 = float((boot_values > 0).mean())
    p_lt0 = float((boot_values < 0).mean())
    sign_p = 2 * min(p_gt0, p_lt0, 1.0)
    sign_p = min(sign_p, 1.0)
    return {
        "observed": float(observed),
        "bootstrap_mean": mean,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_gt0": p_gt0,
        "p_lt0": p_lt0,
        "sign_probability": sign_p,
    }


def primary_bootstrap(sample_scores, iterations, seed):
    """Fixed-task stratified paired bootstrap: resample within each of the
    15 (fixed) tasks, take each task's bootstrap-replicate mean effect, then
    equally-weight-average across the 15 tasks for each replicate."""
    per_task_effect_matrix = {}
    overall_boot = np.zeros((iterations, len(EFFECT_NAMES)), dtype=np.float64)
    observed_task_means = {}

    for task in EXPECTED_TASKS:
        effect_matrix = compute_task_effect_matrix(sample_scores, task)
        per_task_effect_matrix[task] = effect_matrix
        observed_task_means[task] = effect_matrix.mean(axis=0)

        task_seed = derive_seed(seed, "primary", task)
        task_boot_means = bootstrap_task_replicate_means(effect_matrix, iterations, task_seed)
        overall_boot += task_boot_means / len(EXPECTED_TASKS)

    observed_overall = np.mean(
        np.stack([observed_task_means[t] for t in EXPECTED_TASKS], axis=0), axis=0
    )

    result = {}
    for i, name in enumerate(EFFECT_NAMES):
        result[name] = summarize_bootstrap(observed_overall[i], overall_boot[:, i])

    return result, per_task_effect_matrix, observed_task_means


def per_task_bootstrap(per_task_effect_matrix, iterations, seed):
    """Per-task paired bootstrap for each of the 6 effects, plus win/tie/loss
    rates (sample-level) for the four vs-FP16 effects."""
    out = {}
    for task in EXPECTED_TASKS:
        effect_matrix = per_task_effect_matrix[task]
        n_rows = effect_matrix.shape[0]
        task_out = {"n_samples": n_rows}
        for i, name in enumerate(EFFECT_NAMES):
            col = effect_matrix[:, i]
            task_seed = derive_seed(seed, "per_task", task, name)
            rng = np.random.default_rng(task_seed)
            idx = rng.integers(0, n_rows, size=(iterations, n_rows))
            boot_means = col[idx].mean(axis=1)
            summary = summarize_bootstrap(col.mean(), boot_means)

            if name in VS_FP16_EFFECTS:
                wins = int((col > 0).sum())
                ties = int((col == 0).sum())
                losses = int((col < 0).sum())
                summary.update(
                    {
                        "win_rate": wins / n_rows,
                        "tie_rate": ties / n_rows,
                        "loss_rate": losses / n_rows,
                    }
                )
            task_out[name] = summary
        out[task] = task_out
    return out


def secondary_task_level_bootstrap(observed_task_means, iterations, seed):
    """Treats the 15 observed task-level mean effects as a resamplable
    population: with-replacement resample of 15 tasks, 10000 iterations,
    mean + 95% CI. Sensitivity analysis only -- does not replace the
    primary (fixed-task) bootstrap."""
    tasks = EXPECTED_TASKS
    matrix = np.stack([observed_task_means[t] for t in tasks], axis=0)  # (15, 6)
    n_tasks = matrix.shape[0]

    result = {}
    for i, name in enumerate(EFFECT_NAMES):
        task_seed = derive_seed(seed, "secondary", name)
        rng = np.random.default_rng(task_seed)
        idx = rng.integers(0, n_tasks, size=(iterations, n_tasks))
        boot_means = matrix[idx, i].mean(axis=1)
        result[name] = summarize_bootstrap(matrix[:, i].mean(), boot_means)
    return result


def leave_one_task_out(observed_task_means):
    tasks = EXPECTED_TASKS
    matrix = np.stack([observed_task_means[t] for t in tasks], axis=0)  # (15, 6)
    full_mean = matrix.mean(axis=0)

    result = {}
    for i, name in enumerate(EFFECT_NAMES):
        loto_values = {}
        for j, left_out in enumerate(tasks):
            mask = np.ones(len(tasks), dtype=bool)
            mask[j] = False
            loto_values[left_out] = float(matrix[mask, i].mean())

        min_task = min(loto_values, key=loto_values.get)
        max_task = max(loto_values, key=loto_values.get)
        full_sign = np.sign(full_mean[i])
        sign_changes = [t for t, v in loto_values.items() if np.sign(v) != full_sign and full_sign != 0]

        result[name] = {
            "full_15_task_mean": float(full_mean[i]),
            "loto_values": loto_values,
            "min": loto_values[min_task],
            "min_task": min_task,
            "max": loto_values[max_task],
            "max_task": max_task,
            "sign_changed_when_removing": sign_changes,
        }
    return result


def sensitive_task_rankings(observed_task_means, per_task_boot):
    def sorted_by(effect_name, reverse=False):
        return sorted(
            EXPECTED_TASKS,
            key=lambda t: observed_task_means[t][EFFECT_NAMES.index(effect_name)],
            reverse=reverse,
        )

    def ci_entirely_below_zero(effect_name):
        return [
            t for t in EXPECTED_TASKS
            if per_task_boot[t][effect_name]["ci_high"] < 0
        ]

    key_idx = EFFECT_NAMES.index("key_only")
    value_idx = EFFECT_NAMES.index("value_only")

    most_negative_value_only = sorted_by("value_only")[:5]
    most_negative_key_only = sorted_by("key_only")[:5]
    most_negative_joint = sorted_by("joint")[:5]
    most_negative_interaction = sorted_by("interaction")[:5]

    diff_sorted = sorted(
        EXPECTED_TASKS,
        key=lambda t: abs(observed_task_means[t][key_idx] - observed_task_means[t][value_idx]),
        reverse=True,
    )[:5]

    return {
        "value_only_most_negative_5": most_negative_value_only,
        "key_only_most_negative_5": most_negative_key_only,
        "joint_most_negative_5": most_negative_joint,
        "interaction_most_negative_5": most_negative_interaction,
        "largest_key_vs_value_gap_5": diff_sorted,
        "value_only_ci_entirely_below_zero": ci_entirely_below_zero("value_only"),
        "key_only_ci_entirely_below_zero": ci_entirely_below_zero("key_only"),
        "interaction_ci_entirely_below_zero": ci_entirely_below_zero("interaction"),
        "highlighted_tasks": {
            t: {
                name: observed_task_means[t][EFFECT_NAMES.index(name)]
                for name in EFFECT_NAMES
            }
            for t in HIGHLIGHT_TASKS
        },
    }


def write_overall_csv(path, primary, secondary, loto):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "effect", "label", "observed_effect",
                "primary_bootstrap_mean", "primary_se", "primary_ci_low", "primary_ci_high",
                "primary_p_gt0", "primary_p_lt0", "primary_sign_probability",
                "secondary_bootstrap_mean", "secondary_ci_low", "secondary_ci_high",
                "loto_min", "loto_min_task", "loto_max", "loto_max_task",
                "loto_sign_changed",
            ]
        )
        for name in EFFECT_NAMES:
            p = primary[name]
            s = secondary[name]
            l = loto[name]
            writer.writerow(
                [
                    name, EFFECT_LABELS[name], p["observed"],
                    p["bootstrap_mean"], p["se"], p["ci_low"], p["ci_high"],
                    p["p_gt0"], p["p_lt0"], p["sign_probability"],
                    s["bootstrap_mean"], s["ci_low"], s["ci_high"],
                    l["min"], l["min_task"], l["max"], l["max_task"],
                    bool(l["sign_changed_when_removing"]),
                ]
            )


def write_per_task_csv(path, per_task_boot):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["task", "n_samples"]
        for name in EFFECT_NAMES:
            header += [f"{name}_effect", f"{name}_ci_low", f"{name}_ci_high"]
            if name in VS_FP16_EFFECTS:
                header += [f"{name}_win_rate", f"{name}_tie_rate", f"{name}_loss_rate"]
        writer.writerow(header)

        for task in EXPECTED_TASKS:
            row = [task, per_task_boot[task]["n_samples"]]
            for name in EFFECT_NAMES:
                d = per_task_boot[task][name]
                row += [d["observed"], d["ci_low"], d["ci_high"]]
                if name in VS_FP16_EFFECTS:
                    row += [d["win_rate"], d["tie_rate"], d["loss_rate"]]
            writer.writerow(row)


def fmt(x, nd=4):
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x)


def write_report_md(path, ctx):
    primary = ctx["primary"]
    secondary = ctx["secondary"]
    loto = ctx["loto"]
    rankings = ctx["rankings"]
    official_avg = ctx["official_avg"]

    def ci_note(name):
        p = primary[name]
        crosses = p["ci_low"] <= 0 <= p["ci_high"]
        return "CI 跨越 0（不穩定）" if crosses else "CI 未跨越 0"

    lines = []
    lines.append("# K/V Cache Ablation: Paired Sample-Level and Bootstrap Analysis")
    lines.append("")
    lines.append(f"- Generated: {ctx['timestamp']}")
    lines.append(f"- Git commit: `{ctx['git_commit']}`")
    lines.append(f"- Bootstrap iterations: {ctx['iterations']}, base seed: {ctx['seed']}")
    lines.append("")
    lines.append(
        "本分析以既有的五組完整 prediction JSONL 為輸入，"
        "重算 sample-level 分數並驗證與 `result.json` 一致，"
        "再以固定 15-task、task 內 paired bootstrap 為 primary 方法估計不確定性。"
    )
    lines.append("")
    lines.append(
        "**配對方式**：因 prediction 不包含原始 sample ID，本分析以相同 task 內的 "
        "row order 配對，並以 answers、all_classes、length 的逐列一致性作為配對驗證"
        f"（結果：{ctx['pairing_status']}）。"
    )
    lines.append("")

    lines.append("## A. 直接觀察（五組方法平均）")
    lines.append("")
    lines.append("| method | official average (result.json, rounded task scores) |")
    lines.append("|---|---|")
    for method in METHODS:
        lines.append(f"| {method} | {official_avg[method]:.7f} |")
    lines.append("")
    lines.append(
        f"- K2/V16 的平均（{official_avg['K2/V16']:.4f}）接近 FP16（{official_avg['FP16']:.4f}）。\n"
        f"- K16/V2 的平均（{official_avg['K16/V2']:.4f}）比 FP16 低約 "
        f"{official_avg['FP16'] - official_avg['K16/V2']:.4f}。\n"
        f"- K2/V2 的平均（{official_avg['K2/V2']:.4f}）比 FP16 低約 "
        f"{official_avg['FP16'] - official_avg['K2/V2']:.4f}。"
    )
    lines.append("")

    lines.append("## B. Overall paired effects — Primary (task-stratified) bootstrap")
    lines.append("")
    lines.append("| effect | observed | 95% CI | sign probability | CI vs 0 |")
    lines.append("|---|---|---|---|---|")
    for name in EFFECT_NAMES:
        p = primary[name]
        lines.append(
            f"| {EFFECT_LABELS[name]} | {fmt(p['observed'])} | "
            f"[{fmt(p['ci_low'])}, {fmt(p['ci_high'])}] | "
            f"{p['sign_probability']:.4f} | {ci_note(name)} |"
        )
    lines.append("")
    lines.append("Secondary (task-level resampling) bootstrap, for comparison only:")
    lines.append("")
    lines.append("| effect | observed | secondary 95% CI |")
    lines.append("|---|---|---|")
    for name in EFFECT_NAMES:
        s = secondary[name]
        lines.append(f"| {EFFECT_LABELS[name]} | {fmt(s['observed'])} | [{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] |")
    lines.append("")

    lines.append("## C. Leave-one-task-out")
    lines.append("")
    lines.append("| effect | full 15-task mean | min (task removed) | max (task removed) | sign changed by removing any one task |")
    lines.append("|---|---|---|---|---|")
    for name in EFFECT_NAMES:
        l = loto[name]
        changed = ", ".join(l["sign_changed_when_removing"]) if l["sign_changed_when_removing"] else "none"
        lines.append(
            f"| {EFFECT_LABELS[name]} | {fmt(l['full_15_task_mean'])} | "
            f"{fmt(l['min'])} ({l['min_task']}) | {fmt(l['max'])} ({l['max_task']}) | {changed} |"
        )
    lines.append("")
    value_only_loto = loto["value_only"]["loto_values"]
    lines.append(
        f"- 移除 `lcc` 後 Value-only effect：{fmt(value_only_loto['lcc'])}"
        f"（{'仍為負' if value_only_loto['lcc'] < 0 else '轉為非負'}）。"
    )
    pr_loto = {name: loto[name]["loto_values"]["passage_retrieval_en"] for name in EFFECT_NAMES}
    lines.append(
        "- 移除 `passage_retrieval_en` 後各 effect："
        + ", ".join(f"{EFFECT_LABELS[n]}={fmt(v)}" for n, v in pr_loto.items())
    )
    kmv_sign_stable = not loto["key_minus_value"]["sign_changed_when_removing"]
    lines.append(
        f"- Key-only 減 Value-only 的方向在 leave-one-task-out 下"
        f"{'保持穩定（未曾變號）' if kmv_sign_stable else '在移除某些 task 後變號，方向不穩定'}。"
    )
    interaction_dominated = bool(loto["interaction"]["sign_changed_when_removing"])
    lines.append(
        f"- Interaction 的符號{'會因移除單一 task 而改變，顯示可能由少數 task 主導' if interaction_dominated else '在移除任一單一 task 後皆未變號，不像是由單一 task 主導'}。"
    )
    lines.append("")

    lines.append("## D. 敏感任務排序")
    lines.append("")
    lines.append(f"- Value-only 最負面 5 個 task：{', '.join(rankings['value_only_most_negative_5'])}")
    lines.append(f"- Key-only 最負面 5 個 task：{', '.join(rankings['key_only_most_negative_5'])}")
    lines.append(f"- Joint 最負面 5 個 task：{', '.join(rankings['joint_most_negative_5'])}")
    lines.append(f"- Interaction 最負面 5 個 task：{', '.join(rankings['interaction_most_negative_5'])}")
    lines.append(f"- K2/V16 與 K16/V2 差異最大 5 個 task：{', '.join(rankings['largest_key_vs_value_gap_5'])}")
    lines.append(
        f"- Value-only 95% CI 完全低於 0 的 task："
        f"{', '.join(rankings['value_only_ci_entirely_below_zero']) or '（無）'}"
    )
    lines.append(
        f"- Key-only 95% CI 完全低於 0 的 task："
        f"{', '.join(rankings['key_only_ci_entirely_below_zero']) or '（無）'}"
    )
    lines.append(
        f"- Interaction 95% CI 完全低於 0 的 task："
        f"{', '.join(rankings['interaction_ci_entirely_below_zero']) or '（無）'}"
    )
    lines.append("")
    lines.append("特別關注任務（觀察值，不預設顯著）：")
    lines.append("")
    lines.append("| task | key_only | value_only | joint | kivi4 | key_minus_value | interaction |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in HIGHLIGHT_TASKS:
        v = rankings["highlighted_tasks"][t]
        lines.append(
            f"| {t} | " + " | ".join(fmt(v[n]) for n in EFFECT_NAMES) + " |"
        )
    lines.append("")

    lines.append("## E. Bootstrap 支持程度")
    lines.append("")
    lines.append(
        "- Key-only effect 的 primary CI "
        f"{'跨越 0' if primary['key_only']['ci_low'] <= 0 <= primary['key_only']['ci_high'] else '未跨越 0'}，"
        "與觀察到「幾乎無損」的方向一致。"
    )
    lines.append(
        "- Value-only effect 的 primary CI "
        f"{'跨越 0' if primary['value_only']['ci_low'] <= 0 <= primary['value_only']['ci_high'] else '未跨越 0'}。"
    )
    lines.append(
        "- Interaction 的 primary CI "
        f"{'跨越 0' if primary['interaction']['ci_low'] <= 0 <= primary['interaction']['ci_high'] else '未跨越 0，初步支持負向 interaction'}。"
    )
    lines.append(
        f"- Value-only 比 Key-only 敏感的方向在 leave-one-task-out 下"
        f"{'保持穩定' if not loto['key_minus_value']['sign_changed_when_removing'] else '不穩定，會因移除單一 task 而改變'}。"
    )
    lines.append("")

    lines.append("## F. 不可宣稱的內容（限制）")
    lines.append("")
    lines.append("1. 每組方法只有一批 deterministic greedy predictions（無多 seed 重複）。")
    lines.append(
        "2. Bootstrap 估計的是 benchmark samples/tasks 的不確定性，"
        "不是模型重新執行的 run-to-run variance，也不是跨硬體變異的估計。"
    )
    lines.append("3. Prediction 無 sample ID，配對依賴已驗證的 row order 一致性。")
    lines.append("4. LongBench 15-task 平均差距很小，不等於普遍模型品質差異。")
    lines.append("5. 本分析未涵蓋其他模型、context length、硬體或 bit configuration。")
    lines.append("6. 不能只依此證明 Value 量化在所有模型上都比 Key 量化敏感。")
    lines.append("7. Interaction 為 operational ablation interaction 的估計，不是因果機制證明。")
    lines.append("")

    lines.append("## G. 結論")
    lines.append("")
    lines.append(
        "在本次 LongChat-7B / LongBench 設定下，結果初步支持："
        "Value-only 量化（K16/V2）比 Key-only 量化（K2/V16）對整體平均分數的影響更大，"
        "且 joint（K2/V2）量化的降幅明顯大於兩個單側效果的簡單相加，"
        "初步支持存在負向 interaction。"
        "但整體平均層級的差距小於個別 task 的波動幅度（尤其 `lcc` 與 `passage_retrieval_en`），"
        "尚需跨模型與重複實驗驗證，才能將 task 層級的觀察（例如 code-completion 類任務對 Value "
        "量化較敏感）視為穩定結論。"
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary_json(ctx):
    return {
        "timestamp": ctx["timestamp"],
        "git_commit": ctx["git_commit"],
        "bootstrap_seed": ctx["seed"],
        "bootstrap_iterations": ctx["iterations"],
        "input_paths": {m: str(p) for m, p in METHODS.items()},
        "metric_implementation": "eval_long_bench.dataset2metric + metrics.py (unmodified, imported)",
        "pairing_validation": ctx["pairing_result"],
        "recomputed_score_validation": {
            "status": ctx["recompute_result"]["status"],
            "n_checked": ctx["recompute_result"]["n_checked"],
            "n_mismatches": ctx["recompute_result"]["n_mismatches"],
        },
        "method_averages": {
            "official_avg_from_result_json": ctx["official_avg"],
            "raw_avg_from_unrounded_sample_means": ctx["raw_avg"],
            "note": "正式 LongBench 報告分數使用 result.json 內 15 個已 round task score 的平均（official_avg_from_result_json）。",
        },
        "overall_effects_primary_bootstrap": ctx["primary"],
        "overall_effects_secondary_bootstrap": ctx["secondary"],
        "leave_one_task_out": ctx["loto"],
        "sensitive_task_rankings": ctx["rankings"],
        "limitations": [
            "Each method has exactly one deterministic greedy-decoded prediction pass; no repeated seeds.",
            "Bootstrap estimates benchmark sample/task uncertainty, not run-to-run or cross-hardware variance.",
            "Predictions carry no sample ID; pairing relies on validated row order plus answers/all_classes/length equality.",
            "The 15-task LongBench average differences are small and do not imply general model-quality differences.",
            "Only one model (LongChat-7B), one context length, and one host/bit-configuration set is covered.",
            "This analysis alone cannot establish that Value quantization is more sensitive than Key quantization across all models.",
            "The interaction term is an operational ablation interaction, not evidence of a causal mechanism.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="analysis/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions for 5 methods...")
    all_data = {method: load_method_predictions(path) for method, path in METHODS.items()}
    result_jsons = {method: load_result_json(path) for method, path in METHODS.items()}

    print("Validating task sets...")
    validate_task_sets(all_data)

    print("Validating row-order pairing (answers/all_classes/length)...")
    pairing_result = validate_pairing(all_data)
    print(f"  pairing validation: {pairing_result['status']}")

    print("Recomputing sample-level scores and verifying against result.json...")
    recompute_result = recompute_and_verify(all_data, result_jsons)
    print(
        f"  score reproduction: {recompute_result['status']} "
        f"({recompute_result['n_checked']} method-task combinations checked, "
        f"{recompute_result['n_mismatches']} mismatches)"
    )

    sample_scores = recompute_result["sample_scores"]

    print(f"Running primary paired bootstrap ({args.bootstrap_iterations} iterations)...")
    primary, per_task_effect_matrix, observed_task_means = primary_bootstrap(
        sample_scores, args.bootstrap_iterations, args.seed
    )

    print("Running per-task paired bootstrap...")
    per_task_boot = per_task_bootstrap(per_task_effect_matrix, args.bootstrap_iterations, args.seed)

    print("Running secondary task-level bootstrap...")
    secondary = secondary_task_level_bootstrap(observed_task_means, args.bootstrap_iterations, args.seed)

    print("Computing leave-one-task-out...")
    loto = leave_one_task_out(observed_task_means)

    print("Ranking sensitive tasks...")
    rankings = sensitive_task_rankings(observed_task_means, per_task_boot)

    def validate_summary(label, s):
        for key in ("observed", "bootstrap_mean", "se", "ci_low", "ci_high"):
            if not np.isfinite(s[key]):
                raise RuntimeError(f"Non-finite value in {label}: {key}={s[key]}")
        if not (s["ci_low"] <= s["bootstrap_mean"] <= s["ci_high"]):
            raise RuntimeError(f"CI does not bracket bootstrap mean for {label}: {s}")
        if not (s["ci_low"] <= s["observed"] <= s["ci_high"]):
            raise RuntimeError(f"CI does not bracket observed effect for {label}: {s}")

    for name in EFFECT_NAMES:
        validate_summary(f"primary.{name}", primary[name])
        validate_summary(f"secondary.{name}", secondary[name])
    for task in EXPECTED_TASKS:
        for name in EFFECT_NAMES:
            validate_summary(f"per_task.{task}.{name}", per_task_boot[task][name])

    ctx = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": get_git_commit(),
        "seed": args.seed,
        "iterations": args.bootstrap_iterations,
        "pairing_status": pairing_result["status"],
        "pairing_result": pairing_result,
        "recompute_result": recompute_result,
        "official_avg": recompute_result["official_avg"],
        "raw_avg": recompute_result["raw_avg"],
        "primary": primary,
        "secondary": secondary,
        "loto": loto,
        "rankings": rankings,
    }

    summary_path = output_dir / "kv_ablation_summary.json"
    overall_csv_path = output_dir / "kv_ablation_overall.csv"
    per_task_csv_path = output_dir / "kv_ablation_per_task.csv"
    report_path = output_dir / "kv_ablation_report.md"

    summary_path.write_text(json.dumps(build_summary_json(ctx), indent=2, ensure_ascii=False), encoding="utf-8")
    write_overall_csv(overall_csv_path, primary, secondary, loto)
    write_per_task_csv(per_task_csv_path, per_task_boot)
    write_report_md(report_path, ctx)

    print("\nOverall effects (primary bootstrap):")
    for name in EFFECT_NAMES:
        p = primary[name]
        print(
            f"  {name:18s} observed={p['observed']:+.4f} "
            f"95% CI=[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] "
            f"sign_p={p['sign_probability']:.4f}"
        )

    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {overall_csv_path}")
    print(f"Wrote: {per_task_csv_path}")
    print(f"Wrote: {report_path}")
    print("\nANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
