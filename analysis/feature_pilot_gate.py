"""Stage G3B: pre-registered scientific feature-pilot decision gate.

Implements EXACTLY the A/B/C/D gate defined in
docs/stage_g3b_pre_registration.md (hardened in the Stage-G3B-IMPL round).
This module contains ONLY the gate-evaluation combination logic -- it does
not collect, load, or compute any feature/sensitivity data itself. Every
function here operates on caller-supplied values (real or synthetic) so it
is fully testable on CPU without any real feature data existing yet.

Reuses the project's one Spearman implementation
(analysis.analyze_layer_sensitivity_pilot.spearman_corr) rather than
reimplementing correlation math -- consistent with this project's standing
"never reimplement metrics, always import" discipline.

CPU-only, no torch, no GPU.
"""
from analysis.analyze_layer_sensitivity_pilot import spearman_corr

PRIMARY_STAGE_E_TASKS = ("trec", "lcc", "passage_retrieval_en", "2wikimqa")
LOTO_REMOVED_TASK = "lcc"
LAYER0_DIAGNOSTIC_TASKS = ("lcc", "multifieldqa_en", "samsum")

# Pre-registered pragmatic magnitude threshold for Criterion B (Section 3 of
# the Stage-G3B-IMPL hardening round): NOT a universal or
# Spearman-specific statistical convention, just a specific number fixed
# now, before any feature data exists, so it cannot be adjusted after
# seeing results.
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.5


class GateInputError(ValueError):
    """Malformed gate input -- fails closed rather than guessing a missing
    task's correlation or silently ignoring an unexpected task key."""


def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def compute_task_specific_rho(x_by_task_layer, y_by_task_layer, tasks=PRIMARY_STAGE_E_TASKS):
    """Convenience wrapper for the future real analysis: x_by_task_layer /
    y_by_task_layer are dict[task] -> dict[layer_idx] -> value, covering the
    same set of layers per task. Returns dict[task] -> Spearman rho (n =
    number of shared layers), via the project's shared spearman_corr --
    never an independent reimplementation.
    """
    missing_x = set(tasks) - set(x_by_task_layer)
    missing_y = set(tasks) - set(y_by_task_layer)
    if missing_x or missing_y:
        raise GateInputError(f"missing task(s) in inputs: x missing {missing_x}, y missing {missing_y}")
    rho_by_task = {}
    for task in tasks:
        layers = sorted(set(x_by_task_layer[task]) & set(y_by_task_layer[task]))
        if len(layers) != len(set(x_by_task_layer[task])) or len(layers) != len(set(y_by_task_layer[task])):
            raise GateInputError(f"task {task!r}: x and y must cover the exact same layer set")
        x = [x_by_task_layer[task][l] for l in layers]
        y = [y_by_task_layer[task][l] for l in layers]
        rho_by_task[task] = spearman_corr(x, y)
    return rho_by_task


def compute_layer0_diagnostic_rho(relative_l2_by_task, sensitivity_by_task, tasks=LAYER0_DIAGNOSTIC_TASKS):
    """Criterion D's exact n=3 correlation: Layer-0-only relative_l2 vs
    SignedSensitivity across {lcc, multifieldqa_en, samsum}, using the same
    shared spearman_corr. relative_l2_by_task / sensitivity_by_task are
    dict[task] -> scalar (already Layer-0-specific -- this function does
    not select a layer itself).
    """
    missing_x = set(tasks) - set(relative_l2_by_task)
    missing_y = set(tasks) - set(sensitivity_by_task)
    if missing_x or missing_y:
        raise GateInputError(f"missing task(s) in inputs: relative_l2 missing {missing_x}, sensitivity missing {missing_y}")
    x = [relative_l2_by_task[t] for t in tasks]
    y = [sensitivity_by_task[t] for t in tasks]
    return spearman_corr(x, y)


def _validate_rho_by_task(raw_rho_by_task, tasks=PRIMARY_STAGE_E_TASKS):
    got = set(raw_rho_by_task)
    want = set(tasks)
    if got != want:
        raise GateInputError(f"raw_rho_by_task must have exactly keys {sorted(want)}, got {sorted(got)}")


def to_gate_rho(raw_rho):
    """Stage-G3B-IMPL undefined-Spearman convention: raw_rho is EITHER a
    real Spearman coefficient OR None ("undefined" -- produced by
    spearman_corr whenever either the feature vector or the sensitivity
    vector has zero variance across the layers/tasks being correlated; the
    known concrete case is trec's Value SignedSensitivity, which is
    constant across all 8 sampled layers).

    raw_rho=None must NEVER be reported as an observed rho=0 -- that would
    misrepresent "the correlation could not be computed" as "the
    correlation was computed and found to be exactly zero", a genuinely
    different (and false) scientific claim.

    For GATE EVALUATION ONLY, undefined maps to gate_rho=0.0: a
    deliberate, conservative convention meaning "an undefined correlation
    contributes NO evidence for the hypothesized negative association" --
    it can never itself satisfy a rho<0 direction check, and it pulls a
    magnitude median toward zero rather than being silently dropped from
    it. Applied identically to Criteria A, B, C, and D.
    """
    return 0.0 if raw_rho is None else raw_rho


def evaluate_criterion_a(raw_rho_by_task, tasks=PRIMARY_STAGE_E_TASKS):
    """A: gate_rho < 0 (relative_l2-vs-SignedSensitivity direction) in at
    least 3 of the 4 Stage-E tasks. An undefined raw_rho maps to
    gate_rho=0.0, which is never < 0, so it never counts toward the
    negative_count -- consistent with "undefined contributes no evidence".
    """
    _validate_rho_by_task(raw_rho_by_task, tasks)
    gate_rho_by_task = {t: to_gate_rho(raw_rho_by_task[t]) for t in tasks}
    negative_count = sum(1 for t in tasks if gate_rho_by_task[t] < 0)
    return {
        "raw_rho_by_task": dict(raw_rho_by_task),
        "gate_rho_by_task": gate_rho_by_task,
        "negative_count": negative_count,
        "required": 3,
        "pass": negative_count >= 3,
    }


def evaluate_criterion_b(raw_rho_by_task, threshold=DEFAULT_EFFECT_SIZE_THRESHOLD, tasks=PRIMARY_STAGE_E_TASKS):
    """B: median(abs(gate_rho)) over exactly the 4 Stage-E tasks >=
    threshold (a pre-registered pragmatic magnitude threshold, not a
    universal or Spearman-specific statistical convention). An undefined
    raw_rho contributes gate_rho=0.0 to the median -- it is NEVER dropped
    from the 4-element set (dropping it would silently shrink n and could
    inflate the median; substituting 0 is the conservative choice).
    """
    _validate_rho_by_task(raw_rho_by_task, tasks)
    gate_rho_by_task = {t: to_gate_rho(raw_rho_by_task[t]) for t in tasks}
    median_abs = _median([abs(gate_rho_by_task[t]) for t in tasks])
    return {
        "raw_rho_by_task": dict(raw_rho_by_task),
        "gate_rho_by_task": gate_rho_by_task,
        "median_abs_gate_rho": median_abs,
        "threshold": threshold,
        "pass": median_abs >= threshold,
    }


def evaluate_criterion_c(raw_rho_by_task, removed_task=LOTO_REMOVED_TASK, tasks=PRIMARY_STAGE_E_TASKS):
    """C (hardened, Stage-G3B-IMPL): after removing lcc, at least 2 of the
    remaining 3 tasks have gate_rho < 0, AND the median of those 3
    remaining gate_rho values (signed, not absolute) is < 0. Explicitly
    requires 2/3, not 3/3. An undefined remaining task (e.g. trec's Value
    SignedSensitivity) is NEVER dropped -- it contributes gate_rho=0.0 to
    both the negative-count and the median, exactly like Criteria A/B.
    """
    _validate_rho_by_task(raw_rho_by_task, tasks)
    if removed_task not in tasks:
        raise GateInputError(f"removed_task {removed_task!r} is not one of {tasks}")
    remaining = [t for t in tasks if t != removed_task]
    gate_rho_remaining = {t: to_gate_rho(raw_rho_by_task[t]) for t in remaining}
    negative_count = sum(1 for t in remaining if gate_rho_remaining[t] < 0)
    median_remaining = _median([gate_rho_remaining[t] for t in remaining])
    passes = negative_count >= 2 and median_remaining < 0
    return {
        "remaining_tasks": remaining,
        "raw_rho_remaining": {t: raw_rho_by_task[t] for t in remaining},
        "gate_rho_remaining": gate_rho_remaining,
        "negative_count": negative_count,
        "required_negative_count": 2,
        "median_remaining_gate_rho": median_remaining,
        "pass": passes,
    }


def evaluate_criterion_d(layer0_raw_rho):
    """D (hardened, Stage-G3B-IMPL): the Layer-0 n=3 diagnostic correlation
    (relative_l2 vs SignedSensitivity across lcc/multifieldqa_en/samsum,
    from compute_layer0_diagnostic_rho) must have gate_rho < 0. No
    significance threshold (n=3 is too small for one). If either n=3
    vector has zero variance, raw_rho is undefined -> gate_rho=0.0 -> this
    criterion FAILS (0 is not < 0) -- undefined here can never pass by
    default. Distribution features (p99_abs/outlier_fraction) may never be
    substituted here after seeing data -- this criterion is defined on
    relative_l2 only.
    """
    gate_rho = to_gate_rho(layer0_raw_rho)
    return {"layer0_raw_rho": layer0_raw_rho, "layer0_gate_rho": gate_rho, "pass": gate_rho < 0}


def evaluate_axis_gate(raw_rho_by_task, layer0_raw_rho, threshold=DEFAULT_EFFECT_SIZE_THRESHOLD, tasks=PRIMARY_STAGE_E_TASKS):
    """Evaluates all four criteria for ONE tensor axis (Key or Value) and
    returns whether that axis alone satisfies the full pre-registered gate.
    All inputs are RAW rho (None allowed = undefined); the raw-to-gate_rho=0
    conversion (to_gate_rho) happens once per criterion, consistently.
    """
    a = evaluate_criterion_a(raw_rho_by_task, tasks)
    b = evaluate_criterion_b(raw_rho_by_task, threshold, tasks)
    c = evaluate_criterion_c(raw_rho_by_task, LOTO_REMOVED_TASK, tasks)
    d = evaluate_criterion_d(layer0_raw_rho)
    return {
        "criterion_a": a,
        "criterion_b": b,
        "criterion_c": c,
        "criterion_d": d,
        "axis_go": a["pass"] and b["pass"] and c["pass"] and d["pass"],
    }


def evaluate_feature_pilot_gate(
    key_raw_rho_by_task,
    key_layer0_raw_rho,
    value_raw_rho_by_task,
    value_layer0_raw_rho,
    threshold=DEFAULT_EFFECT_SIZE_THRESHOLD,
    tasks=PRIMARY_STAGE_E_TASKS,
):
    """The final pre-registered gate: FEATURE_PILOT_STAGE = GO if AND ONLY
    IF at least one of {Key, Value} independently satisfies criteria
    A-D (evaluate_axis_gate). No CONDITIONAL_GO. Distribution features
    (p99_abs/outlier_fraction) remain separately reported as primary
    exploratory features but never substitute for a failed
    reconstruction-error (relative_l2) gate on either axis. All *_rho
    inputs are RAW rho (None = undefined, per to_gate_rho's convention).
    """
    key_result = evaluate_axis_gate(key_raw_rho_by_task, key_layer0_raw_rho, threshold, tasks)
    value_result = evaluate_axis_gate(value_raw_rho_by_task, value_layer0_raw_rho, threshold, tasks)
    stage = "GO" if (key_result["axis_go"] or value_result["axis_go"]) else "NO_GO"
    return {"key": key_result, "value": value_result, "FEATURE_PILOT_STAGE": stage}
