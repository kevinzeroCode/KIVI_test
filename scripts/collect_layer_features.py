"""Stage G1: layer x task K/V feature-collection driver -- DRY-RUN SKELETON ONLY.

This round implements ONLY --dry-run and CPU-testable configuration
validation: argument parsing, task/layer/axis validation (reusing the same
task universe as the rest of the project), output-plan preview (what
layer x task x axis feature records WOULD be produced), and the same
conflicting-process safety check used by run_layer_sensitivity_pilot.py.

It deliberately does NOT (this round):
  - import torch, transformers, or datasets
  - load a model or run any forward pass
  - hook k_proj/v_proj or capture any real activation
  - write any feature-record file under --output-root

Those all require the still-open GPU-parity items documented in Stage G1's
audit (see utils/feature_extraction.py's module docstring and the Stage G1
final report) and are explicitly deferred to a later round.

Usage (the only supported mode this round):
    ./.venv/bin/python scripts/collect_layer_features.py --dry-run
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.pilot_policy import (  # noqa: E402
    EXPECTED_NUM_LAYERS,
    PILOT_AXES,
    SUPPORTED_TASK_COUNTS,
    select_task_counts,
)
from utils.feature_schema import FEATURE_RECORD_FIELDS  # noqa: E402 -- torch-free, keeps --dry-run import-light

DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_feature_pilot")
DEFAULT_TASKS = ("trec", "lcc", "passage_retrieval_en", "2wikimqa")
DEFAULT_NUM_SAMPLES = 20

# Mirrors run_layer_sensitivity_pilot.py's conflicting-process discipline
# (see that script's check_no_conflicting_process docstring for why the
# genuine-invocation regex and "bash -c" exclusion exist).
CONFLICTING_PROCESS_PATTERNS = [
    "pred_long_bench.py",
    "layer_policy_smoke.py",
    "run_layer_sensitivity_pilot.py",
    "collect_layer_features.py",
]


class FeatureCollectionConfigError(ValueError):
    pass


def check_no_conflicting_process(patterns=CONFLICTING_PROCESS_PATTERNS, self_pid=None):
    self_pid = self_pid if self_pid is not None else os.getpid()
    pattern = "|".join(re.escape(p) for p in patterns)
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        return ["WARNING: pgrep not available; conflicting-process check could not run"]

    genuine_invocation_re = re.compile(
        r"(?:^|/|\s)\S*python\S*\s+\S*(?:" + "|".join(re.escape(p) for p in patterns) + r")\b"
    )
    conflicts = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str = line.split(None, 1)[0]
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if "bash -c" in line:
            continue
        if not genuine_invocation_re.search(line):
            continue
        conflicts.append(line)
    return conflicts


def validate_layers(layers, num_hidden_layers=EXPECTED_NUM_LAYERS):
    if not layers:
        raise FeatureCollectionConfigError("--layers must be non-empty")
    bad = [l for l in layers if not (0 <= l < num_hidden_layers)]
    if bad:
        raise FeatureCollectionConfigError(f"layer index/indices out of range [0, {num_hidden_layers}): {bad}")
    if len(set(layers)) != len(layers):
        raise FeatureCollectionConfigError(f"duplicate layer indices in --layers: {layers}")
    return list(layers)


def validate_axes(axes, supported_axes=PILOT_AXES):
    if not axes:
        raise FeatureCollectionConfigError("--axes must be non-empty")
    bad = [a for a in axes if a not in supported_axes]
    if bad:
        raise FeatureCollectionConfigError(f"unsupported axis/axes {bad}; supported: {list(supported_axes)}")
    return list(axes)


def validate_num_samples(num_samples):
    if num_samples <= 0:
        raise FeatureCollectionConfigError(f"--num-samples must be positive, got {num_samples}")
    return num_samples


def build_collection_plan(layers, axes, task_counts, num_samples):
    """Returns the deterministic, ordered list of (task, layer, axis) cells
    this configuration WOULD collect features for. Never merges tasks --
    preserves feature(layer, task) granularity per the Stage G0/G1 design
    requirement that layer-index alone cannot explain task-dependent
    response.
    """
    plan = []
    for task in task_counts:
        n = min(num_samples, task_counts[task])
        for layer in layers:
            for axis in axes:
                plan.append({"task": task, "layer_idx": layer, "axis": axis, "num_samples": n})
    return plan


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--cache_dir", default="./cached_models")
    p.add_argument("--tasks", nargs="+", default=None, help="Defaults to the same 4-task Stage-D screening set. Any task in utils.pilot_policy.SUPPORTED_TASK_COUNTS may be requested explicitly.")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Per-task sample cap (capped further by that task's available example count).")
    p.add_argument("--layers", type=int, nargs="+", default=list(range(EXPECTED_NUM_LAYERS)))
    p.add_argument("--axes", nargs="+", default=list(PILOT_AXES), choices=list(PILOT_AXES))
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if not args.dry_run:
        print(
            "Real feature collection (model load, forward pass, activation capture) is "
            "not implemented in this round -- only --dry-run configuration validation is "
            "available. Refusing to proceed.",
            file=sys.stderr,
        )
        return 1

    task_names = list(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    task_counts = select_task_counts(task_names)
    layers = validate_layers(args.layers)
    axes = validate_axes(args.axes)
    num_samples = validate_num_samples(args.num_samples)

    plan = build_collection_plan(layers, axes, task_counts, num_samples)
    conflicts = check_no_conflicting_process()

    print(f"Selected tasks ({len(task_counts)}): {list(task_counts)}")
    print(f"Layers ({len(layers)}): {layers}")
    print(f"Axes ({len(axes)}): {axes}")
    print(f"Per-task sample cap: {num_samples}")
    print(f"\nPlanned feature-record cells (task x layer x axis, per-sample granularity preserved): {len(plan)}")
    total_samples = sum(cell["num_samples"] for cell in plan)
    print(f"Planned total (task, layer, axis, sample) feature records: {total_samples}")
    print(f"\nFeature record schema ({len(FEATURE_RECORD_FIELDS)} fields): {list(FEATURE_RECORD_FIELDS)}")
    print(f"\nOutput root (not written this round): {args.output_root}")
    print(f"\nConflicting-process check ({', '.join(CONFLICTING_PROCESS_PATTERNS)}): "
          f"{'NONE FOUND' if not conflicts else conflicts}")
    print(
        "\nNo model, torch, or transformers import occurred; no activation capture, no "
        "forward pass, no output file was written (--dry-run configuration validation only)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
