"""Deterministic layer x task K/V feature-record schema (Stage G1 Part 6).

Deliberately torch-free: this module exists separately from
utils/feature_extraction.py so that --dry-run configuration tooling (e.g.
scripts/collect_layer_features.py) can report the schema without importing
torch, matching the same deferred-heavy-import discipline used by
scripts/run_layer_sensitivity_pilot.py's --dry-run path.
"""

IDENTITY_FIELDS = ("task", "sample_idx", "layer_idx", "tensor_axis", "num_tokens")
RECONSTRUCTION_FIELDS = ("relative_l2", "mse", "max_abs_error")
DISTRIBUTION_FIELDS = ("mean", "std", "variance", "max_abs", "p50_abs", "p95_abs", "p99_abs", "outlier_fraction")
FEATURE_RECORD_FIELDS = IDENTITY_FIELDS + RECONSTRUCTION_FIELDS + DISTRIBUTION_FIELDS
