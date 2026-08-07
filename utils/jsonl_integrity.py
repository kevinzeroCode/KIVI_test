"""Strict JSONL validation shared by pred_long_bench.py and
scripts/repair_partial_jsonl.py.

Deliberately has no dependency on torch/transformers/datasets so the repair
tool (and its tests) can import it without pulling in the full model stack.
"""
import json
import os
from collections import namedtuple

JsonlInspection = namedtuple(
    "JsonlInspection",
    [
        "exists",
        "valid_rows",
        "nonempty_rows",
        "invalid_rows",
        "first_invalid_line",
        "invalid_line_numbers",
        "valid_prefix_rows",
        "corruption_is_tail_only",
        "ends_with_newline",
    ],
)


def inspect_jsonl(path):
    """Strictly validate a JSONL file line-by-line.

    Every non-empty line must `json.loads` to a JSON object (dict). This
    replaces a naive `sum(1 for line in f if line.strip())` row count, which
    would miscount a corrupted line (e.g. a NUL-filled tail left by an
    abrupt host crash/freeze) as a completed row -- silently shifting
    resume's start_idx past a sample that was never actually generated.

    `valid_prefix_rows` counts valid rows only up to the first invalid line;
    `corruption_is_tail_only` is True iff every invalid line comes after
    every valid line (i.e. valid rows form a contiguous prefix), which is
    the only shape scripts/repair_partial_jsonl.py is allowed to repair.
    """
    if not os.path.exists(path):
        return JsonlInspection(
            exists=False,
            valid_rows=0,
            nonempty_rows=0,
            invalid_rows=0,
            first_invalid_line=None,
            invalid_line_numbers=[],
            valid_prefix_rows=0,
            corruption_is_tail_only=True,
            ends_with_newline=True,
        )

    with open(path, "rb") as f:
        raw = f.read()
    ends_with_newline = raw.endswith(b"\n") if raw else True

    text = raw.decode("utf-8", errors="surrogateescape")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    valid_rows = 0
    nonempty_rows = 0
    invalid_line_numbers = []
    valid_prefix_rows = 0
    prefix_still_valid = True

    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        nonempty_rows += 1
        try:
            obj = json.loads(line)
            is_valid = isinstance(obj, dict)
        except (json.JSONDecodeError, ValueError):
            is_valid = False

        if is_valid:
            valid_rows += 1
            if prefix_still_valid:
                valid_prefix_rows += 1
        else:
            invalid_line_numbers.append(line_no)
            prefix_still_valid = False

    invalid_rows = nonempty_rows - valid_rows
    first_invalid_line = invalid_line_numbers[0] if invalid_line_numbers else None
    corruption_is_tail_only = valid_prefix_rows == valid_rows

    return JsonlInspection(
        exists=True,
        valid_rows=valid_rows,
        nonempty_rows=nonempty_rows,
        invalid_rows=invalid_rows,
        first_invalid_line=first_invalid_line,
        invalid_line_numbers=invalid_line_numbers,
        valid_prefix_rows=valid_prefix_rows,
        corruption_is_tail_only=corruption_is_tail_only,
        ends_with_newline=ends_with_newline,
    )
