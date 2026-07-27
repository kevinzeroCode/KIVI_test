#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path


TASKS = [
    ("narrativeqa", "NarrativeQA", 20.65, 20.79),
    ("qasper", "Qasper", 29.42, 28.69),
    ("multifieldqa_en", "MultiFieldQA", 43.15, 41.02),
    ("hotpotqa", "HotpotQA", 33.05, 32.91),
    ("musique", "MuSiQue", 14.66, 13.82),
    ("2wikimqa", "2WikiMultihopQA", 24.14, 23.00),
    ("gov_report", "GovReport", 30.85, 30.47),
    ("qmsum", "QMSum", 22.84, 22.59),
    ("multi_news", "MultiNews", 26.55, 26.28),
    ("lcc", "LCC", 54.83, 54.11),
    ("repobench-p", "RepoBench-P", 58.94, 57.62),
    ("triviaqa", "TriviaQA", 83.99, 83.19),
    ("samsum", "SAMSum", 40.75, 41.28),
    ("trec", "TRec", 66.50, 66.50),
    ("passage_retrieval_en", "PassageRetrieval", 30.50, 32.25),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def count_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main():
    args = parse_args()
    result_path = args.prediction_dir / "result.json"
    with result_path.open("r", encoding="utf-8") as stream:
        scores = json.load(stream)

    expected = {task for task, *_ in TASKS}
    missing = sorted(expected - scores.keys())
    if missing:
        raise ValueError(f"Missing scores: {', '.join(missing)}")

    rows = []
    for task, label, baseline, paper_kivi2 in TASKS:
        prediction_path = args.prediction_dir / f"{task}.jsonl"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        reproduced = float(scores[task])
        rows.append(
            {
                "label": label,
                "baseline": baseline,
                "paper": paper_kivi2,
                "reproduced": reproduced,
                "delta": reproduced - paper_kivi2,
                "samples": count_jsonl(prediction_path),
            }
        )

    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    total_samples = sum(row["samples"] for row in rows)
    mean_abs_delta = sum(abs(row["delta"]) for row in rows) / len(rows)

    lines = [
        "# KIVI-2 LongBench Reproduction Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Experiment Setup",
        "",
        "- Model: `lmsys/longchat-7b-v1.5-32k`",
        "- Method: KIVI-2 (`k_bits=2`, `v_bits=2`)",
        "- Group size: `32`",
        "- Residual length: `128`",
        "- Context limit: `31,500` tokens",
        "- Decoding: greedy (`num_beams=1`, `do_sample=False`)",
        f"- Evaluation set: LongBench, 15 tasks, {total_samples:,} samples",
        "- Metric implementation: repository `eval_long_bench.py`",
        "",
        "The paper columns are copied from `docs/long_bench.md`, Table 1. "
        "Delta is reproduction minus paper KIVI-2.",
        "",
        "## Results",
        "",
        "| Task | Paper FP16 | Paper KIVI-2 | Reproduced KIVI-2 | Delta | Samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['baseline']:.2f} | {row['paper']:.2f} | "
            f"{row['reproduced']:.2f} | {row['delta']:+.2f} | {row['samples']} |"
        )
    lines.extend(
        [
            f"| **Average** | **{mean('baseline'):.2f}** | **{mean('paper'):.2f}** | "
            f"**{mean('reproduced'):.2f}** | **{mean('delta'):+.2f}** | **{total_samples:,}** |",
            "",
            "## Summary",
            "",
            f"- Reproduced average: **{mean('reproduced'):.2f}**",
            f"- Paper KIVI-2 average: **{mean('paper'):.2f}**",
            f"- Average delta: **{mean('delta'):+.2f}**",
            f"- Mean absolute per-task delta: **{mean_abs_delta:.2f}**",
            "",
            "## Runtime Compatibility Notes",
            "",
            "The 24 GB GPU run uses memory-bounded chunking for rotary embedding, "
            "KV packing, RMSNorm, and MLP prefill operations. The language-model head "
            "is evaluated on CPU, and predictions are checkpointed per sample so an "
            "interrupted run can resume. Quantization settings and LongBench decoding "
            "parameters remain those used for the reported KIVI-2 experiment.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
