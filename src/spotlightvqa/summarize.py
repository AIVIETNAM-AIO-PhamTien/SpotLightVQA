"""Collect training / VLM runs into one table (JSON + LaTeX rows for the progress report).

Training runs named ``<split>_f<fold>_<tag>`` (e.g. ``concept_f3_full``) are grouped by
``<split>_<tag>``. For the concept split, test predictions of all folds are pooled so every
synthetic question is scored exactly once by a model that never saw its concept; real-world
scores are averaged over folds (mean and std).
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .audit import load_all
from .metrics import vqa_scores

COLUMNS = ("overall", "type:q1", "type:q2", "type:q3", "consistency")
_RUN = re.compile(r"^(legacy|concept)_f(\d+)_(.+)$")


def _read_predictions(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["question_id"]: r["prediction"] for r in map(json.loads, lines)}


def _row(scores):
    return {c: (100 * scores[c]["accuracy"] if c in scores and scores[c]["accuracy"] is not None else None)
            for c in COLUMNS}


def collect(data_root, runs_root):
    syn, _, _ = load_all(data_root)
    runs_root = Path(runs_root)
    groups = defaultdict(list)
    for metrics in sorted(runs_root.glob("train/*/metrics.json")):
        match = _RUN.match(metrics.parent.name)
        if match:
            groups[(match.group(1), match.group(3))].append(metrics.parent)
    table = {}
    for (split, tag), folders in sorted(groups.items()):
        pooled = {}
        real_rows = []
        for folder in folders:
            pooled.update(_read_predictions(folder / "predictions_test.jsonl"))
            real_rows.append(_row(json.loads((folder / "metrics.json").read_text())["real_world"]))
        tested = [r for r in syn if r["question_id"] in pooled]
        entry = {"folds": len(folders), "synthetic_test": _row(vqa_scores(tested, pooled)),
                 "real_world_mean": {c: statistics.fmean(v[c] for v in real_rows if v[c] is not None)
                                     if any(v[c] is not None for v in real_rows) else None for c in COLUMNS},
                 "real_world_std": {c: statistics.pstdev([v[c] for v in real_rows if v[c] is not None])
                                    if sum(v[c] is not None for v in real_rows) > 1 else 0.0 for c in COLUMNS}}
        if split == "concept":
            entry["complete_cv"] = len(tested) == sum(r["closed"] for r in syn)
        table[f"{split}_{tag}"] = entry
    for metrics in sorted(runs_root.glob("vlm/*/metrics.json")):
        data = json.loads(metrics.read_text())
        table[f"vlm_{metrics.parent.name}"] = {k.replace("-", "_"): _row(v) for k, v in data.items()}
    return table


def latex_rows(table):
    fmt = lambda v: "--" if v is None else f"{v:.2f}"  # noqa: E731
    lines = []
    for name, entry in table.items():
        for part in ("synthetic_test", "real_world_mean", "synthetic_direct", "synthetic",
                     "real_world_direct", "real_world"):
            if part in entry:
                lines.append(f"{name.replace('_', '-')} ({part.replace('_', ' ')}) & "
                             + " & ".join(fmt(entry[part][c]) for c in COLUMNS) + r" \\")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--runs", default="outputs")
    p.add_argument("--output", default="outputs/summary.json")
    args = p.parse_args(argv)
    table = collect(args.data_root, args.runs)
    Path(args.output).write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(json.dumps(table, indent=2))
    print(latex_rows(table))


if __name__ == "__main__":
    main()
