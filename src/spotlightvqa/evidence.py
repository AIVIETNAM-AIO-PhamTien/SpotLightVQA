"""Evaluate the pre-registered hypotheses of progress report 2 from the run outputs.

H2 (primary)  Spotlight beats Direct on Q2+Q3 with the same VLM, on the same questions.
              One test per subset (synthetic, real-world), Holm-corrected, alpha = 0.05.
H3 (mechanism) On synthetic questions with a known target box:
              (a) Direct -> Oracle crop measures what zooming on the right panel is worth;
                  Oracle -> Spotlight measures what imperfect self-grounding costs;
              (b) Spotlight is right more often when its box hits the target (IoU >= 0.5).
H4 (training) On concept-held-out synthetic test questions (5 folds pooled):
              E5 (Spotlight loss) vs E2 (report-1 recipe) and vs the question-only floor;
              E0 vs E2 on the legacy test questions measures memorisation.

    python -m spotlightvqa.evidence --runs outputs --output outputs/evidence.json
"""
import argparse
import json
from pathlib import Path

from .audit import load_all
from .baselines import question_only
from .mpvqa import iou
from .splits import concept_folds, legacy_split
from .stats import correct, holm, paired_compare
from .summarize import _RUN, _read_predictions


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def vlm_evidence(rows_by_id, run):
    records = [r for r in _records(run / "predictions.jsonl") if "direct" in r]
    if not records:
        return None
    pred = {r["question_id"]: r["prediction"] for r in records}
    direct = {r["question_id"]: r["direct"] for r in records}
    out = {"run": run.name, "H2": {}, "H2_secondary": {}, "H3": {}}
    for subset in ("synthetic", "real-world"):
        rows = [rows_by_id[q] for q in pred if rows_by_id[q]["subset"] == subset]
        out["H2"][subset] = paired_compare([r for r in rows if r["question_type"] != "q1"],
                                           direct, pred, "Direct", "Spotlight")
        for qtype in ("q2", "q3"):
            out["H2_secondary"][f"{subset}:{qtype}"] = paired_compare(
                [r for r in rows if r["question_type"] == qtype], direct, pred, "Direct", "Spotlight")
    primary = [k for k, v in out["H2"].items() if v]
    for key, p in zip(primary, holm([out["H2"][k]["p_mcnemar"] for k in primary])):
        out["H2"][key]["p_holm"] = p

    oracle_ids = [r["question_id"] for r in records if "oracle" in r]
    if oracle_ids:
        rows = [rows_by_id[q] for q in oracle_ids]
        oracle = {r["question_id"]: r["oracle"] for r in records if "oracle" in r}
        out["H3"]["q2_direct_vs_oracle"] = paired_compare(rows, direct, oracle, "Direct", "Oracle crop")
        out["H3"]["q2_spotlight_vs_oracle"] = paired_compare(rows, pred, oracle, "Spotlight", "Oracle crop")
    hits = {True: [], False: []}
    for record in records:
        if not record.get("gt_box") or rows_by_id[record["question_id"]]["question_type"] == "q1":
            continue
        row = rows_by_id[record["question_id"]]
        box = record.get("box") if row["question_type"] == "q2" else (record.get("boxes") or {}).get(row["answer"])
        hit = iou(box, record["gt_box"]) >= 0.5
        hits[hit].append(correct([row], {row["question_id"]: record["prediction"]})[0])
    total = len(hits[True]) + len(hits[False])
    if total:
        out["H3"]["grounding"] = {
            "n": total, "hit_rate_iou50": 100 * len(hits[True]) / total,
            "spotlight_acc_when_hit": 100 * sum(hits[True]) / len(hits[True]) if hits[True] else None,
            "spotlight_acc_when_miss": 100 * sum(hits[False]) / len(hits[False]) if hits[False] else None}
    return out


def training_evidence(syn, runs_root, seed=42):
    pooled = {}
    for metrics in sorted(Path(runs_root).glob("train/*/metrics.json")):
        match = _RUN.match(metrics.parent.name)
        if match:
            key = f"{match.group(1)}_{match.group(3)}"
            pooled.setdefault(key, {}).update(_read_predictions(metrics.parent / "predictions_test.jsonl"))
    closed = [r for r in syn if r["closed"]]
    out = {}
    concept = {k: v for k, v in pooled.items() if k.startswith("concept_")}
    floor = {}
    for fold in concept_folds(syn, seed=seed):
        floor.update(question_only(fold["train"], fold["test"]))
    complete = {k: v for k, v in concept.items() if all(r["question_id"] in v for r in closed)}
    names = sorted(complete)
    base = next((n for n in names if "E2" in n), None)
    for name in names:
        rows = closed
        out[f"{name}_vs_question_only"] = paired_compare(rows, floor, complete[name], "Question-only", name)
        if base and name != base:
            out[f"{name}_vs_{base}"] = paired_compare(rows, complete[base], complete[name], base, name)
    legacy = next((v for k, v in pooled.items() if k.startswith("legacy_")), None)
    if legacy and base:
        test_ids = {r["question_id"] for r in legacy_split(syn, seed=seed)["test"]}
        rows = [r for r in closed if r["question_id"] in test_ids and r["question_id"] in legacy]
        out["memorisation_E0_vs_E2_on_legacy_test"] = paired_compare(
            rows, legacy, complete[base], "E0 (legacy split)", f"{base} (concept held out)")
    out["incomplete_runs"] = sorted(set(concept) - set(complete))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--runs", default="outputs")
    p.add_argument("--output", default="outputs/evidence.json")
    args = p.parse_args(argv)
    syn, real, _ = load_all(args.data_root)
    rows_by_id = {r["question_id"]: r for r in syn + real if r["closed"]}
    result = {"vlm": [e for run in sorted(Path(args.runs).glob("vlm/*")) if (run / "predictions.jsonl").exists()
                      for e in [vlm_evidence(rows_by_id, run)] if e],
              "training": training_evidence(syn, args.runs)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
