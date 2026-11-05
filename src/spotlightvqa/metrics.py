"""Deterministic local metric; deliberately separate from the GPT-4 judge."""
import re
from collections import defaultdict


def normalize_answer(answer: str) -> str:
    if not isinstance(answer, str):
        raise TypeError("Answers must be strings")
    return re.sub(r"\s+", " ", answer.replace("</s>", "").translate(
        str.maketrans("", "", ").,"))).strip().lower()


def evaluate(records: list[dict], predictions: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot evaluate an empty dataset")
    truth = {r["question_id"]: r for r in records}
    pred = {r["question_id"]: r["prediction"] for r in predictions}
    if len(truth) != len(records) or len(pred) != len(predictions):
        raise ValueError("Duplicate question_id")
    if truth.keys() != pred.keys():
        raise ValueError("Prediction IDs must exactly match dataset IDs")
    groups = defaultdict(lambda: [0, 0])
    details = []
    for qid, row in truth.items():
        expected = normalize_answer(row["answer"])
        correct = bool(expected) and normalize_answer(pred[qid]) == expected
        for key in ("overall", "subset:" + row["subset"], "type:" + str(row["question_type"])):
            groups[key][0] += int(correct)
            groups[key][1] += 1
        details.append({"question_id": qid, "prediction": pred[qid], "correct": correct})
    return {"metric": "normalized_exact_match_v1", "official_paper_metric": False,
            "scores": {k: {"correct": c, "total": n, "accuracy": c / n}
                       for k, (c, n) in groups.items()}, "details": details}


def vqa_scores(rows, predictions, closed_only=True):
    """Closed-set MultipanelVQA scores, grouped per question type, subset, style and panel count.

    ``predictions`` maps question_id -> predicted label. Each row is grouped by its *own* style
    (report 1 printed the overall accuracy for every style). Consistency counts an image only
    when all three of its questions are in the evaluated set and all three are correct; images
    with an open-ended question are left out of the consistency denominator instead of
    silently counted as failures. ``chance`` is the expected accuracy of uniform guessing over
    the valid labels of each question (2 for yes/no, number of options for Q3).
    """
    rows = [r for r in rows if r["closed"]] if closed_only else list(rows)
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    missing = [r["question_id"] for r in rows if r["question_id"] not in predictions]
    if missing:
        raise ValueError(f"{len(missing)} predictions missing, e.g. {missing[0]}")
    groups = defaultdict(lambda: [0, 0, 0.0])
    per_image = defaultdict(list)
    for row in rows:
        correct = normalize_answer(str(predictions[row["question_id"]])) == normalize_answer(row["answer"])
        chance = 1.0 / (row["n_options"] if row["question_type"] == "q3" else 2)
        keys = ["overall", "type:" + row["question_type"], "subset:" + row["subset"],
                "style:" + row.get("style", "?")]
        if row.get("variant"):
            keys.append("variant:" + row["variant"])
        if row.get("n_panels"):
            keys.append(f"panels:{row['n_panels']}")
        for key in keys:
            groups[key][0] += int(correct)
            groups[key][1] += 1
            groups[key][2] += chance
        per_image[row["image_id"]].append(correct)
    scores = {k: {"correct": c, "total": n, "accuracy": c / n, "chance": ch / n}
              for k, (c, n, ch) in sorted(groups.items())}
    complete = [v for v in per_image.values() if len(v) == 3]
    scores["consistency"] = {"correct": sum(all(v) for v in complete), "total": len(complete),
                             "accuracy": sum(all(v) for v in complete) / len(complete) if complete else None}
    return scores
