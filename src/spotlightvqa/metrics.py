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
