"""Data audit for progress report 2: dataset statistics, split leakage, image-free baselines.

Runs on CPU from the HF metadata only (no images needed).
"""
import json
from collections import Counter
from pathlib import Path

from .baselines import majority, question_only
from .metrics import vqa_scores
from .mpvqa import load_metadata_json, load_subset, sub_images_of
from .splits import audit, concept_folds, legacy_split


def _find(folder, name):
    """HF snapshot keeps files under ``test/``; a metadata-only download may keep them at the top."""
    for candidate in (folder / "test" / name, folder / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(folder / "test" / name)


def load_all(data_root):
    """Rows of both subsets plus the synthetic generation metadata. Images live in ``<subset>/test/``."""
    root = Path(data_root)
    syn = load_subset(_find(root / "synthetic", "metadata.jsonl"), root / "synthetic/test", "synthetic")
    real = load_subset(_find(root / "real-world", "metadata.jsonl"), root / "real-world/test", "real-world")
    meta = load_metadata_json(_find(root / "synthetic", "meta_data_VQAv2.json"),
                              _find(root / "synthetic", "meta_data_MegicBrush.json"))
    return syn, real, meta


def stats(rows):
    closed = [r for r in rows if r["closed"]]
    out = {"images": len({r["image_id"] for r in rows}), "qa": len(rows), "qa_closed": len(closed),
           "answers": {q: dict(Counter(r["answer"] for r in closed if r["question_type"] == q).most_common())
                       for q in ("q1", "q2", "q3")}}
    if rows and rows[0]["subset"] == "synthetic":
        out["concepts"] = len({r["concept"] for r in rows})
        out["cores"] = len({r["core"] for r in rows})
        out["variants"] = dict(Counter(r["variant"] for r in rows if r["question_type"] == "q1"))
    return out


def _brief(scores):
    keys = ["overall", "type:q1", "type:q2", "type:q3", "consistency"]
    return {k: round(100 * scores[k]["accuracy"], 2) for k in keys
            if k in scores and scores[k]["accuracy"] is not None}


def run(data_root, output, n_folds=5, seed=42):
    syn, real, meta = load_all(data_root)
    sub = lambda r: sub_images_of(r, meta)  # noqa: E731
    result = {"stats": {"synthetic": stats(syn), "real-world": stats(real)},
              "unique_sub_images": len(set().union(*(sub(r) for r in syn)))}

    legacy = legacy_split(syn, seed=seed)
    result["legacy_split"] = {
        "sizes": {k: {"images": len({r["image_id"] for r in v}), "qa_closed": sum(r["closed"] for r in v)}
                  for k, v in legacy.items()},
        "leakage_test_vs_train": audit(legacy["train"], legacy["test"], sub),
        "baselines": {}}
    for name, fn in (("majority", majority), ("question_only", question_only)):
        result["legacy_split"]["baselines"][name] = {
            "test_synthetic": _brief(vqa_scores(legacy["test"], fn(legacy["train"], legacy["test"]))),
            "real_world": _brief(vqa_scores(real, fn(legacy["train"], real)))}

    folds = concept_folds(syn, n_folds=n_folds, seed=seed)
    result["concept_folds"] = []
    pooled = {"majority": {}, "question_only": {}}
    for fold in folds:
        entry = {"fold": fold["fold"], "test_concepts": fold["test_concepts"],
                 "val_concepts": fold["val_concepts"],
                 "sizes": {k: sum(r["closed"] for r in fold[k]) for k in ("train", "val", "test")},
                 "leakage_test_vs_train": audit(fold["train"], fold["test"], sub)}
        for name, fn in (("majority", majority), ("question_only", question_only)):
            predictions = fn(fold["train"], fold["test"])
            pooled[name].update(predictions)
            entry[name] = _brief(vqa_scores(fold["test"], predictions))
        result["concept_folds"].append(entry)
    result["concept_cv_pooled"] = {name: _brief(vqa_scores(syn, p)) for name, p in pooled.items()}
    result["chance"] = {"synthetic": {k: round(100 * v["chance"], 2) for k, v in vqa_scores(
        syn, {r["question_id"]: r["answer"] for r in syn}).items() if "chance" in v and k.startswith(("overall", "type:"))},
        "real_world": {k: round(100 * v["chance"], 2) for k, v in vqa_scores(
            real, {r["question_id"]: r["answer"] for r in real}).items() if "chance" in v and k.startswith(("overall", "type:"))}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
