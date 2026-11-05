"""Train/val/test splits for training on MultipanelVQA, plus a leakage audit.

MultipanelVQA is released as a *test* benchmark. Synthetic images come from only 10
concepts (5 VQAv2 + 5 MagicBrush questions); every layout of a concept reuses the same
sub-images and the same question. Splitting by image or by Q1||Q2 text therefore puts the
same sub-images and the same question in train and test. ``concept_folds`` holds whole
concepts out so the test fold contains objects/questions never seen during training.
"""
import random
from collections import defaultdict


def _by_group(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def legacy_split(rows, seed=42, ratios=(0.70, 0.15, 0.15)):
    """Reproduce report 1: shuffle groups keyed by Q1||Q2 text of each image, 70/15/15 by image."""
    per_image = _by_group(rows, lambda r: r["image_id"])
    key_of = {}
    for image_id, qa in per_image.items():
        text = {r["question_type"]: r["question"] for r in qa}
        key_of[image_id] = text.get("q1", "") + "||" + text.get("q2", "")
    groups = _by_group(per_image, lambda image_id: key_of[image_id])
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total, out, filled = len(per_image), {"train": [], "val": [], "test": []}, 0
    bounds = (ratios[0] * total, (ratios[0] + ratios[1]) * total)
    for name in names:
        part = "train" if filled < bounds[0] else "val" if filled < bounds[1] else "test"
        for image_id in groups[name]:
            out[part].extend(per_image[image_id])
        filled += len(groups[name])
    return out


def concept_folds(rows, n_folds=5, seed=42):
    """Leave-concepts-out folds. Each fold tests on ``10 / n_folds`` concepts never trained on.

    Concepts are dealt per source so every fold has VQAv2 and MagicBrush concepts when possible.
    Validation of fold k uses the test concepts of fold k+1 (also unseen in training).
    """
    concepts = defaultdict(list)
    for concept in sorted({r["concept"] for r in rows}):
        concepts[concept.split(":")[0]].append(concept)
    rng = random.Random(seed)
    assigned = [[] for _ in range(n_folds)]
    position = 0
    for source in sorted(concepts):
        names = concepts[source]
        rng.shuffle(names)
        for name in names:
            assigned[position % n_folds].append(name)
            position += 1
    folds = []
    for k in range(n_folds):
        test, val = set(assigned[k]), set(assigned[(k + 1) % n_folds])
        folds.append({"fold": k, "test_concepts": sorted(test), "val_concepts": sorted(val),
                      "train": [r for r in rows if r["concept"] not in test | val],
                      "val": [r for r in rows if r["concept"] in val],
                      "test": [r for r in rows if r["concept"] in test]})
    return folds


def audit(train, test, sub_images=None):
    """Count what the test part shares with the train part."""
    def share(key):
        seen = {key(r) for r in train}
        hits = [r for r in test if key(r) in seen]
        return {"test_rows_sharing": len(hits), "test_rows": len(test),
                "fraction": len(hits) / len(test) if test else 0.0}

    report = {"concept": share(lambda r: r["concept"]), "core_qa": share(lambda r: r["core"]),
              "question_text": share(lambda r: r["question"])}
    if sub_images is not None:
        train_images = set().union(*(sub_images(r) for r in train)) if train else set()
        test_rows = [r for r in test if r["subset"] == "synthetic"]
        hits = [r for r in test_rows if sub_images(r) & train_images]
        full = [r for r in test_rows if sub_images(r) <= train_images]
        report["sub_image_any"] = {"test_rows_sharing": len(hits), "test_rows": len(test_rows),
                                   "fraction": len(hits) / len(test_rows) if test_rows else 0.0}
        report["sub_image_all"] = {"test_rows_sharing": len(full), "test_rows": len(test_rows),
                                   "fraction": len(full) / len(test_rows) if test_rows else 0.0}
    return report
