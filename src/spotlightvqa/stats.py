"""Paired significance tests for comparing two predictors on the same questions.

* McNemar exact test on the discordant pairs (two-sided binomial, no normal approximation).
* Cluster bootstrap CI of the accuracy difference: images are resampled, not questions,
  because the 3 questions of an image (and the 5 variants of a core) are not independent.
* Holm step-down correction for a family of hypotheses.
"""
import math
import random
from collections import defaultdict

from .metrics import normalize_answer


def correct(rows, predictions):
    return [normalize_answer(str(predictions[r["question_id"]])) == normalize_answer(r["answer"]) for r in rows]


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value; ``a``/``b`` are per-question correctness lists."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0, only_a, only_b
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), only_a, only_b


def cluster_bootstrap_diff(rows, a, b, cluster="image_id", n_boot=5000, seed=42):
    """95% percentile CI of acc(b) - acc(a), resampling clusters with replacement."""
    groups = defaultdict(lambda: [0, 0, 0])  # (correct_a, correct_b, n)
    for row, x, y in zip(rows, a, b):
        g = groups[row[cluster]]
        g[0] += x
        g[1] += y
        g[2] += 1
    stats = list(groups.values())
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ca = cb = n = 0
        for _ in range(len(stats)):
            s = stats[rng.randrange(len(stats))]
            ca, cb, n = ca + s[0], cb + s[1], n + s[2]
        diffs.append((cb - ca) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def paired_compare(rows, pred_a, pred_b, name_a="A", name_b="B", **bootstrap):
    """Everything needed to claim ``B`` beats ``A`` on these rows (or that it does not)."""
    if not rows:
        return None
    a, b = correct(rows, pred_a), correct(rows, pred_b)
    p, only_a, only_b = mcnemar_exact(a, b)
    low, high = cluster_bootstrap_diff(rows, a, b, **bootstrap)
    return {"a": name_a, "b": name_b, "n_questions": len(rows),
            "n_images": len({r["image_id"] for r in rows}),
            "acc_a": 100 * sum(a) / len(a), "acc_b": 100 * sum(b) / len(b),
            "diff": 100 * (sum(b) - sum(a)) / len(a), "ci95": [100 * low, 100 * high],
            "only_a_correct": only_a, "only_b_correct": only_b, "p_mcnemar": p}


def holm(pvalues):
    """Holm-adjusted p-values, same order as the input."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
