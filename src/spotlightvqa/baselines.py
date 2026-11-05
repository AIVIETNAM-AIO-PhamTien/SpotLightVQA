"""Image-free reference baselines (CPU): majority answer and question-only classifier.

They measure how much of a score the *question text alone* explains. A vision model is only
useful where it clearly beats these numbers.
"""
from collections import Counter

from .mpvqa import valid_labels


def majority(train, test):
    """Predict the most frequent train answer of the question type (restricted to valid labels)."""
    counts = {}
    for row in train:
        if row["closed"]:
            counts.setdefault(row["question_type"], Counter())[row["answer"]] += 1
    predictions = {}
    for row in test:
        allowed = valid_labels(row)
        ranked = [a for a, _ in counts.get(row["question_type"], Counter()).most_common() if a in allowed]
        predictions[row["question_id"]] = ranked[0] if ranked else allowed[0]
    return predictions


def question_only(train, test, seed=42):
    """TF-IDF (word + char n-grams) + logistic regression per question type, no image."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    predictions = majority(train, test)
    for qtype in ("q1", "q2", "q3"):
        rows = [r for r in train if r["closed"] and r["question_type"] == qtype]
        targets = [r for r in test if r["question_type"] == qtype]
        if not targets or len({r["answer"] for r in rows}) < 2:
            continue
        model = make_pipeline(
            make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            LogisticRegression(max_iter=2000, C=4.0, random_state=seed))
        model.fit([r["question"] for r in rows], [r["answer"] for r in rows])
        classes = list(model.classes_)
        for row, proba in zip(targets, model.predict_proba([r["question"] for r in targets])):
            allowed = set(valid_labels(row))
            scored = [(p, c) for p, c in zip(proba, classes) if c in allowed]
            if scored:
                predictions[row["question_id"]] = max(scored)[1]
    return predictions
