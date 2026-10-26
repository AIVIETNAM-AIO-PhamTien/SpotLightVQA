import pytest

from spotlightvqa.metrics import evaluate, normalize_answer


def rows():
    return [{"question_id": "a", "answer": "Red.", "subset": "synthetic", "question_type": "qa_0"},
            {"question_id": "b", "answer": "no", "subset": "real-world", "question_type": "qa_1"},
            {"question_id": "c", "answer": "b)", "subset": "real-world", "question_type": "qa_2"}]


def test_normalization():
    assert normalize_answer("  RED.</s>  ") == "red"
    assert normalize_answer("a   red, dog") == "a red dog"


def test_question_denominator_and_groups():
    result = evaluate(rows(), [{"question_id": "c", "prediction": "B"},
                               {"question_id": "a", "prediction": "red"},
                               {"question_id": "b", "prediction": "yes"}])
    assert result["scores"]["overall"] == {"correct": 2, "total": 3, "accuracy": 2 / 3}
    assert result["scores"]["subset:real-world"]["accuracy"] == 0.5
    assert result["official_paper_metric"] is False


@pytest.mark.parametrize("predictions", [[], [{"question_id": "a", "prediction": "red"}] * 3,
    [{"question_id": k, "prediction": "red"} for k in ["a", "b", "unknown"]]])
def test_invalid_alignment(predictions):
    with pytest.raises(ValueError):
        evaluate(rows(), predictions)


def test_no_substring_or_yes_no_shortcut():
    records = [{"question_id": "a", "answer": "yes", "subset": "synthetic", "question_type": "qa_0"}]
    assert evaluate(records, [{"question_id": "a", "prediction": "yes and no"}])["scores"]["overall"]["correct"] == 0
    records[0]["answer"] = ""
    assert evaluate(records, [{"question_id": "a", "prediction": ""}])["scores"]["overall"]["correct"] == 0
    with pytest.raises(ValueError):
        evaluate([], [])
