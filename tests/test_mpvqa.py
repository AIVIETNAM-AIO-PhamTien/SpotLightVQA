import json

import pytest
from PIL import Image

from spotlightvqa.baselines import majority
from spotlightvqa.data import prepare
from spotlightvqa.metrics import vqa_scores
from spotlightvqa.mpvqa import load_subset, normalize_closed, parse_options, parse_synthetic_name, valid_labels
from spotlightvqa.splits import audit, concept_folds, legacy_split

Q3 = "Which subfigure has a dog?\na) top\nb) bottom left\nc) right\n(please select one)"


def test_closed_answer_normalization():
    assert normalize_closed("Yes.", "q1", "") == "yes"
    assert normalize_closed("no, her face is not covered yet", "q2", "") == "no"
    assert normalize_closed("indoors", "q2", "") is None
    assert normalize_closed("b)", "q3", Q3) == "b"
    assert normalize_closed("bottom left", "q3", Q3) == "b"   # option text, not letter
    assert normalize_closed("d", "q3", Q3) is None            # not among the listed options
    assert parse_options(Q3) == {"a": "top", "b": "bottom left", "c": "right"}


def test_synthetic_names():
    assert parse_synthetic_name("synthetic_image/magicbrush__0_2_22.png") == {
        "source": "magicbrush", "variant": "original", "concept_idx": 0, "n_panels": 2, "layout_id": 22}
    assert parse_synthetic_name("synthetic_image/VQA_with_background_images_4_8_11.png")["variant"] == \
        "with_background_images"


def _synthetic(tmp_path, concepts=4, layouts=3):
    records = []
    for source in ("VQA", "magicbrush"):
        for idx in range(concepts):
            for layout in range(layouts):
                for variant, style in (("", "original"), ("enlarged_images", "enlarged")):
                    name = f"synthetic_image/{source}_{variant}_{idx}_2_{layout}.png"
                    records.append({"file_name": name, "synthetic_style": style,
                                    "question_1": f"Do all subfigures contain obj{idx}?",
                                    "question_2": f"Is obj{idx} at top {layout}?",
                                    "question_3": Q3, "answer_1": "No", "answer_2": "Yes" if idx % 2 else "No",
                                    "answer_3": "a" if idx % 2 else "c)"})
    path = tmp_path / "metadata.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return load_subset(path, tmp_path, "synthetic")


def test_per_style_and_consistency(tmp_path):
    rows = _synthetic(tmp_path)
    truth = {r["question_id"]: r["answer"] for r in rows}
    wrong = next(r for r in rows if r["style"] == "enlarged" and r["question_type"] == "q3")
    truth[wrong["question_id"]] = "b"
    scores = vqa_scores(rows, truth)
    assert scores["style:original"]["accuracy"] == 1.0
    assert scores["style:enlarged"]["accuracy"] < 1.0      # styles no longer all equal overall
    assert scores["consistency"]["total"] == len({r["image_id"] for r in rows})
    assert scores["consistency"]["correct"] == scores["consistency"]["total"] - 1
    assert scores["type:q3"]["chance"] == pytest.approx(1 / 3)


def test_consistency_skips_images_with_open_questions(tmp_path):
    rows = _synthetic(tmp_path)
    for row in rows:
        if row["question_type"] == "q2":
            row["closed"] = False
    scores = vqa_scores(rows, {r["question_id"]: r["answer"] for r in rows})
    assert scores["consistency"]["total"] == 0 and scores["consistency"]["accuracy"] is None


def test_concept_folds_have_no_leakage(tmp_path):
    rows = _synthetic(tmp_path)
    folds = concept_folds(rows, n_folds=4)
    tested = []
    for fold in folds:
        assert not {r["concept"] for r in fold["train"]} & {r["concept"] for r in fold["test"]}
        assert not {r["concept"] for r in fold["train"]} & {r["concept"] for r in fold["val"]}
        assert audit(fold["train"], fold["test"])["concept"]["fraction"] == 0.0
        tested += fold["test"]
    assert sorted(r["question_id"] for r in tested) == sorted(r["question_id"] for r in rows)


def test_legacy_split_keeps_styles_together_but_leaks_concepts(tmp_path):
    rows = _synthetic(tmp_path)
    split = legacy_split(rows)
    cores = {part: {r["core"] for r in v} for part, v in split.items()}
    assert not cores["train"] & cores["test"]
    assert audit(split["train"], split["test"])["concept"]["fraction"] == 1.0


def test_majority_respects_valid_options(tmp_path):
    rows = _synthetic(tmp_path)
    q3 = [r for r in rows if r["question_type"] == "q3"]
    assert valid_labels(q3[0]) == ["a", "b", "c"]
    train = [dict(r, answer="d") for r in q3]   # a label the test questions do not offer
    predictions = majority(train, q3)
    assert set(predictions.values()) <= {"a", "b", "c"}


def test_prepare_reads_hf_metadata(tmp_path):
    rows = _synthetic(tmp_path, concepts=1, layouts=2)
    (tmp_path / "synthetic_image").mkdir(exist_ok=True)
    for image in {r["image"] for r in rows}:
        Image.new("RGB", (8, 8)).save(image)
    sample = prepare(tmp_path / "metadata.jsonl", tmp_path, "synthetic", 2, 42)
    assert len({r["image_id"] for r in sample}) == 2
    assert all(len([r for r in sample if r["image_id"] == i]) == 3 for i in {r["image_id"] for r in sample})


def test_spotlight_loss_prefers_attention_on_target():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from spotlightvqa.model import spotlight_loss
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    focused = torch.tensor([[0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3], [0.25] * 4])
    spread = torch.full((2, 4), 0.25)
    assert spotlight_loss(focused, target) < spotlight_loss(spread, target)
    assert spotlight_loss(spread, torch.zeros(2, 4)) == 0


def test_spotlight_box_parsing_and_stem():
    pytest.importorskip("torch")
    from spotlightvqa.vlm import parse_box, question_stem
    # Qwen2.5-VL answers in pixels of the resized input (here 500x500 for a 1000x1000 image).
    text = '```json\n[{"bbox_2d": [10, 20, 260, 270], "label": "subfigure"}]\n```'
    assert parse_box(text, (500, 500), (1000, 1000)) == [20, 40, 520, 540]
    assert parse_box(text, (500, 500), (1000, 1000), fmt="rel1000") == [10, 20, 260, 270]
    assert parse_box("no box here", (500, 500), (1000, 1000)) is None
    assert parse_box('{"bbox_2d": [0, 0, 500, 500]}', (500, 500), (1000, 1000)) is None  # whole image
    assert question_stem(Q3) == "Which subfigure has a dog?"


def test_paired_statistics():
    from spotlightvqa.stats import holm, mcnemar_exact
    p, only_a, only_b = mcnemar_exact([True] * 10 + [False] * 10, [True] * 10 + [True] * 10)
    assert (only_a, only_b) == (0, 10) and p == pytest.approx(2 / 2 ** 10)
    assert mcnemar_exact([True, False], [True, False])[0] == 1.0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
