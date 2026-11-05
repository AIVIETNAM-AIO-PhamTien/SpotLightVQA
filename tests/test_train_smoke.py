"""End-to-end smoke test of the trainer on a fake dataset with tiny random encoders (CPU)."""
import json

import pytest
from PIL import Image

Q3 = "Which subfigure has obj?\na) top\nb) bottom\n(please select one)"


def _fake_root(root):
    syn, meta = root / "synthetic", {"VQA": [], "magicbrush": []}
    (syn / "test/synthetic_image").mkdir(parents=True)
    records = []
    for source in ("VQA", "magicbrush"):
        for idx in range(5):
            gallery = f"{idx}_2_0.png"
            meta[source].append({"composited_gallery": gallery, "qa_target_subimage_idx": idx % 2,
                                 "layout": {"layout_size": [1000, 1000],
                                            "pos": [[0, 0, 500, 500], [500, 500, 1000, 1000]]},
                                 "sub_images": [{"image_paths": f"{source}{idx}a"}, {"image_paths": f"{source}{idx}b"}]})
            for variant, style in (("", "original"), ("enlarged_images", "enlarged")):
                name = f"synthetic_image/{source}_{variant}_{idx}_2_0.png"
                Image.new("RGB", (64, 64), (40 * idx, 0, 0)).save(syn / "test" / name)
                records.append({"file_name": name, "synthetic_style": style, "question_1": f"Any obj{idx}?",
                                "question_2": f"Is {source} obj at top?", "question_3": Q3, "answer_1": "Yes",
                                "answer_2": "Yes" if idx % 2 else "No", "answer_3": "a" if idx % 2 else "b"})
    (syn / "test/metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (syn / "test/meta_data_VQAv2.json").write_text(json.dumps(meta["VQA"]))
    (syn / "test/meta_data_MegicBrush.json").write_text(json.dumps(meta["magicbrush"]))
    real = root / "real-world/test/real-world_image"
    real.mkdir(parents=True)
    rows = []
    for i in range(3):
        Image.new("RGB", (80, 50), "blue").save(real / f"{i}.jpg")
        rows.append({"file_name": f"real-world_image/{i}.jpg", "question_1": "Any dog?", "question_2": "Dog left?",
                     "question_3": Q3, "answer_1": "No", "answer_2": "two", "answer_3": "b)"})
    (root / "real-world/test/metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("extra", [["--split", "concept", "--spotlight-weight", "0.5", "--augment"],
                                   ["--split", "legacy", "--blind", "--no-mask-answers"]])
def test_train_end_to_end(tmp_path, extra):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    from spotlightvqa.train import main
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("roberta-base")
    except Exception as error:  # offline machine
        pytest.skip(f"roberta-base tokenizer unavailable: {error}")
    _fake_root(tmp_path / "raw")
    out = tmp_path / "run"
    metrics = main(["--data-root", str(tmp_path / "raw"), "--tiny", "--epochs", "1", "--batch-size", "4",
                    "--image-size", "32", "--workers", "0", "--cache-dir", str(tmp_path / "cache"),
                    "--output", str(out), *extra])
    assert {"val", "test", "real_world"} <= set(metrics)
    assert metrics["real_world"]["overall"]["total"] == 6          # open Q2 answers excluded
    assert metrics["real_world"]["consistency"]["total"] == 0      # no real image has 3 closed QA
    for name in ("history.json", "metrics.json", "predictions_test.jsonl", "best.pt"):
        assert (out / name).is_file()
