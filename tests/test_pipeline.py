import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from spotlightvqa.data import prepare, read_jsonl
from spotlightvqa.synthetic import generate
from spotlightvqa.data import write_jsonl
from spotlightvqa.cli import main


def test_prepare_keeps_all_questions_and_seed(tmp_path):
    items = []
    for i in range(4):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{i}.png")
        items.append({"composited_gallery": f"{i}.png", "qa": [
            {"question": f"q{j}", "answer": "yes"} for j in range(3)]})
    path = tmp_path / "official.json"
    path.write_text(json.dumps(items))
    first = prepare(path, tmp_path, "synthetic", 2, 42)
    assert len(first) == 6
    assert len({r["image"] for r in first}) == 2
    assert first == prepare(path, tmp_path, "synthetic", 2, 42)
    Path(first[0]["image"]).unlink()
    with pytest.raises(FileNotFoundError):
        prepare(path, tmp_path, "synthetic", 2, 42)


@pytest.mark.skipif(not Path("third_party/MultipanelVQA/.git").exists(), reason="Upstream checkout not installed")
def test_upstream_smoke(tmp_path):
    provenance = generate("third_party/MultipanelVQA", tmp_path)
    assert provenance["images"] == 3
    records = read_jsonl(tmp_path / "questions.jsonl")
    assert [r["num_panels"] for r in records] == [2, 4, 8]
    for record in records:
        with Image.open(record["image"]) as image:
            assert image.size == (1000, 1000)
            assert image.getpixel((150, 550)) == (255, 0, 0)
    metadata = json.loads((tmp_path / "upstream_metadata.json").read_text())
    assert metadata[0]["sub_images"][0]["image_paths"]["name"] == "fixture_0.png"


def test_inference_evaluation_and_artifacts(tmp_path, monkeypatch):
    image = tmp_path / "red.png"
    Image.new("RGB", (8, 8), "red").save(image)
    data = tmp_path / "data.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(data, [{"question_id": "fixture:0", "image": str(image), "question": "Color?",
                        "answer": "red", "subset": "smoke_fixture", "question_type": "color"}])
    adapter = types.ModuleType("fixture_predictor")

    def predict(image_file, prompt):
        assert prompt == "Color?"
        with Image.open(image_file) as actual:
            assert actual.getpixel((0, 0)) == (255, 0, 0)
        return "Red."

    adapter.call_model = predict
    monkeypatch.setitem(sys.modules, "fixture_predictor", adapter)
    monkeypatch.setattr(sys, "argv", ["spotlightvqa", "infer", "--data", str(data),
                        "--predictor", "fixture_predictor:call_model", "--output", str(predictions)])
    main()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs"
    monkeypatch.setattr(sys, "argv", ["spotlightvqa", "evaluate", "--data", str(data),
                        "--predictions", str(predictions), "--model", "fixture-adapter", "--protocol",
                        "test-only", "--output", str(output)])
    main()
    main()
    runs = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(output.glob("*/experiment.json"))]
    assert len(runs) == 2
    assert runs[0]["run_id"] != runs[1]["run_id"]
    assert not (tmp_path / "experiments").exists()
    card = runs[0]
    assert card["run_type"] == "evaluation"
    assert card["git_commit"] != "HEAD"
    assert card["scores"]["overall"]["accuracy"] == 1.0
    assert len(card["dataset_sha256"]) == 64
    artifacts = output / card["run_id"]
    assert Path(card["artifacts"]) == artifacts
    assert (artifacts / "experiment.json").is_file()
    assert read_jsonl(artifacts / "predictions.jsonl")[0]["prediction"] == "Red."
