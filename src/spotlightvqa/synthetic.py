"""Run upstream notebook's compositor with offline, visibly labeled fixtures."""
import ast
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import write_jsonl

UPSTREAM_COMMIT = "d2588ce5dc363b2ced6ff0f3497e775141865ed9"


def generate(upstream, output, seed=42):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, found {revision}")
    notebook = upstream / "synthetic_data_generation/fill_subfigures_into_layouts.ipynb"
    cells = ["".join(c["source"]) for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    # Extract only literal demo data; avoid notebook display/download cells.
    demo_branch = next(n for n in ast.parse(cells[1]).body if isinstance(n, ast.If)).orelse
    demo = {}
    exec(compile(ast.Module(body=demo_branch, type_ignores=[]), str(notebook), "exec"), demo)
    layout = demo["layout_data"]["8"][0]
    image_data = demo["image_data"]
    output.mkdir(parents=True, exist_ok=True)
    fixtures = output / "fixtures"
    fixtures.mkdir(exist_ok=True)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
    examples = image_data["8"].pop("What is the animal?")
    for index, (example, color) in enumerate(zip(examples, colors)):
        image = Image.new("RGB", (240, 180), color)
        ImageDraw.Draw(image).text((12, 12), f"FIXTURE {index}: {color}", fill="black")
        filename = f"fixture_{index}.png"
        image.save(fixtures / filename)
        example.update(question="What color is the panel?", answer=color,
                       target_img={"path": "fixtures/", "name": filename})
    image_data["8"]["What color is the panel?"] = examples
    layouts = {}
    for count in (2, 4, 8):
        layouts[str(count)] = [{**layout, "layout_id": f"{count}_0", "pos": layout["pos"][:count],
                               "positional_descriptor": [f"{count}_0_{i}" for i in range(count)]}]
    env = {"demo_mode": False, "layout_data": layouts, "image_data": image_data,
           "unique_ids": [["8", "What color is the panel?"]],
           "base_dataset_image_path": str(output), "image_path": {}}
    original_random, original_numpy = random.getstate(), np.random.get_state()
    previous = Path.cwd()
    try:
        random.seed(seed)
        np.random.seed(seed)
        os.chdir(output)
        compositor = next(c for c in cells if "test_set = []" in c)
        exec(compile(compositor, str(notebook), "exec"), env)
    finally:
        os.chdir(previous)
        random.setstate(original_random)
        np.random.set_state(original_numpy)
    records = []
    for item in env["test_set"]:
        # Upstream uses a stale image_path for sub_images metadata. Repair it here.
        for index, panel in enumerate(item["sub_images"]):
            panel["image_paths"] = examples[index]["target_img"]
        x1, y1, x2, y2 = item["layout"]["pos"][0]
        records.append({"question_id": f"fixture:{item['layout']['layout_id']}:0",
                        "image": str(output / item["composited_gallery"]),
                        "question": f"What color is the panel centered at pixel ({(x1+x2)//2}, {(y1+y2)//2})?",
                        "answer": examples[0]["answer"], "subset": "smoke_fixture",
                        "question_type": "position", "num_panels": len(item["sub_images"])})
    write_jsonl(output / "questions.jsonl", records)
    write_jsonl(output / "reference_predictions.jsonl", [
        {"question_id": r["question_id"], "prediction": r["answer"]} for r in records])
    (output / "upstream_metadata.json").write_text(json.dumps(env["test_set"], indent=2), encoding="utf-8")
    provenance = {"upstream_commit": revision, "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
                  "seed": seed, "images": len(records), "source": "offline_color_fixtures",
                  "benchmark_data": False, "adaptations": ["2/4/8-panel slices of demo layout",
                  "local fixture images instead of COCO", "repair stale sub_images.image_paths metadata"]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
