import json
import random
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def prepare(source, image_root, subset, limit, seed):
    """Sample complete images, keeping all associated questions together."""
    source = Path(source)
    items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a nonempty official JSON list of multipanel records")
    if limit < 1:
        raise ValueError("Image limit must be positive")
    # Sort first so the same file contents in another order give the same sample.
    items = sorted(items, key=lambda r: r["composited_gallery"])
    if len({r["composited_gallery"] for r in items}) != len(items):
        raise ValueError("Duplicate image records")
    sampled = random.Random(seed).sample(items, min(limit, len(items)))
    rows = []
    for item in sampled:
        image = (Path(image_root) / item["composited_gallery"].lstrip("/\\")).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        for index, qa in enumerate(item["qa"]):
            if not isinstance(qa["question"], str) or not isinstance(qa["answer"], str):
                raise ValueError("Questions and answers must be strings")
            rows.append({"question_id": f"{subset}:{item['composited_gallery']}:{index}",
                         "image": str(image), "question": qa["question"], "answer": qa["answer"],
                         "subset": subset, "question_type": f"qa_{index}"})
    if not rows:
        raise ValueError("No questions found")
    return rows
