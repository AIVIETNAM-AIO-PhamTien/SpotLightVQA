"""Load the official MultipanelVQA HuggingFace release into one QA row per question.

HF layout (``test/metadata.jsonl``): one record per image with ``question_1..3`` and
``answer_1..3``; the synthetic subset also has ``synthetic_style``. Synthetic file names
encode ``<source>_<variant>_<concept idx>_<n panels>_<layout id>``; the five styles of the
same (source, idx, n panels, layout) share the exact same QA ("core"), and all layouts of one
(source, idx) reuse the same handful of sub-images and the same question ("concept").
"""
import json
import re
from pathlib import Path

QTYPES = ("q1", "q2", "q3")
QTYPE_NAMES = {"q1": "Q1 tổng quan (yes/no)", "q2": "Q2 vị trí→vật thể (yes/no)",
               "q3": "Q3 vật thể→vị trí (chọn ô)"}
YES_NO = ("yes", "no")
OPTION_LETTERS = ("a", "b", "c", "d", "e")
LABELS = YES_NO + OPTION_LETTERS          # closed answer vocabulary of the classifier
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

_SYN_NAME = re.compile(r"^(magicbrush|VQA)_(.*?)_?(\d+)_(\d+)_(\d+)$")
_OPTION = re.compile(r"^\s*([a-e])\)\s*(.+?)\s*$", re.IGNORECASE)


def parse_options(question):
    """Return ``{letter: option text}`` for a multiple-choice question."""
    options = {}
    for line in question.splitlines():
        match = _OPTION.match(line)
        if match:
            options[match.group(1).lower()] = match.group(2).strip().lower()
    return options


def normalize_closed(answer, qtype, question):
    """Map a raw answer to the closed vocabulary, or ``None`` if it is open-ended.

    Yes/no: leading "yes"/"no" word ("Yes.", "yes, they are ..."); anything else is open.
    Choice: a letter that exists in the options ("b)", "c."), or the exact option text
    ("bottom left" → "c"). Nothing is guessed beyond that.
    """
    text = answer.strip().lower()
    if qtype in ("q1", "q2"):
        word = re.split(r"[\s,.)!]+", text, maxsplit=1)[0] if text else ""
        return word if word in YES_NO else None
    options = parse_options(question)
    letter = text.rstrip(").").strip()
    if letter in options:
        return letter
    for key, value in options.items():
        if text.rstrip(".") == value.rstrip("."):
            return key
    return None


def parse_synthetic_name(file_name):
    stem = Path(file_name).stem
    match = _SYN_NAME.match(stem)
    if not match:
        raise ValueError(f"Unexpected synthetic file name: {file_name}")
    source, variant, idx, panels, layout = match.groups()
    return {"source": source, "variant": variant or "original", "concept_idx": int(idx),
            "n_panels": int(panels), "layout_id": int(layout)}


def load_subset(metadata, image_root, subset):
    """Read one HF ``metadata.jsonl`` and return a list of QA rows (3 per image)."""
    metadata, image_root = Path(metadata), Path(image_root)
    rows = []
    with metadata.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError(f"No records in {metadata}")
    for record in records:
        file_name = record["file_name"]
        info = {"style": record.get("synthetic_style", "real-world")}
        if subset == "synthetic":
            name = parse_synthetic_name(file_name)
            info.update(name)
            info["concept"] = f"{name['source']}:{name['concept_idx']}"
            info["core"] = f"{info['concept']}:{name['n_panels']}:{name['layout_id']}"
        else:
            info.update(source="real-world", concept=f"real:{file_name}", core=f"real:{file_name}",
                        n_panels=None, layout_id=None)
        for k, qtype in enumerate(QTYPES, start=1):
            question, raw = record[f"question_{k}"], record[f"answer_{k}"]
            closed = normalize_closed(raw, qtype, question)
            n_options = len(parse_options(question)) if qtype == "q3" else 2
            rows.append({"question_id": f"{subset}:{file_name}:{qtype}", "image_id": f"{subset}:{file_name}",
                         "image": str((image_root / file_name).resolve()), "subset": subset,
                         "question_type": qtype, "question": question, "answer_raw": raw,
                         "answer": closed if closed is not None else raw.strip().lower(),
                         "closed": closed is not None, "n_options": n_options, **info})
    return rows


def valid_labels(row):
    """Labels a classifier may output for this row (answer-type mask)."""
    if row["question_type"] == "q3":
        options = parse_options(row["question"])
        return [letter for letter in OPTION_LETTERS if letter in options]
    return list(YES_NO)


def load_metadata_json(*paths):
    """Index the synthetic generation metadata (sub-images, layout boxes) by HF file stem.

    ``meta_data_VQAv2.json`` / ``meta_data_MegicBrush.json`` name galleries ``<idx>_<n>_<layout>.png``.
    """
    index = {}
    for path in paths:
        source = "VQA" if "VQAv2" in Path(path).name else "magicbrush"
        for item in json.loads(Path(path).read_text(encoding="utf-8")):
            index[f"{source}:{Path(item['composited_gallery']).stem}"] = item
    return index


def target_box(row, meta_index, image_size, variants=("original",)):
    """Ground-truth box (pixels of the HF image) of the target panel, or ``None``.

    Only known for synthetic rows of the given variants: the generation metadata stores the
    layout of the original composition; other variants (enlarged, background, ...) may move
    or resize panels, so they are excluded by default.
    """
    if row["subset"] != "synthetic" or row.get("variant") not in variants:
        return None
    item = meta_index[f"{row['source']}:{row['concept_idx']}_{row['n_panels']}_{row['layout_id']}"]
    (lw, lh), (w, h) = item["layout"]["layout_size"], image_size
    x1, y1, x2, y2 = item["layout"]["pos"][item["qa_target_subimage_idx"]]
    return [round(x1 * w / lw), round(y1 * h / lh), round(x2 * w / lw), round(y2 * h / lh)]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def sub_images_of(row, meta_index):
    if row["subset"] != "synthetic":
        return set()
    key = f"{row['source']}:{row['concept_idx']}_{row['n_panels']}_{row['layout_id']}"
    return {f"{row['source']}:{s['image_paths']}" for s in meta_index[key]["sub_images"]}
