"""Zero-shot VLM on MultipanelVQA: direct answering vs. Spotlight (ground-then-zoom).

Answers are scored, not generated: the next-token log-probabilities of the valid answers
("Yes"/"No", or the option letters the question lists) are compared, so results are
deterministic and need no GPT-4 judge.

``--method direct``     the full multipanel image + question (paper setting).
``--method spotlight``  training-free decomposition that uses the VLM's own grounding:
  * Q2 (position -> object): the VLM boxes the subfigure the question refers to; the crop is
    zoomed and the question is answered on the crop.
  * Q3 (object -> position): every listed option is grounded ("the subfigure at <option>"),
    cropped, and judged on its own: s(o) = log P(Yes | crop_o) - log P(No | crop_o). The answer
    is argmax_o s(o). This turns one hard choice over a cluttered image into n easy
    single-panel checks.
  * Q1 (whole figure) keeps the direct answer.
  A failed or implausible box falls back to the full image, so the method never has less
  information than the direct baseline. Boxes and direct answers are stored for analysis.

Every spotlight record also carries, when the synthetic metadata knows it (variant
``original``), the ground-truth target box ``gt_box`` and, for Q2, the ``oracle`` answer on the
ground-truth crop. Direct / Spotlight / Oracle on the same questions separate the effect of
zooming from the quality of the VLM's own grounding (cf. the neighbour-removal ablation of
the MultipanelVQA paper, here without any human-provided box).

    python -m spotlightvqa.vlm --method spotlight --output outputs/vlm/qwen25vl3b_spotlight
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

from .audit import load_all
from .metrics import vqa_scores
from .mpvqa import parse_options, target_box, valid_labels

SUFFIX = {"yn": "\nAnswer with a single word: Yes or No.",
          "mc": "\nAnswer with the letter of the correct option only."}
_PLEASE = re.compile(r"\(\s*please[^)]*\)", re.IGNORECASE)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def question_stem(question):
    """Question text without the option lines and the '(please ...)' instruction."""
    lines = [line for line in question.splitlines() if not re.match(r"^\s*[a-e]\)", line, re.IGNORECASE)]
    return _PLEASE.sub("", " ".join(lines)).strip()


def parse_box(text, input_size, image_size, fmt="abs"):
    """First ``[x1, y1, x2, y2]`` in the model output, mapped to original-image pixels.

    ``abs``: coordinates on the resized model input (Qwen2.5-VL); ``rel1000``: 0-1000 scale.
    Returns ``None`` for missing, degenerate or implausible (<0.5% or >90% of the image) boxes.
    """
    after = text.split("bbox_2d", 1)[-1]
    numbers = [float(n) for n in _NUM.findall(after)[:4]]
    if len(numbers) < 4:
        return None
    (in_w, in_h), (width, height) = input_size, image_size
    sx, sy = (width / 1000, height / 1000) if fmt == "rel1000" else (width / in_w, height / in_h)
    x1, y1, x2, y2 = numbers[0] * sx, numbers[1] * sy, numbers[2] * sx, numbers[3] * sy
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    area = (x2 - x1) * (y2 - y1) / (width * height)
    if not 0.005 <= area <= 0.9:
        return None
    return [round(x1), round(y1), round(x2), round(y2)]


def crop(image, box, margin=0.04):
    x1, y1, x2, y2 = box
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    return image.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                       min(image.width, int(x2 + mx)), min(image.height, int(y2 + my))))


def candidate_ids(tokenizer, label):
    """First token ids of the spellings a model may use for one label."""
    forms = [label, label.capitalize(), label.upper(), " " + label, " " + label.capitalize()]
    if len(label) == 1:
        forms += [label + ")", label.upper() + ")"]
    ids = set()
    for form in forms:
        tokens = tokenizer.encode(form, add_special_tokens=False)
        if tokens:
            ids.add(tokens[0])
    return sorted(ids)


class VLM:
    def __init__(self, name, min_pixels, max_pixels, bbox_format, dtype="auto"):
        from transformers import AutoProcessor
        self.name, self.bbox_format, self._ids = name, bbox_format, {}
        self.processor = AutoProcessor.from_pretrained(name, min_pixels=min_pixels, max_pixels=max_pixels)
        if dtype == "auto":
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = "bf16" if bf16 else "fp16"
        self._load(dtype)

    def _load(self, dtype):
        from transformers import AutoModelForImageTextToText
        self.dtype = dtype
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.name, dtype=torch_dtype, device_map="auto").eval()
        print(f"loaded {self.name} in {dtype}", flush=True)

    def _fallback_fp32(self):
        """fp16 (GPUs without bf16, e.g. T4) can overflow; the first non-finite logit switches to fp32."""
        if self.dtype == "fp32":
            raise RuntimeError("Non-finite logits even in fp32")
        print("non-finite logits in", self.dtype, "-> reloading in fp32", flush=True)
        del self.model
        torch.cuda.empty_cache()
        self._load("fp32")

    def _inputs(self, image, prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)

    @torch.no_grad()
    def score(self, image, prompt, labels):
        logits = self.model(**self._inputs(image, prompt)).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            self._fallback_fp32()
            logits = self.model(**self._inputs(image, prompt)).logits[0, -1].float()
        logits = logits.log_softmax(-1)
        out = {}
        for label in labels:
            if label not in self._ids:
                self._ids[label] = candidate_ids(self.processor.tokenizer, label)
            out[label] = max(logits[i].item() for i in self._ids[label])
        return out

    @torch.no_grad()
    def ground(self, image, description):
        prompt = f"Locate {description} in this multi-panel image. Output its bbox coordinates in JSON format."
        inputs = self._inputs(image, prompt)
        generated = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        text = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if "image_grid_thw" in inputs:  # Qwen-VL: patches of 14 px on the resized input
            _, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
            input_size = (grid_w * 14, grid_h * 14)
        else:
            input_size = image.size
        return parse_box(text, input_size, image.size, self.bbox_format), text


def answer_direct(vlm, row, image):
    suffix = SUFFIX["mc" if row["question_type"] == "q3" else "yn"]
    scores = vlm.score(image, row["question"] + suffix, valid_labels(row))
    return max(scores, key=scores.get), scores


def answer_spotlight(vlm, row, image, gt_box=None):
    direct, direct_scores = answer_direct(vlm, row, image)
    record = {"direct": direct, "direct_scores": direct_scores, "gt_box": gt_box}
    if row["question_type"] == "q1":
        return direct, record
    stem = question_stem(row["question"])
    if row["question_type"] == "q2":
        box, raw = vlm.ground(image, f'the subfigure that this question asks about: "{stem}"')
        record.update(box=box, raw=raw)
        prompt = ("This image is a zoomed-in crop of the subfigure the question refers to.\n"
                  + row["question"] + SUFFIX["yn"])
        if gt_box is not None:
            oracle = vlm.score(crop(image, gt_box), prompt, ["yes", "no"])
            record["oracle"] = max(oracle, key=oracle.get)
        if box is None:
            return direct, record
        scores = vlm.score(crop(image, box), prompt, ["yes", "no"])
        record["scores"] = scores
        return max(scores, key=scores.get), record
    options = parse_options(row["question"])
    margins, boxes = {}, {}
    for letter in valid_labels(row):
        box, _ = vlm.ground(image, f'the subfigure described as "{options[letter]}"')
        boxes[letter] = box
        if box is None:
            view, where = image, f'Consider only the subfigure described as "{options[letter]}".\n'
        else:
            view, where = crop(image, box), f'This image is a zoomed-in crop of one subfigure ("{options[letter]}").\n'
        prompt = (where + f"Question about the whole multi-panel figure: {stem}\n"
                  "Is this subfigure the correct answer to the question?" + SUFFIX["yn"])
        scores = vlm.score(view, prompt, ["yes", "no"])
        margins[letter] = scores["yes"] - scores["no"]
    record.update(boxes=boxes, margins=margins)
    return max(margins, key=margins.get), record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--method", choices=["direct", "spotlight"], default="direct")
    p.add_argument("--bbox-format", choices=["abs", "rel1000"], default="abs",
                   help="abs for Qwen2.5-VL, rel1000 for Qwen3-VL")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--subsets", nargs="+", default=["synthetic", "real-world"])
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--variants", default="", help="Comma list of synthetic variants to keep, e.g. original")
    p.add_argument("--limit-images", type=int, default=0, help="Random images per subset (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    syn, real, meta = load_all(args.data_root)
    rows = []
    for subset, data in (("synthetic", syn), ("real-world", real)):
        if subset not in args.subsets:
            continue
        data = [r for r in data if r["closed"]]
        if subset == "synthetic" and args.variants:
            data = [r for r in data if r["variant"] in args.variants.split(",")]
        if args.limit_images:
            images = sorted({r["image_id"] for r in data})
            keep = set(random.Random(args.seed).sample(images, min(args.limit_images, len(images))))
            data = [r for r in data if r["image_id"] in keep]
        rows += data

    path = out / "predictions.jsonl"
    done, direct = {}, {}
    if path.exists():  # resume after a Colab disconnect
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["question_id"]] = record["prediction"]
                direct[record["question_id"]] = record.get("direct", record["prediction"])
    vlm = VLM(args.model, args.min_pixels, args.max_pixels, args.bbox_format, args.dtype)
    started = time.time()
    with path.open("a", encoding="utf-8") as stream:
        todo = [r for r in rows if r["question_id"] not in done]
        for n, row in enumerate(todo, 1):
            with Image.open(row["image"]) as image:
                image = image.convert("RGB")
                if args.method == "spotlight":
                    label, extra = answer_spotlight(vlm, row, image, target_box(row, meta, image.size))
                else:
                    label, extra = answer_direct(vlm, row, image)
            done[row["question_id"]] = label
            direct[row["question_id"]] = extra.get("direct", label)
            stream.write(json.dumps({"question_id": row["question_id"], "prediction": label, **(
                extra if args.method == "spotlight" else {"scores": extra})}) + "\n")
            if n % 50 == 0:
                stream.flush()
                print(f"{n}/{len(todo)}  {(time.time() - started) / n:.2f}s/QA", flush=True)

    metrics = {}
    for subset in args.subsets:
        part = [r for r in rows if r["subset"] == subset]
        if part:
            metrics[subset] = vqa_scores(part, done)
            if args.method == "spotlight":  # same run, same images: the paired direct baseline
                metrics[subset + "_direct"] = vqa_scores(part, direct)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: {m: v[m]["accuracy"] for m in ("overall", "type:q1", "type:q2", "type:q3", "consistency")
                          if m in v} for k, v in metrics.items()}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
