"""Train / evaluate CrossModalMultipanelVQA on MultipanelVQA (GPU, e.g. Colab).

Example (Colab):
    python -m spotlightvqa.train --data-root data/raw --split concept --fold 0 --image-size 448 \
        --spotlight-weight 0.2 --augment --output outputs/train/concept_f0_full

Every run writes ``config.json``, ``history.json``, ``metrics.json`` (val/test/real-world,
from the best checkpoint by validation consistency) and ``predictions_<part>.jsonl``.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .audit import load_all
from .metrics import vqa_scores
from .model import CrossModalMultipanelVQA, spotlight_loss
from .mpvqa import LABEL2ID, LABELS, valid_labels
from .splits import concept_folds, legacy_split

MEAN = STD = (0.5, 0.5, 0.5)  # google/vit-base-patch16-224 preprocessing


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageCache:
    """Resize every image once to ``size``x``size`` RGB uint8 and keep it in a memory-mapped file."""

    def __init__(self, paths, size, cache_dir):
        self.index = {p: i for i, p in enumerate(sorted(set(paths)))}
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = f"images_{size}_{len(self.index)}"
        file, done = cache_dir / f"{stem}.u8", cache_dir / f"{stem}.json"
        shape = (len(self.index), size, size, 3)
        if not done.exists() or json.loads(done.read_text()) != sorted(self.index):
            array = np.lib.format.open_memmap(file, mode="w+", dtype=np.uint8, shape=shape)
            for path, i in self.index.items():
                with Image.open(path) as image:
                    array[i] = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC))
            array.flush()
            del array
            done.write_text(json.dumps(sorted(self.index)))
        self.array = np.load(file, mmap_mode="r")

    def __getitem__(self, path):
        return self.array[self.index[path]]


def target_patch_mask(row, meta, grid, variants):
    """Patches covered by the target panel box (synthetic Q2/Q3 of the given variants), else zeros."""
    mask = np.zeros(grid * grid, dtype=np.float32)
    if row["subset"] != "synthetic" or row["question_type"] == "q1" or row["variant"] not in variants:
        return mask
    item = meta[f"{row['source']}:{row['concept_idx']}_{row['n_panels']}_{row['layout_id']}"]
    width, height = item["layout"]["layout_size"]
    x1, y1, x2, y2 = item["layout"]["pos"][item["qa_target_subimage_idx"]]
    centers = (np.arange(grid) + 0.5) / grid
    cols = (centers >= x1 / width) & (centers <= x2 / width)
    rows = (centers >= y1 / height) & (centers <= y2 / height)
    return (rows[:, None] & cols[None, :]).astype(np.float32).reshape(-1)


class MPVQADataset(Dataset):
    def __init__(self, rows, cache, tokenizer, meta, grid, max_len=128, augment=False,
                 spotlight_variants=("original",)):
        self.rows = [r for r in rows if r["closed"]]
        self.cache, self.meta, self.grid, self.augment = cache, meta, grid, augment
        self.spotlight_variants = set(spotlight_variants)
        if not self.rows:
            raise ValueError("Dataset part has no closed-answer questions")
        self.tokens = tokenizer([r["question"] for r in self.rows], truncation=True,
                                max_length=max_len, padding="max_length", return_tensors="pt")
        self.jitter = None
        if augment:
            from torchvision import transforms as T
            # Photometric only: flips/crops would change which panel is "left"/"top".
            self.jitter = T.Compose([T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.8),
                                     T.RandomGrayscale(p=0.1),
                                     T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = torch.from_numpy(np.array(self.cache[row["image"]])).permute(2, 0, 1).float() / 255
        if self.jitter is not None:
            image = self.jitter(image)
        image = (image - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        label_mask = torch.zeros(len(LABELS), dtype=torch.bool)
        label_mask[[LABEL2ID[label] for label in valid_labels(row)]] = True
        return {"pixel_values": image, "input_ids": self.tokens["input_ids"][i],
                "attention_mask": self.tokens["attention_mask"][i], "label_mask": label_mask,
                "label": LABEL2ID[row["answer"]],
                "target": torch.from_numpy(target_patch_mask(row, self.meta, self.grid,
                                                             self.spotlight_variants)),
                "index": i}


def build_splits(args, syn):
    if args.split == "legacy":
        return legacy_split(syn, seed=args.seed)
    fold = concept_folds(syn, n_folds=args.n_folds, seed=args.seed)[args.fold]
    return {k: fold[k] for k in ("train", "val", "test")}


@torch.no_grad()
def predict(model, loader, device, amp_dtype, mask_answers):
    model.eval()
    out = {}
    for batch in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(batch["pixel_values"].to(device), batch["input_ids"].to(device),
                              batch["attention_mask"].to(device),
                              batch["label_mask"].to(device) if mask_answers else None)
        for i, k in zip(batch["index"].tolist(), logits.argmax(-1).tolist()):
            out[loader.dataset.rows[i]["question_id"]] = LABELS[k]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", choices=["legacy", "concept"], default="concept")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1, help="Effective batch = batch-size x grad-accum")
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--no-mask-answers", action="store_true", help="Report-1 behaviour: argmax over all labels")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--spotlight-weight", type=float, default=0.0)
    p.add_argument("--spotlight-variants", default="original")
    p.add_argument("--blind", action="store_true", help="Zero the image tokens (question-only ablation)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-images", type=int, default=0, help="Use only N images per part (smoke test)")
    p.add_argument("--tiny", action="store_true", help="Random tiny encoders (smoke test, no download)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if args.image_size % 16:
        raise ValueError("--image-size must be a multiple of 16")
    seed_everything(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    syn, real, meta = load_all(args.data_root)
    parts = build_splits(args, syn)
    parts["real_world"] = real
    if args.limit_images:
        for name, rows in parts.items():
            keep = set(sorted({r["image_id"] for r in rows})[:args.limit_images])
            parts[name] = [r for r in rows if r["image_id"] in keep]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    cache = ImageCache([r["image"] for rows in parts.values() for r in rows], args.image_size, args.cache_dir)
    grid = args.image_size // 16
    variants = tuple(v for v in args.spotlight_variants.split(",") if v)
    empty = [k for k, v in parts.items() if not any(r["closed"] for r in v)]
    if {"train", "val"} & set(empty):
        raise ValueError(f"Empty split part(s): {empty}")
    parts = {k: v for k, v in parts.items() if k not in empty}
    sets = {k: MPVQADataset(v, cache, tokenizer, meta, grid, augment=args.augment and k == "train",
                            spotlight_variants=variants) for k, v in parts.items()}
    loaders = {k: DataLoader(d, batch_size=args.batch_size, shuffle=k == "train", num_workers=args.workers,
                             pin_memory=device.type == "cuda", drop_last=False) for k, d in sets.items()}

    model = CrossModalMultipanelVQA(len(LABELS), image_size=args.image_size, blind=args.blind,
                                    tiny=args.tiny).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": args.lr_backbone},
        {"params": model.head_parameters(), "lr": args.lr_head}], weight_decay=args.weight_decay)
    accum = max(1, args.grad_accum)
    per_epoch = math.ceil(len(loaders["train"]) / accum)
    steps = args.epochs * per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mask_answers = not args.no_mask_answers

    history, best = [], (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, totals = time.time(), {"loss": 0.0, "ce": 0.0, "spot": 0.0, "n": 0}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loaders["train"], 1):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, attn = model(batch["pixel_values"].to(device, non_blocking=True),
                                     batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                     batch["label_mask"].to(device) if mask_answers else None)
                ce = criterion(logits, batch["label"].to(device))
                spot = spotlight_loss(attn, batch["target"].to(device)) if args.spotlight_weight else logits.new_zeros(())
                loss = ce + args.spotlight_weight * spot
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            n = batch["label"].size(0)
            for key, value in (("loss", loss), ("ce", ce), ("spot", spot)):
                totals[key] += float(value.detach()) * n
            totals["n"] += n
        val = vqa_scores(sets["val"].rows, predict(model, loaders["val"], device, amp_dtype, mask_answers))
        record = {"epoch": epoch, "seconds": round(time.time() - started, 1),
                  **{k: totals[k] / totals["n"] for k in ("loss", "ce", "spot")},
                  "val_overall": val["overall"]["accuracy"], "val_consistency": val["consistency"]["accuracy"],
                  **{f"val_{q}": val[f"type:{q}"]["accuracy"] for q in ("q1", "q2", "q3") if f"type:{q}" in val}}
        history.append(record)
        print(json.dumps(record))
        key = (val["consistency"]["accuracy"] or 0.0, val["overall"]["accuracy"])
        if key > best:
            best = key
            torch.save(model.state_dict(), out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    metrics = {"best_epoch": max(history, key=lambda r: ((r["val_consistency"] or 0.0), r["val_overall"]))["epoch"]}
    for name, loader in loaders.items():
        if name == "train":
            continue
        predictions = predict(model, loader, device, amp_dtype, mask_answers)
        metrics[name] = vqa_scores(sets[name].rows, predictions)
        with (out / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as stream:
            for qid, label in predictions.items():
                stream.write(json.dumps({"question_id": qid, "prediction": label}) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: {m: v[m]["accuracy"] for m in ("overall", "consistency")}
                      for k, v in metrics.items() if isinstance(v, dict)}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
