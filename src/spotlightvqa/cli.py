import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .data import prepare, read_jsonl, write_jsonl
from .metrics import evaluate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(args):
    result = evaluate(read_jsonl(args.data), read_jsonl(args.predictions))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(args.output) / run_id
    output.mkdir(parents=True)
    card = {"run_id": run_id, "run_type": "evaluation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model, "seed": args.seed, "protocol": args.protocol,
            "dataset": str(Path(args.data).resolve()), "dataset_sha256": sha256(args.data),
            "predictions_sha256": sha256(args.predictions),
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()), "python": platform.python_version(),
            "platform": platform.platform(), "command": sys.argv,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "metric": result["metric"], "scores": result["scores"], "notes": args.notes,
            "artifacts": str(output.resolve())}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "experiment.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    write_jsonl(output / "predictions.jsonl", read_jsonl(args.predictions))
    print(json.dumps({"run_id": run_id, "scores": result["scores"], "artifacts": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="MultipanelVQA week-one pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("synthetic", help="Run upstream compositor with offline fixtures")
    gen.add_argument("--upstream", default="third_party/MultipanelVQA")
    gen.add_argument("--output", default="data/synthetic/smoke")
    gen.add_argument("--seed", type=int, default=42)
    prep = sub.add_parser("prepare", help="Convert and sample one official subset by image")
    prep.add_argument("--source", required=True)
    prep.add_argument("--image-root", required=True)
    prep.add_argument("--subset", choices=["real-world", "synthetic"], required=True)
    prep.add_argument("--images", type=int, default=10)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--output", required=True)
    infer = sub.add_parser("infer", help="Run a supplied module:function(image_file, prompt)")
    infer.add_argument("--data", required=True)
    infer.add_argument("--predictor", required=True)
    infer.add_argument("--output", required=True)
    ev = sub.add_parser("evaluate", help="Local exact match with evaluation artifacts")
    ev.add_argument("--data", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--model", required=True, help="Use reference-oracle-smoke for fixture answers")
    ev.add_argument("--protocol", required=True, help="Prompt, decoding settings, model revision and hardware")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--notes", default="")
    ev.add_argument("--output", default="outputs")
    args = parser.parse_args()
    if args.command == "synthetic":
        from .synthetic import generate
        print(json.dumps(generate(args.upstream, args.output, args.seed), indent=2))
    elif args.command == "prepare":
        write_jsonl(args.output, prepare(args.source, args.image_root, args.subset, args.images, args.seed))
    elif args.command == "infer":
        module, function = args.predictor.split(":", 1)
        call_model = getattr(importlib.import_module(module), function)
        predictions = []
        for row in read_jsonl(args.data):
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            prediction = call_model(row["image"], row["question"])
            if not isinstance(prediction, str):
                raise TypeError("Predictor must return a string")
            predictions.append({"question_id": row["question_id"], "prediction": prediction})
        write_jsonl(args.output, predictions)
    else:
        score(args)


if __name__ == "__main__":
    main()
