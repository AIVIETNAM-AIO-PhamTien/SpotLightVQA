"""Kaggle kernel: run every GPU experiment of progress report 2 (works on 1 or 2 GPUs).

Each GPU is a worker that pulls jobs from one queue (VLM first: it is the longest), so on
2xT4 the VLM and the training runs proceed in parallel. Finished jobs (``metrics.json``
present) are skipped; outputs of a previous session attached as input are copied back first,
so re-running after the 12-hour limit continues where it stopped.

Output: /kaggle/working/outputs (evidence.json, summary.json, audit/, train/, vlm/, logs/).
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = "https://github.com/AIVIETNAM-AIO-PhamTien/SpotLightVQA.git"
BRANCH = "tiendo2"
WORK = Path("/kaggle/working")
CODE = Path("/tmp/SpotLightVQA")   # code, data, caches: not saved as kernel output
OUT = WORK / "outputs"
BUDGET_H = 11.0          # stop starting new jobs after this many hours (Kaggle limit 12 h)
EPOCHS = 5
STARTED = time.time()


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def setup():
    if not CODE.exists():
        sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, str(CODE)])
    sh([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.56", "accelerate>=1.0",
        "huggingface_hub>=0.25", "scikit-learn"])
    os.chdir(CODE)
    os.environ["PYTHONPATH"] = str(CODE / "src")
    OUT.mkdir(parents=True, exist_ok=True)
    for previous in Path("/kaggle/input").glob("*/outputs"):   # resume from an attached earlier run
        print("resume from", previous, flush=True)
        shutil.copytree(previous, OUT, dirs_exist_ok=True)
    if (CODE / "outputs").exists() and not (CODE / "outputs").is_symlink():
        shutil.rmtree(CODE / "outputs")
    (CODE / "outputs").symlink_to(OUT)
    sh([sys.executable, "-c", (
        "from huggingface_hub import snapshot_download\n"
        "for r, d in [('yfan1997/MultipanelVQA_real-world', 'data/raw/real-world'),"
        " ('yfan1997/MultipanelVQA_synthetic', 'data/raw/synthetic')]:\n"
        "    snapshot_download(r, repo_type='dataset', local_dir=d, max_workers=8)")])
    sh([sys.executable, "-m", "spotlightvqa.cli", "audit", "--data-root", "data/raw",
        "--output", "outputs/audit/audit.json"])
    # Build the image caches once, before parallel workers could race on the same file.
    sh([sys.executable, "-c", (
        "from spotlightvqa.audit import load_all\nfrom spotlightvqa.train import ImageCache\n"
        "s, r, _ = load_all('data/raw')\n"
        "for size in (224, 448): ImageCache([x['image'] for x in s + r], size, 'data/cache')")])


def jobs():
    py = sys.executable
    out = [("vlm_spotlight", "outputs/vlm/qwen25vl3b_spotlight",
            [py, "-m", "spotlightvqa.vlm", "--data-root", "data/raw", "--method", "spotlight",
             "--variants", "original", "--output", "outputs/vlm/qwen25vl3b_spotlight"])]
    configs = [  # (tag, split, folds, extra flags); 448 runs use batch 16 x accumulation 2 on T4
        ("E5_full", "concept", range(5), ["--image-size", "448", "--augment", "--spotlight-weight", "0.2",
                                          "--batch-size", "16", "--grad-accum", "2"]),
        ("E4_hires_aug", "concept", range(5), ["--image-size", "448", "--augment",
                                               "--batch-size", "16", "--grad-accum", "2"]),
        ("E2_report1", "concept", range(5), ["--image-size", "224", "--no-mask-answers", "--batch-size", "32"]),
        ("E0_report1", "legacy", [0], ["--image-size", "224", "--no-mask-answers", "--batch-size", "32"]),
    ]
    for tag, split, folds, extra in configs:
        for fold in folds:
            name = f"{split}_f{fold}_{tag}"
            folder = f"outputs/train/{name}"
            out.append((name, folder, [py, "-m", "spotlightvqa.train", "--data-root", "data/raw",
                                       "--split", split, "--fold", str(fold), "--epochs", str(EPOCHS),
                                       "--workers", "2", "--output", folder, *extra]))
    return out


def worker(gpu, tasks, status, lock):
    while True:
        try:
            name, folder, cmd = tasks.get_nowait()
        except queue.Empty:
            return
        if Path(folder, "metrics.json").exists():
            with lock:
                status[name] = {"state": "skipped (done earlier)"}
            continue
        if (time.time() - STARTED) / 3600 > BUDGET_H:
            with lock:
                status[name] = {"state": "not started (time budget)"}
            continue
        log = OUT / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        begin = time.time()
        with log.open("a", encoding="utf-8") as stream:
            code = subprocess.run(cmd, cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        Path(folder, "best.pt").unlink(missing_ok=True)   # 0.9 GB checkpoints are not needed afterwards
        with lock:
            status[name] = {"state": "ok" if code == 0 else f"failed ({code})", "gpu": gpu,
                            "minutes": round((time.time() - begin) / 60, 1)}
            (OUT / "status.json").write_text(json.dumps(status, indent=2))
        print(name, status[name], flush=True)


def main():
    setup()
    import torch
    gpus = max(1, torch.cuda.device_count())
    print("GPUs:", gpus, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    tasks, status, lock = queue.Queue(), {}, threading.Lock()
    for job in jobs():
        tasks.put(job)
    threads = [threading.Thread(target=worker, args=(g, tasks, status, lock)) for g in range(gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
    for cmd in (["-m", "spotlightvqa.evidence", "--data-root", "data/raw", "--runs", "outputs",
                 "--output", "outputs/evidence.json"],
                ["-m", "spotlightvqa.summarize", "--data-root", "data/raw", "--runs", "outputs",
                 "--output", "outputs/summary.json"]):
        subprocess.run([sys.executable, *cmd], cwd=CODE)
    print(json.dumps(status, indent=2))
    print("total hours", round((time.time() - STARTED) / 3600, 2))


if __name__ == "__main__":
    main()
