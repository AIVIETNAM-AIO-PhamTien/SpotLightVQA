"""Build the Kaggle kernel folder and push it (the kernel clones branch tiendo2 from GitHub).

    python scripts/kaggle/push.py [--user tienphamml] [--resume]
    kaggle kernels status tienphamml/spotlightvqa-tiendo2
    kaggle kernels output tienphamml/spotlightvqa-tiendo2 -p outputs/kaggle

Commit and push code changes first: the kernel only sees what is on GitHub.
``--resume`` attaches the previous output of the kernel, so a run cut by the 12-hour limit
continues (finished jobs are skipped).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "spotlightvqa-tiendo2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="tienphamml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    kernel = ROOT / "dist" / "kaggle_kernel"
    if kernel.exists():
        shutil.rmtree(kernel)
    kernel.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/kaggle/run_kaggle.py", kernel / "run_kaggle.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{SLUG}", "title": SLUG, "code_file": "run_kaggle.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [f"{args.user}/{SLUG}"] if args.resume else []}, indent=1))
    print(kernel)
    if not args.dry_run:
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel)], check=True)


if __name__ == "__main__":
    main()
