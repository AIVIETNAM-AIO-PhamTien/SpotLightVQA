"""Generate notebooks/colab_tiendo2.ipynb (run: python scripts/make_notebook.py)."""
import json
from pathlib import Path

CELLS = [
("md", """# SpotLightVQA — thí nghiệm tiến độ 2 (Colab)

Chạy lần lượt từ trên xuống. GPU khuyến nghị: **L4 hoặc A100** (T4 chạy được nhưng chậm, không có bf16).
Kết quả ghi vào `outputs/` và sao lưu sang Google Drive sau mỗi run. Khi Colab ngắt, chạy lại từ đầu:
run đã có `metrics.json` được bỏ qua, phần VLM tự resume từ `predictions.jsonl`.

| Bước | Nội dung |
|---|---|
| 1–3 | Cài đặt, tải dữ liệu, audit rò rỉ (CPU) |
| 4 | Train: tái hiện lần 1 (E0), baseline chia theo khái niệm (E2), công thức cải tiến (E4), + Spotlight loss (E5) |
| 5 | Qwen2.5-VL-3B zero-shot: Direct và Spotlight (ground-then-zoom) trong cùng một lần chạy |
| 6 | Tổng hợp bảng cho báo cáo |

Thời gian chưa đo; bước 4 và 5 in tốc độ thực tế (giây/epoch, giây/câu) để ước lượng phần còn lại.
Không nên giảm cỡ mẫu: các kiểm định ở bước 6 cần đủ câu hỏi để có ý nghĩa thống kê."""),
("code", """!nvidia-smi
from google.colab import drive
drive.mount('/content/drive')
DRIVE_OUT = '/content/drive/MyDrive/SpotLightVQA_tiendo2_outputs'"""),
("md", """## 1. Lấy code
Cách A: clone GitHub (sau khi đã push nhánh `tiendo2`). Cách B: upload file `SpotLightVQA-tiendo2.zip`."""),
("code", """import os, subprocess, pathlib, sys
REPO = 'https://github.com/AIVIETNAM-AIO-PhamTien/SpotLightVQA.git'
BRANCH = 'tiendo2'
os.chdir('/content')
if not pathlib.Path('SpotLightVQA').exists():
    if subprocess.run(['git', 'clone', '-b', BRANCH, REPO, 'SpotLightVQA']).returncode != 0:
        from google.colab import files
        print('Clone lỗi -> upload SpotLightVQA-tiendo2.zip')
        name = next(iter(files.upload()))
        subprocess.run(['unzip', '-q', name, '-d', 'SpotLightVQA'], check=True)
os.chdir('/content/SpotLightVQA')
!pip install -q -r requirements-colab.txt
os.environ['PYTHONPATH'] = '/content/SpotLightVQA/src'
sys.path.insert(0, '/content/SpotLightVQA/src')
!python -m pytest -q"""),
("md", "## 2. Tải MultipanelVQA từ HuggingFace"),
("code", """from huggingface_hub import snapshot_download
for rid, d in [('yfan1997/MultipanelVQA_real-world', 'data/raw/real-world'),
               ('yfan1997/MultipanelVQA_synthetic', 'data/raw/synthetic')]:
    snapshot_download(rid, repo_type='dataset', local_dir=d, max_workers=16)
!ls data/raw/synthetic/test/synthetic_image | wc -l
!ls data/raw/real-world/test/real-world_image | wc -l"""),
("md", "## 3. Audit dữ liệu (CPU): rò rỉ của cách chia báo cáo 1 + baseline không dùng ảnh"),
("code", """def backup():
    subprocess.run(['mkdir', '-p', DRIVE_OUT])
    subprocess.run(['rsync', '-a', '--exclude', '*.pt', 'outputs/', DRIVE_OUT + '/'])

!python -m spotlightvqa.cli audit --data-root data/raw --output outputs/audit/audit.json
backup()"""),
("md", """## 4. Train

| Mã | Split | Cấu hình | Câu hỏi nghiên cứu |
|---|---|---|---|
| E0 | legacy | như lần 1 (224, không che logit) | tái hiện con số 97% |
| E2 | concept | như lần 1 | còn bao nhiêu khi khái niệm test chưa từng thấy |
| E4 | concept | che logit + 448 + augment màu | công thức cải tiến, chưa có Spotlight loss |
| E5 | concept | E4 + Spotlight loss 0.2 | E5 vs E4 chỉ khác đúng Spotlight loss |

Tên run `<split>_f<fold>_<tag>` để bước 6 gom fold."""),
("code", """FOLDS = [0, 1, 2, 3, 4]
EPOCHS = 5
BS = 32
EXPERIMENTS = {
    'E0_report1': ('legacy',  ['--image-size', '224', '--no-mask-answers']),
    'E2_report1': ('concept', ['--image-size', '224', '--no-mask-answers']),
    'E4_hires_aug': ('concept', ['--image-size', '448', '--augment']),
    'E5_full':    ('concept', ['--image-size', '448', '--augment', '--spotlight-weight', '0.2']),
}
RUN_ONLY = list(EXPERIMENTS)"""),
("code", """for tag in RUN_ONLY:
    split, extra = EXPERIMENTS[tag]
    for fold in (FOLDS if split == 'concept' else [0]):
        out = f'outputs/train/{split}_f{fold}_{tag}'
        if pathlib.Path(out, 'metrics.json').exists():
            print('skip', out)
            continue
        cmd = ['python', '-m', 'spotlightvqa.train', '--data-root', 'data/raw', '--split', split,
               '--fold', str(fold), '--epochs', str(EPOCHS), '--batch-size', str(BS),
               '--workers', '4', '--output', out, *extra]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        pathlib.Path(out, 'best.pt').unlink(missing_ok=True)   # checkpoint ~0.9 GB, không cần giữ
        backup()"""),
("md", """## 5. Qwen2.5-VL-3B zero-shot: Direct và Spotlight (ground-then-zoom)
Một lần chạy cho cả hai: mỗi câu được trả lời trực tiếp trên ảnh đầy đủ (Direct), rồi Spotlight để VLM tự khoanh ô liên quan, cắt và phóng to ô đó trước khi trả lời. `metrics.json` có `synthetic` / `real-world` (Spotlight) và `*_direct` (Direct) trên cùng tập ảnh.
Tập đánh giá: **toàn bộ 420 ảnh biến thể `original`** (biến thể duy nhất có toạ độ ô đúng, cần cho kiểm định cơ chế H3) **+ 100 ảnh real-world**."""),
("code", """VLM = 'Qwen/Qwen2.5-VL-3B-Instruct'
out = 'outputs/vlm/qwen25vl3b_spotlight'
if not pathlib.Path(out, 'metrics.json').exists():
    subprocess.run(['python', '-m', 'spotlightvqa.vlm', '--data-root', 'data/raw', '--model', VLM,
                    '--method', 'spotlight', '--variants', 'original', '--output', out], check=True)
backup()"""),
("code", """# Xem nhanh vài ô Spotlight khoanh được (để đưa hình vào báo cáo)
import json
from PIL import Image, ImageDraw
from spotlightvqa.audit import load_all
syn, real, _ = load_all('data/raw')
rows = {r['question_id']: r for r in syn + real}
shown = 0
for line in open(f'{out}/predictions.jsonl', encoding='utf-8'):
    rec = json.loads(line)
    if rec.get('box') and shown < 4:
        row = rows[rec['question_id']]
        im = Image.open(row['image']).convert('RGB'); ImageDraw.Draw(im).rectangle(rec['box'], outline='red', width=6)
        print(row['question'], '| đáp án:', row['answer'], '| direct:', rec['direct'], '| spotlight:', rec['prediction'])
        display(im.resize((400, int(400 * im.height / im.width))))
        shown += 1"""),
("md", """## 6. Kiểm định giả thuyết + tổng hợp cho báo cáo
`evidence.json`: H2 (Spotlight vs Direct, McNemar + bootstrap theo ảnh, Holm), H3 (Direct/Spotlight/Oracle, tỉ lệ khoanh trúng ô), H4 (E5 vs E4, E4 vs E2, từng cấu hình vs Question-only; E0 vs E2 trên test cũ).
Các dòng LaTeX in ra dùng để điền Bảng 5–6 trong `report/bao_cao_tien_do_2.tex`; hoặc gửi lại file `summary.json`."""),
("code", """!python -m spotlightvqa.evidence --data-root data/raw --runs outputs --output outputs/evidence.json
!python -m spotlightvqa.summarize --data-root data/raw --runs outputs --output outputs/summary.json
backup()
!cd outputs && zip -qr ../outputs_tiendo2.zip . -x '*.pt'
from google.colab import files
files.download('outputs_tiendo2.zip')"""),
]


def main():
    notebook = {"cells": [], "nbformat": 4, "nbformat_minor": 0,
                "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    for kind, source in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        notebook["cells"].append(cell)
    path = Path(__file__).resolve().parents[1] / "notebooks" / "colab_tiendo2.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
