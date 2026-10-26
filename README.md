# SpotLightVQA — pipeline tuần 1

Triển khai phần **Pipeline / người 5** của `plan-tuần-1.pdf`: môi trường dùng chung, khung repo, metric có test và mẻ synthetic nhỏ chạy từ code gốc. Cả nhóm dùng **CPython 3.15.2** (bản chuẩn có GIL) và cùng `requirements.txt` đã ghim phiên bản bằng `==`. Phiên bản Python được ghi trong `.python-version`. Pipeline CPU không cần tải VLM.

## Cài đặt (PowerShell, tại thư mục repo)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Resolve-Path src).Path
git clone https://github.com/UCSB-AI/MultipanelVQA.git third_party/MultipanelVQA
git -C third_party/MultipanelVQA checkout d2588ce5dc363b2ced6ff0f3497e775141865ed9
.\.venv\Scripts\python.exe -m pytest -q
```

Tải [Python 3.15.2](https://www.python.org/downloads/release/python-3152/) và kiểm tra `python --version` trả về `Python 3.15.2` trước khi tạo môi trường. Nếu đã có `.venv` dùng Python cũ, đổi tên môi trường cũ rồi tạo `.venv` mới bằng Python 3.15.2; cài lại requirements không đổi phiên bản Python của môi trường. Mỗi thành viên tạo môi trường riêng từ cùng `requirements.txt`, cài bằng `pip install -r requirements.txt` và kiểm tra bằng `python -m pip check`. Khi thay đổi dependency, cập nhật phiên bản cố định trong file này và cả nhóm cài lại từ cùng bản repo.

Code chạy trực tiếp từ `src/`. Mỗi lần mở terminal PowerShell mới, chạy `$env:PYTHONPATH = (Resolve-Path src).Path` tại gốc repo trước khi dùng CLI. `pytest.ini` tự cấu hình đường dẫn cho test.

Linux: dùng Python 3.15.2 với `python3.15 -m venv .venv`, thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`, và đặt `export PYTHONPATH="$PWD/src"` trước khi dùng CLI. Trên Colab, kiểm tra phiên bản Python của kernel; kernel khác 3.15.2 chưa đáp ứng môi trường chung này. Trong notebook, thêm `src/` vào `sys.path` của kernel nếu import trực tiếp. Thư viện VLM/GPU cài trong môi trường riêng theo model; adapter dùng hàm `call_model(image_file, prompt)` trả về chuỗi. Không cần API key cho pipeline local.

## Chạy mẻ synthetic thử và kiểm tra toàn luồng

```powershell
.\.venv\Scripts\python.exe -m spotlightvqa.cli synthetic
.\.venv\Scripts\python.exe -m spotlightvqa.cli evaluate --data data/synthetic/smoke/questions.jsonl --predictions data/synthetic/smoke/reference_predictions.jsonl --model reference-oracle-smoke --protocol "offline fixtures; reference answers; CPU; no model inference" --notes "Pipeline validation only; not a baseline result"
```

Tạo **3 ảnh 1000×1000, lần lượt 2/4/8 panel**, cùng QA JSONL, metadata gốc và provenance. Ảnh nguồn là các ô màu có nhãn fixture, không phải COCO/VQAv2. Script chạy nguyên cell compositor của notebook gốc ở commit cố định, thay dữ liệu đầu vào bằng fixture và sửa metadata `image_paths` bị lấy từ biến cũ. Có thể mở ảnh trong `data/synthetic/smoke/`.

`reference_predictions.jsonl` sao chép đáp án để kiểm tra plumbing; điểm 100% ở bước này **không phải kết quả VLM**. Để tái tạo benchmark synthetic bằng ảnh thật, làm theo [hướng dẫn upstream](https://github.com/UCSB-AI/MultipanelVQA#generate-synthetic-multipanel-images) và hai notebook gốc; cần chuẩn bị VQAv2/COCO. Không tự coi fixture là dữ liệu benchmark.

## Cấu trúc

```text
data/raw/          dữ liệu gốc, không commit
data/processed/    QA chuẩn hóa và tập con
data/synthetic/    ảnh và metadata sinh thử
src/spotlightvqa/  CLI, adapter dữ liệu, metric, generator
notebooks/         khám phá dữ liệu
docs/              giao thức, nguồn, biên bản kiểm tra
outputs/<run_id>/  experiment.json, metrics.json, predictions.jsonl
tests/             test metric và pipeline
third_party/       checkout gốc, không commit
```

Chi tiết chuẩn bị hai subset, giao thức metric và hướng dẫn bàn giao: [docs/protocol.md](docs/protocol.md). Kết quả kiểm tra trên máy hiện tại: [docs/validation.md](docs/validation.md).
