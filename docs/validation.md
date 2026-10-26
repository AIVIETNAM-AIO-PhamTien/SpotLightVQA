# Biên bản kiểm tra pipeline

Kiểm tra lại ngày 27/10/2026 (Asia/Bangkok) trên Windows, **CPython 3.15.2** bản chuẩn có GIL. Runtime embeddable chính thức được tải vào `.tmp/python315/`, xác minh SHA-256 theo [Python.org](https://www.python.org/downloads/release/python-3152/), cài các dependency đã ghim trong `requirements.txt` vào thư mục riêng của runtime. Test dùng `pytest.ini`; CLI chạy trực tiếp từ `src/`. Môi trường `.venv` cũ vẫn dùng Python 3.11.9; cần tạo môi trường mới theo README để chuyển sang 3.15.2.

Các phiên bản đã kiểm tra: Pillow 13.0.0, NumPy 2.5.4, OpenCV headless 5.0.0.93, pytest 9.1.1. Metadata editable cũ trong `src/spotlightvqa.egg-info/` được chuyển vào `.tmp/` để không còn khai báo dependency của cấu hình đã bỏ.

| Kiểm tra | Kết quả |
| --- | --- |
| pip check trên runtime 3.15.2 | No broken requirements found |
| `python -m pytest -q` | 9 passed |
| CLI synthetic dùng notebook upstream | 3 ảnh PNG, 1000×1000, 2/4/8 panel |
| CLI evaluate với reference predictions | 3/3 QA; ghi metrics, experiment card, predictions |
| Test adapter → inference → evaluate → artifacts | Passed, adapter fixture kiểm tra đường dẫn ảnh và câu hỏi |
| Kiểm tra ảnh | Đã mở ảnh 8 panel; màu và vị trí đúng layout demo |

Artifact kiểm tra Python 3.15.2 ở `.tmp/python315-smoke/`, kết quả evaluate ở `.tmp/python315-outputs/<run_id>/`. Artifact thử ban đầu trên Python 3.11.9 vẫn ở `data/synthetic/smoke/` và `outputs/<run_id>/`. Bản ghi smoke là evaluation, không phải log train. Chưa có commit ban đầu, vì vậy Git commit được ghi null; source đang untracked. Không tự tạo commit.

**Giới hạn kiểm tra:** ảnh nguồn là fixture màu; prediction ở lượt smoke lấy từ đáp án chuẩn. Chưa tải benchmark đầy đủ, chưa chạy VLM/GPU và chưa chạy GPT-4 judge. Không có điểm baseline để so với paper. Khâu lấy tập con đã test bằng annotation giả theo schema upstream; khi tải dữ liệu thật cần kiểm tra đường dẫn và schema thực tế trước khi chạy.

Notebook upstream được kiểm tra ở commit `d2588ce5dc363b2ced6ff0f3497e775141865ed9`; checksum notebook và các thay đổi dữ liệu đầu vào được ghi trong file `provenance.json` tại mỗi thư mục artifact synthetic. Chưa kiểm tra bộ dependency mới trên Linux/macOS hoặc môi trường VLM/GPU.
