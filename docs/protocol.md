# Giao thức dùng chung

## Nguồn và phạm vi

- Kế hoạch: `C:\Users\Admin\Downloads\plan-tuần-1.pdf`, mục 3 Pipeline, trang 2. Chỉ triển khai phần pipeline; survey và chạy baseline VLM thuộc các đầu việc khác.
- Paper: https://aclanthology.org/2024.acl-long.370/
- Repo: https://github.com/UCSB-AI/MultipanelVQA
- Commit được kiểm tra: `d2588ce5dc363b2ced6ff0f3497e775141865ed9`.
- Synthetic: https://huggingface.co/datasets/yfan1997/MultipanelVQA_synthetic
- Real-world: https://huggingface.co/datasets/yfan1997/MultipanelVQA_real-world

## Chuẩn bị benchmark cho người chạy baseline

Dữ liệu lớn không tải tự động. Tải từ hai nguồn Hugging Face theo README upstream (cần Git LFS nếu clone), lưu vào `data/raw/`. Kiểm tra cấu trúc tải về để xác định JSON annotations và thư mục chứa ảnh; không giả định tên file. Lệnh `prepare` nhận định dạng upstream: một JSON list; mỗi record có `composited_gallery` và `qa` là list các cặp `question`/`answer` chuỗi.

Ví dụ dưới đây dùng đường dẫn minh họa, cần thay bằng file thực tế:

```powershell
.\.venv\Scripts\python.exe -m spotlightvqa.cli prepare --source data/raw/synthetic/annotations.json --image-root data/raw/synthetic/images --subset synthetic --images 10 --seed 42 --output data/processed/synthetic.jsonl
.\.venv\Scripts\python.exe -m spotlightvqa.cli prepare --source data/raw/real-world/annotations.json --image-root data/raw/real-world/images --subset real-world --images 10 --seed 42 --output data/processed/real-world.jsonl
```

Lấy mẫu theo **ảnh**, giữ toàn bộ QA của ảnh được chọn. `--images` là số ảnh tối đa cho từng subset, không phải số QA. `qa_0`, `qa_1`, `qa_2` chỉ là chỉ số câu hỏi trong annotation gốc; chưa gán tên loại câu hỏi khi chưa kiểm tra annotation. Seed 42, thứ tự đầu vào được chuẩn hóa trước khi lấy mẫu. Không dùng tập đánh giá để tune prompt. Chạy/chấm từng subset riêng hoặc nối hai file JSONL rồi chấm để có tổng hợp; ID có prefix subset.

Schema mỗi dòng:

```json
{"question_id":"synthetic:image.png:0","image":"absolute/path/image.png","question":"...","answer":"...","subset":"synthetic","question_type":"qa_0"}
```

Adapter model do người chạy baseline cung cấp, ví dụ `my_model.py` có hàm `call_model(image_file, prompt)`. Module phải import được trong môi trường đang chạy. Hàm chịu trách nhiệm cố định seed, model revision, prompt wrapper, image preprocessing, decoding và thiết bị. CLI chuyển nguyên câu hỏi annotation vào hàm, không tự sửa prompt.

```powershell
.\.venv\Scripts\python.exe -m spotlightvqa.cli infer --data data/processed/synthetic.jsonl --predictor my_model:call_model --output outputs/predictions-synthetic.jsonl
.\.venv\Scripts\python.exe -m spotlightvqa.cli evaluate --data data/processed/synthetic.jsonl --predictions outputs/predictions-synthetic.jsonl --model "MODEL_ID@REVISION" --seed 42 --protocol "Ghi prompt wrapper; preprocessing; decoding; GPU; adapter revision ở đây"
```

Prediction JSONL: `{"question_id":"...","prediction":"..."}`. Chấm điểm từ chối ID trùng, thiếu hoặc thừa để tránh vô tình báo điểm trên tập dự đoán chưa đầy đủ. Khi infer lỗi, lệnh dừng, không ghi file dự đoán thiếu. Chưa có resume/batching cho job GPU lớn.

## Metric

Metric local `normalized_exact_match_v1`: bỏ `</s>`, `)`, `.`, `,`; chuyển chữ thường; thu gọn khoảng trắng. Đáp án chuẩn hóa rỗng luôn sai. Không tự suy ra đồng nghĩa, không bỏ mạo từ, không dùng VQAv2 consensus. Accuracy = số QA đúng / tổng QA; thống kê overall, theo subset và chỉ số câu hỏi. Điểm ở thang [0, 1].

Đây là metric local có tính xác định, **không tương đương metric chính thức**. Code upstream `inference/utils/infer_on_data.py` có nhánh khớp trực tiếp, heuristic yes/no và gọi GPT-4 judge khi chưa kết luận được. Muốn so với paper, dùng đúng giao thức judge upstream, ghi model judge/revision, prompt, cấu hình và lưu quyết định từng câu. Pipeline này không tự gọi dịch vụ tính phí.

Đã thấy hai vấn đề ở entrypoint upstream tại commit trên: biến `total` tăng theo ảnh trong khi `correct` tăng theo câu hỏi; entrypoint unpack 5 giá trị nhưng hàm trả 4. Không chạy nguyên entrypoint rồi coi số đó là accuracy theo QA. Metric local đếm QA và có test bảo vệ mẫu số. Nếu sửa giao thức upstream để chạy baseline, ghi patch và ảnh hưởng trong experiment card.

## Kết quả và khả năng tái lập

Mỗi lần `evaluate` tạo một `outputs/<run_id>/` mới, lưu prediction, điểm chi tiết và metadata trong `experiment.json`: hash dữ liệu/dự đoán, seed khai báo, model, protocol, Git commit/dirty, Python, danh sách package và đường dẫn artifacts. Chưa commit repo thì `git_commit` là null; khi làm nghiên cứu nên commit code trước lúc chạy và dùng model revision cụ thể.

Seed của evaluate là metadata giao thức, không điều khiển model bên ngoài. Repo hiện chưa có lệnh train. Dữ liệu, model và output lớn được gitignore.

Bàn giao cho nhóm: mỗi người xác nhận máy Kaggle/Colab/GPU sẵn có; chọn 1–2 VLM nhỏ; chạy cùng tập con của cả hai subset và cùng giao thức. Ghi khác biệt về model revision, độ phân giải, lượng tử hóa, prompt, judge và cỡ tập con khi so với paper.
