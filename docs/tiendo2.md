# Tiến độ 2: audit, giao thức mới, cải tiến

Báo cáo đầy đủ ở `report/bao_cao_tien_do_2.tex`, dùng cùng template `ai_conquer2026.cls` với lần 1; biên dịch bằng XeLaTeX/pdfLaTeX trên Overleaf.

## Kết quả audit (CPU, `python -m spotlightvqa.cli audit`)

- Synthetic: 2100 ảnh = 10 khái niệm × 7 số ô × 6 layout × 5 biến thể; chỉ có **60 ảnh con**.
- Cách chia ở lần 1 (nhóm theo Q1‖Q2): 100% câu Test có cùng khái niệm và cùng ảnh con với Train.
- Baseline chỉ đọc câu hỏi trên Test của cách chia lần 1: Overall 88.48, Q1 96.83, **Q2 100.00**, Q3 73.02. Trên Real-World: 51.24, cao hơn mô hình lần 1 (47.67).
- Q1 synthetic: câu có "all" → No ở 1180/1235 câu, câu có "any" → Yes ở 805/865 câu.

File đầy đủ: `outputs/audit/audit.json`.

## Giao thức

- `--split concept`: 5 fold, mỗi fold có 1 khái niệm VQAv2 và 1 khái niệm MagicBrush. Val là các khái niệm của fold kế tiếp. `summarize` gộp dự đoán Test của 5 fold.
- `--split legacy`: chỉ dùng để tái hiện số của lần 1.
- Mỗi câu chỉ được trả lời trong tập đáp án hợp lệ của nó. Consistency chỉ tính ảnh có đủ 3 câu đóng.

## Thiết kế chứng minh (chi tiết ở mục 4 của báo cáo)

| Giả thuyết | So sánh trên cùng câu hỏi | Lệnh |
|---|---|---|
| H1 (đã chứng minh) | rò rỉ của cách chia lần 1 + mốc sàn không nhìn ảnh | `spotlightvqa.cli audit` |
| H2 (chính) | Direct vs Spotlight (ground-then-zoom), Q2+Q3; 420 ảnh `original` + 100 ảnh real-world | `spotlightvqa.vlm --method spotlight --variants original` |
| H3 (cơ chế) | Direct → Oracle crop → Spotlight; độ chính xác theo IoU ≥ 0.5 | cùng lần chạy H2 |
| H4 (huấn luyện) | E5 vs E4 (chỉ khác Spotlight loss), E4 vs E2, từng cấu hình vs Question-only (5 fold theo khái niệm); E0 vs E2 trên Test cũ | `spotlightvqa.train` (E0, E2, E4, E5) |

Kiểm định: McNemar chính xác; CI 95% bằng bootstrap theo ảnh (B = 5000); Holm cho 2 phép thử của H2.
`python -m spotlightvqa.evidence --runs outputs` sinh `outputs/evidence.json`.

Sửa notebook bằng `scripts/make_notebook.py` rồi chạy lại script, không sửa file `.ipynb` trực tiếp.

Chạy trên Kaggle (2×T4): `python scripts/kaggle/push.py` rồi theo dõi bằng `kaggle kernels status tienphamml/spotlightvqa-tiendo2`.
