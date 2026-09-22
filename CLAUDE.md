# CLAUDE.md

คำแนะนำสำหรับทำงานในโปรเจกต์นี้ เริ่มจาก **[CLAUDE.md นี้](#โปรเจกต์)** (สถานะปัจจุบัน) · รายละเอียดเต็มที่ **[README.md](./README.md)** + **[docs/specs/text-classification-pipeline.md](./docs/specs/text-classification-pipeline.md)** + **[docs/plans/implementation-plan.md](./docs/plans/implementation-plan.md)**

## โปรเจกต์

Classifier ข้อความทวิตไทยเป็น **19 หมวด (single-label — 18 เนื้อหา + `no_match`)** พร้อม confidence เทรนจาก **weak label (tag rule) ทั้งหมด** โดยตรง — ไม่ใช้ LLM (Gemini) และไม่มีชุดมนุษย์ label:
`data/weak_labels.csv` → clean text → map taxonomy (weak 16 → หมวดใน categories.json) → stratified split → fine-tune `airesearch/wangchanberta-base-att-spm-uncased` (เปลี่ยนเป็น PhayaThaiBERT ได้ที่ `--model`)

**Decision (2026-08-27):** ยกเลิก LLM ออกจาก pipeline — เดิม `llm_label.py` (Gemini label/G2) และ G1 per-tag precision gate (referee = LLM) ถูกถอด เพราะ weak labels มี label + confidence อยู่แล้ว (logic ซ้ำซ้อน) และไม่มี referee/มนุษย์ label มาเป็น ground truth

**Decision (2026-09-22):** ถอด G4 (confidence threshold 0.6) ออก — `no_match` เปลี่ยนจาก label ที่ derive จาก threshold เป็น **คลาสเทรนที่ 19** ใน `categories.json`; predict = argmax 19 คลาส (ไม่มีคอลัมน์ `low_confidence` แล้ว) · temperature scaling ยังอยู่ (ปรับค่า confidence ไม่ตัดสิน label)

Core check:
- **G3** ประเมิน macro-F1 บน `val` ที่ hold-out จาก weak (agreement กับ weak rule — ไม่ใช่ความจริงสัมบูรณ์) · ใน `evaluate.py`

## Pipeline

```
weak_labels ─T3 dataset(clean+map 16→19+stratified split+save)→ merged/train/val.csv ─T4 train→ models/ ─T5 eval(G3)+calibrate─┬─T6 serve/predict
raw_posts ─T2 preprocess: เก็บไว้ก่อน (ไว้คราวหน้า) ────────────────────────────────────────────────────────────────────────────┘
```

- `data/raw_posts.csv` (258k แถวดิบ) **ยังไม่ใช้** — เปิดขั้นตอน preprocess/LLM อีกครั้งเมื่อต้องการ ground truth (ถามก่อน)

## Commands

```bash
uv sync                     # ติดตั้ง deps (ใช้ uv เสมอ ไม่ใช้ pip/conda เอง)
uv run pytest               # tests — รันก่อน commit ทุกครั้ง
uv run python -m textcls.dataset --weak data/weak_labels.csv --categories data/categories.json --out data/
uv run python -m textcls.train --train data/train.csv --val data/val.csv --categories data/categories.json --out models/
uv run tensorboard --logdir models/model/runs   # ดูกราฟ train (metrics ต่อ epoch อยู่ models/model/metrics.json)
uv run python -m textcls.evaluate --model models/model/ --test data/val.csv
uv run python -m textcls.calibrate --model models/model/ --val data/val.csv
uv run python -m textcls.predict --model models/model/ --input in.csv --output out.csv            # บังคับ --text-column เพื่อโหมดเงียบ
uv run python -m textcls.predict --input in.xlsx                            # ไม่ระบุ --text-column = interactive (เลือก model จาก models/ + sheet/คอลัมน์, preview, .xlsx ได้)
```

## โครงสร้าง

- `src/textcls/` — source (src-layout) · ไฟล์ละหนึ่งสเตจ: `config.py` `preprocess.py` `dataset.py` `train.py` `infer.py` (shared: โหลดโมเดล + predict logits) `calibrate.py` `evaluate.py` `predict.py` (`serve.py` ยังไม่สร้าง — เลื่อน)
- `config.py` — **ทุก** path รวมที่เดียว
- `configs/weak_label_map.json` — mapping weak→หมวด 19 (มนุษย์แก้ได้; default ยังรอ confirm)
- `data/` `models/` — **gitignored** (ข้อมูลจริง + checkpoints) ห้าม commit
- `docs/` — `specs/` spec · `plans/implementation-plan.md` แผน · `notes/data-contract.md` โครงสร้างคอลัมน์ทุกไฟล์ · `ideas/` `interviews/` แนวคิดต้นทาง

## สถานะสำคัญ (ณ 2026-09-22)

- Tasks: 1 ✅ · 2 (preprocess) deferred · 3 (dataset) ✅ · 4 (train) ✅ · 5 (eval+calibrate) ✅ · 6 (predict CLI) ✅ · `serve.py` (API) เลื่อน
- train monitoring: `metrics.json` ต่อ epoch (eval loss + macro-F1 แบบ G3) ผ่าน `MetricsCallback` + TensorBoard (`report_to=["tensorboard"]` → log ที่ `models/<tag>/runs/`, ดูด้วย `uv run tensorboard --logdir models/<tag>/runs`) — dep `tensorboard` เพิ่มเมื่อ 2026-09-01 (อนุมัติแล้ว); หมายเหตุ: transformers v5 ไม่มี `--logging_dir` แล้ว
- dead code ถอดแล้ว: `llm_label.py`/`weak_label.py` + tests, `--no-weak`/`--human-test`, dep `google-genai` — **เก็บ `tiktoken`/`protobuf` ไว้** (transformers ต้องใช้สกัด tokenizer WangchanBERTa)
- Smoke ครบ chain บน GPU: train 736 แถว (`models/model/`) → calibrate (T=0.22) → evaluate (macro-F1 0.48) → predict — **ยังไม่เทรนเต็ม 128k**
- mapping weak→19: `religion`/`child_sexual_content`/`no_match` ไม่มี weak source → ไม่มี train data (class weight = 0)
- predict (2026-09-15): merge interactive CLI เข้าไฟล์เดียว — รับ `.xlsx` (dep `openpyxl`, อนุมัติแล้ว), ไม่ระบุ `--text-column` = interactive (เลือก sheet/คอลัมน์ + preview + tqdm progress + bar chart; dep `tqdm`, อนุมัติแล้ว), ไม่ระบุ `--output` → `<input>_predicted.<ext>`, แถวข้อความว่างเว้นว่าง; score ผ่าน calib เสมอ — คอลัมน์ผลลัพธ์ `category`/`confidence` (ไม่ใช่ `label`/`score` ตามสคริปต์ต้นทาง; ตัด `low_confidence` ออก 2026-09-22)
- 42 tests ผ่าน

## ข้อห้าม / ถามก่อน

- **Never:** commit secrets/API key · commit `data/` `models/` · เปิดใช้ LLM/preprocess จาก raw โดยไม่ได้รับอนุมัติ
- **Ask first:** เปลี่ยนโมเดล base · เพิ่ม dependency ใหม่ · เปลี่ยนชุดประเมิน (val) · ถอดโค้ดที่ตายแล้ว (llm_label/weak_label) ก่อนเคลียร์ให้ชัด
- Style: type hints ทุกฟังก์ชัน · ไม่มี magic number เปล่า (ย้ายขึ้น `config.py`) · config ที่เดียว · ฟังก์ชันเล็กหนึ่งหน้าที่
