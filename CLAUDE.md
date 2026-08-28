# CLAUDE.md

คำแนะนำสำหรับทำงานในโปรเจกต์นี้ เริ่มจาก **[SESSION.md](./SESSION.md)** (สถานะปัจจุบัน) · รายละเอียดเต็มที่ **[README.md](./README.md)** + **[docs/specs/text-classification-pipeline.md](./docs/specs/text-classification-pipeline.md)**

## โปรเจกต์

Classifier ข้อความทวิตไทยเป็น **18 หมวด (single-label)** พร้อม confidence เทรนจาก **weak label (tag rule) ทั้งหมด** โดยตรง — ไม่ใช้ LLM (Gemini) และไม่มีชุดมนุษย์ label:
`data/weak_labels.csv` → clean text → map taxonomy (weak 16 → 18 หมวด) → stratified split → fine-tune `airesearch/wangchanberta-base-att-spm-uncased` (เปลี่ยนเป็น PhayaThaiBERT ได้ที่ `--model`)

**Decision (2026-08-27):** ยกเลิก LLM ออกจาก pipeline — เดิม `llm_label.py` (Gemini label/G2) และ G1 per-tag precision gate (referee = LLM) ถูกถอด เพราะ weak labels มี label + confidence อยู่แล้ว (logic ซ้ำซ้อน) และไม่มี referee/มนุษย์ label มาเป็น ground truth

Core checks (2 ตัว):
- **G3** ประเมิน macro-F1 บน `val` ที่ hold-out จาก weak (agreement กับ weak rule — ไม่ใช่ความจริงสัมบูรณ์) · ใน `evaluate.py`
- **G4** temperature scaling + threshold 0.6 → ต่ำกว่า = `low confidence` · ใน `config.py` (`CONFIDENCE_THRESHOLD`)

## Pipeline

```
weak_labels ─T3 dataset(clean+map 16→18+stratified split+save)→ merged/train/val.csv ─T4 train→ models/ ─T5 eval(G3)+calibrate(G4)─┬─T6 serve/predict
raw_posts ─T2 preprocess: เก็บไว้ก่อน (ไว้คราวหน้า) ────────────────────────────────────────────────────────────────────────────┘
```

- `data/raw_posts.csv` (258k แถวดิบ) **ยังไม่ใช้** — เปิดขั้นตอน preprocess/LLM อีกครั้งเมื่อต้องการ ground truth (ถามก่อน)
- โค้ดที่ตายแล้วรอการถอด: `llm_label.py` + test, `weak_label.py` + test, flag `--no-weak`, `--human-test` ของ dataset

## Commands

```bash
uv sync                     # ติดตั้ง deps (ใช้ uv เสมอ ไม่ใช้ pip/conda เอง)
uv run pytest               # tests — รันก่อน commit ทุกครั้ง
uv run python -m textcls.dataset --weak data/weak_labels.csv --categories data/categories.json --out data/
uv run python -m textcls.train --train data/train.csv --val data/val.csv --categories data/categories.json --out models/
uv run python -m textcls.evaluate --model models/model/ --test data/val.csv
uv run python -m textcls.calibrate --model models/model/ --val data/val.csv
```

(CLI เป้าหมายของ dataset/commands ใหม่ — โค้ด dataset.py ยังต้อง rework: ลบ `--llm`/`--human-test`/G1 mapping)

## โครงสร้าง

- `src/textcls/` — source (src-layout) · ไฟล์ละหนึ่งสเตจ: `config.py` `preprocess.py` `dataset.py` `train.py` (+ `evaluate.py` `calibrate.py` `serve.py` `predict.py` ยังไม่สร้าง = Task 5-6)
- `config.py` — **ทุก** path/threshold รวมที่เดียว (`CONFIDENCE_THRESHOLD=0.6`)
- `configs/weak_label_map.json` — mapping weak→18 หมวด (มนุษย์แก้ได้; default ยังรอ confirm)
- `data/` `models/` — **gitignored** (ข้อมูลจริง + checkpoints) ห้าม commit
- `docs/` — `specs/` spec · `plans/implementation-plan.md` แผน · `notes/data-contract.md` โครงสร้างคอลัมน์ทุกไฟล์ · `ideas/` `interviews/` แนวคิดต้นทาง

## สถานะสำคัญ (ณ 2026-08-27)

- Pipeline ใหม่ (decision ลบ LLM/human) เพิ่งเขียนลง plan/spec/CLAUDE นี้ — **โค้ดยังตามไม่ทัน**: `dataset.py` ยังรับ `--human-test` (required), `train.py` ยังมี `--no-weak`, `llm_label.py`/`weak_label.py` ยังอยู่ (จะถอด)
- Tasks ตามแผนใหม่: 1 (scaffold) ✅ · 2 (preprocess) deferred · 3 (dataset rework) ⏳ · 4 (train) ✅ smoke · 5 (eval+calibrate) ⏳ · 6 (serve+predict) ⏳
- Smoke train 100 ประโยคจาก weak บน GPU ผ่านแล้ว (`models/smoke/`) · 50 tests ผ่าน
- mapping weak→18 หมวดยังไม่ confirm

## ข้อห้าม / ถามก่อน

- **Never:** commit secrets/API key · commit `data/` `models/` · เปิดใช้ LLM/preprocess จาก raw โดยไม่ได้รับอนุมัติ
- **Ask first:** เปลี่ยนโมเดล base · เพิ่ม dependency ใหม่ · เปลี่ยนชุดประเมิน (val) หรือ threshold gate (G4) · ถอดโค้ดที่ตายแล้ว (llm_label/weak_label) ก่อนเคลียร์ให้ชัด
- Style: type hints ทุกฟังก์ชัน · ไม่มี magic number เปล่า (ย้ายขึ้น `config.py`) · config ที่เดียว · ฟังก์ชันเล็กหนึ่งหน้าที่
