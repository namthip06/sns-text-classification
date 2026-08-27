# Implementation Plan: ระบบ Text Classification 18 หมวด (ไทยทวิต) + Weak Label Gating

> อ้างอิง: idea `docs/ideas/text-classification-pipeline.md` · spec `docs/specs/text-classification-pipeline.md`
> สร้างจาก: skill planning-and-task-breakdown

## Overview

สร้าง local classifier จำแนกข้อความ Twitter ไทยเป็น 18 หมวด (single-label) พร้อม confidence สำหรับแสดงผล เทรนจาก 100k unlabeled ผ่าน pipeline: preprocess → LLM label (Gemini) + weak label (tag rule) ที่ผ่าน G1 gate → merge → fine-tune PhayaThaiBERT (A/B: with/without weak) → calibrate + threshold (G4) → deploy (API + CLI)

จุดเสี่ยงของโปรเจกต์คือ **weak label ที่ error 15-20% ต้องไม่กัดกร่อนคุณภาพ** วิธีรับมือคือ วัดก่อน (G1 per-tag precision เทียบ referee) → gate → พิสูจน์ด้วย A/B (G3) แผนนี้เรียงตาม dependency: ต่อ foundations ก่อน แล้วแต่ละ task ส่งมอบ feature ที่ทำงานได้จริง

## Architecture Decisions

- **Vertical slice ตาม module ของ spec** — แต่ละ task = 1 module + test ที่รันได้จริง เดิน pipeline ได้ทีละขั้น ไม่ใช่เทรนตอนสุดท้าย
- **config.py เป็น single source of truth** — paths, 18 หมวด, thresholds (G1/G4) อยู่ไฟล์เดียว ตาม spec Code Style
- **LLM label ชุดเดียวใช้ซ้ำเป็น referee (G1) และเป็น label ฝึก (Stage 3-4)** — ไม่สร้าง ground truth แยกเพิ่ม (จาก idea doc)
- **test = ชุดมนุษย์ label แยกจาก val เสมอ** — กันข้อมูลรั่วข้าม hyperparameter tuning กับ evaluate สุดท้าย
- **A/B (G3) ใช้เทรน 2 ตัว**: `with_weak` (weak ที่ผ่าน G1) vs `no_weak` (LLM อย่างเดียว) → เทียบ macro-F1 บนชุดมนุษย์
- **Gemini เป็น provider default สำหรับ LLM label** (ตาม spec) — เปลี่ยน provider ต้อง ask first

## Task List

### Phase 0: Scaffold

#### Task 1: Scaffold โปรเจกต์ + config.py
**Description:** ตั้งโครงสร้างโปรเจกต์ด้วย uv (`pyproject.toml`, `.python-version`, `.gitignore` ที่มี `data/` `models/`) สร้าง `textcls/` src-layout และ `config.py` ที่รวม paths, 18 หมวด (จาก `data/categories.json`), thresholds G1/G4

**Acceptance criteria:**
- [ ] `uv sync` รันได้ไม่มี error, `uv run pytest` รันได้ (0 test)
- [ ] `textcls/config.py` มี constants: DATA_DIR, MODEL_DIR, CATEGORIES_PATH, PRECISION_THRESHOLD=0.85
- [ ] `.gitignore` มี `data/`, `models/`, `.env`
- [ ] API key ผ่าน env var เท่านั้น (อ่านจาก `GEMINI_API_KEY`)

**Verification:**
- [ ] `uv run pytest` ผ่าน
- [ ] `uv run python -c "import textcls.config"` ผ่าน

**Dependencies:** None
**Files likely touched:** `pyproject.toml`, `.gitignore`, `.python-version`, `textcls/__init__.py`, `textcls/config.py`
**Estimated scope:** Small (1-2 files + config)

### Phase 1: Data prep + Labeling

#### Task 2: preprocess.py
**Description:** แปลง raw posts → preprocessed.parquet: ตัด URL/@mention, ตัวซ้ำ → `[CREP]`, `555` → `[LAUGH]`, เก็บ `#` ไว้ก่อน (ตัดตอน train)

**Acceptance criteria:**
- [ ] CLI `python -m textcls.preprocess --input --output` รันได้
- [ ] ตัวซ้ำถูกแทนเป็น `[CREP]`, `555` → `[LAUGH]`
- [ ] URL และ @mention ถูกตัดออก, `#` ยังเก็บคำไว้

**Verification:**
- [ ] `uv run pytest tests/test_preprocess.py` ผ่าน

**Dependencies:** Task 1
**Files likely touched:** `textcls/preprocess.py`, `tests/test_preprocess.py`
**Estimated scope:** Small

#### Task 3: llm_label.py — Gemini label + agreement (G2)
**Description:** สุ่ม sample 5k (bias-sampling หลังวัด distribution แรก 2-3k) → เรียก Gemini label 18 หมวด, ตอบ JSON, temp ~0.2. สุ่ม label ซ้ำ 2 รอบบางส่วนเพื่อวัด agreement (G2) → เขียน `*.jsonl`. ใช้ mock แทน API จริงใน test

**Acceptance criteria:**
- [ ] CLI `python -m textcls.llm_label --input --output --model gemini-2.5-pro --category-file --sample 5000 --bias-sampling` รันได้
- [ ] Prompt มี 18 หมวด + รายละเอียด + few-shot, output เป็น JSON parse ได้
- [ ] สุ่มตัวอย่าง ~10% label ซ้ำ 2 รอบ → คำนวณ agreement rate (G2) ลงไฟล์รายงาน
- [ ] คลาสหายากถูก oversample ผ่าน bias-sampling

**Verification:**
- [ ] `uv run pytest tests/test_llm_label.py` ผ่าน (mock API)
- [ ] มีไฟล์ agreement report (เช่น `data/llm_labels/agreement.json`)

**Dependencies:** Task 1 (Config/CLI pattern)
**Files likely touched:** `textcls/llm_label.py`, `tests/test_llm_label.py`
**Estimated scope:** Medium (prompt + API + agreement logic)

#### Task 4: weak_label.py — G1 per-tag precision gate
**Description:** โหลดผล tag rule (`weak_labels.csv`) เทียบ referee (LLM label) คำนวณ per-tag precision → tag แม่น ≥ 0.85 ปลดล็อกเป็น label ฝึก, ต่ำกว่า → routing-only (ตัดจาก train) → `weak_gated.csv` + รายงานตาราง (tag, precision, verdict)

**Acceptance criteria:**
- [ ] `per_tag_precision()` คำนวณ precision ต่อ tag ถูกต้อง
- [ ] `gate_weak_labels()` ตัด tag ที่ precision < threshold ออกจาก train set
- [ ] CLI `python -m textcls.weak_label --weak --referee --precision-threshold 0.85 --output` รันได้
- [ ] รายงานตาราง `tag, precision, gate verdict` ถูกเขียน

**Verification:**
- [ ] `uv run pytest tests/test_weak_label.py` ผ่าน
- [ ] ตัวอย่าง: weak `gated.csv` ไม่มีแถวของ tag ที่ precision ต่ำกว่าเกณฑ์

**Dependencies:** Task 1 (config/threshold) — **runtime** ต้องมี LLM label จาก Task 3 เพื่อวัด
**Files likely touched:** `textcls/weak_label.py`, `tests/test_weak_label.py`
**Estimated scope:** Medium

### Checkpoint: Data ครบ
- [ ] `data/preprocessed.parquet` มี ~100k แถว
- [ ] `data/llm_labels/*.jsonl` มี sample + agreement report
- [ ] `data/weak_gated.csv` ผ่าน G1
- [ ] Review ตาราง per-tag precision กับมนุษย์ก่อนเทรน (fail fast: tag ไหนพังให้ดรอป)

### Phase 2: Dataset + Train

#### Task 5: dataset.py — merge + split + imbalance
**Description:** merge weak ที่ผ่าน G1 + LLM labels → train/val/test (stratified ตาม label; **test = ชุดมนุษย์ label ~1-2k**) คำนวณ class weight, cap สัดส่วน weak label ใน train ≤ 40%

**Acceptance criteria:**
- [ ] Stratified split ไม่ให้คลาสหายไปจาก val/test
- [ ] `test.csv` มาจากชุดมนุษย์ label เท่านั้น (ไม่ผสม weak/LLM)
- [ ] Class weight คำนวณถูกต้อง (inverse frequency)
- [ ] สัดส่วน weak label ใน train ≤ cap (default 40%)

**Verification:**
- [ ] `uv run pytest tests/test_dataset.py` ผ่าน
- [ ] ตรวจ `test.csv` ว่าไม่มี weak rows ปน (assert columns source)

**Dependencies:** Task 3, 4 (data)
**Files likely touched:** `textcls/dataset.py`, `tests/test_dataset.py`
**Estimated scope:** Medium

#### Task 6: train.py — fine-tune + A/B (G3)
**Description:** fine-tune PhayaThaiBERT (CE + class weight, lr 1e-5–3e-5, ไม่กี่ epoch) รองรับ `--with-weak` / `--no-weak` เพื่อเทรน 2 ตัวสำหรับ A/B, checkpoint ลง `models/with_weak/` และ `models/no_weak/`

**Acceptance criteria:**
- [ ] เทรนบน GPU สำเร็จ, checkpoint ถูกบันทึกทั้ง 2 แบบ
- [ ] validation ใช้ `data/val.csv` (ไม่ใช่ชุดมนุษย์ test)
- [ ] เทรน reproducible (seed + config บันทึกไว้)

**Verification:**
- [ ] รัน `--with-weak` และ `--no-weak` จบได้, `models/` มี checkpoint ทั้งคู่
- [ ] ตัวเลข val loss/accuracy ลงใน log

**Dependencies:** Task 5
**Files likely touched:** `textcls/train.py`
**Estimated scope:** Medium (โมเดล training script)

### Checkpoint: Trained 2 ตัว
- [ ] `models/with_weak/` + `models/no_weak/` อยู่ครบ
- [ ] เทรนจบไม่ OOM/crash
- [ ] Review ก่อนไป eval

### Phase 3: Eval + Calibrate + Deploy

#### Task 7: evaluate.py + calibrate.py — G3 compare + G4 threshold
**Description:** evaluate ทั้ง 2 โมเดลบนชุดมนุษย์ test → macro-F1, per-class recall, confusion matrix → สรุป A/B ว่า weak label ช่วย/ถ่วง/ไม่ต่าง แล้ว temperature scaling + จูน threshold บน val → ต่ำกว่า threshold = "low confidence"

**Acceptance criteria:**
- [ ] รายงาน A/B: ตาราง macro-F1 (with_weak vs no_weak) + verdict
- [ ] confusion matrix + per-class recall ถูกเขียน
- [ ] temperature scale ถูก fit บน val และ threshold ถูกเลือก (เช่น default 0.6)
- [ ] ขึ้น label "low confidence" เมื่อ score < threshold

**Verification:**
- [ ] `uv run python -m textcls.evaluate --model ... --test data/test.csv` ผ่านทั้ง 2 ตัว
- [ ] ตัวเลข macro-F1 พิมพ์ออกมา (เกณฑ์ 0.70 เป็น baseline ตั้งต้น)

**Dependencies:** Task 6
**Files likely touched:** `textcls/evaluate.py`, `textcls/calibrate.py`
**Estimated scope:** Medium

#### Task 8: serve.py + predict.py — Deploy 2 ช่องทาง
**Description:** API FastAPI `POST /classify` → `{category, confidence}` + CLI batch `predict.py` (CSV → CSV พร้อมคอลัมน์ category, confidence) ใช้โมเดลที่ผ่าน calibrate แล้ว

**Acceptance criteria:**
- [ ] `POST /classify` คืน `{"category": ..., "confidence": ...}` โดย confidence ∈ [0,1]
- [ ] CLI `python -m textcls.predict --model --input --output` ผลิต CSV พร้อมคอลัมน์ category + confidence
- [ ] Score ต่ำกว่า threshold ถูก mark เป็น "low confidence"

**Verification:**
- [ ] `uv run pytest tests/test_serve.py` ผ่าน (TestClient)
- [ ] smoke test: curl POST /classify + รัน predict.py กับไฟล์ตัวอย่าง

**Dependencies:** Task 6 (model), 7 (threshold)
**Files likely touched:** `textcls/serve.py`, `textcls/predict.py`, `tests/test_serve.py`
**Estimated scope:** Medium

### Checkpoint: Complete
- [ ] Success criteria 1-8 จาก spec ครบ (train ได้, G1-G4 ทำจริง, macro-F1 ≥ 0.70, deploy 2 ช่องทาง)
- [ ] `uv run pytest` ผ่านทั้งหมด
- [ ] Review กับมนุษย์ก่อนถือว่าเสร็จ

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| งบ Gemini API ไม่พอ label 10-20k | High | sample 2-3k แรกวัด distribution ก่อน, bias-sampling เจาะเฉพาะคลาสหายาก, ใช้ flash สำหรับ referee ถ้า pro แพงเกิน |
| LLM label noise ~5-10% ทำให้ referee ผิด | Med | G2 agreement check, คำนวณ agreement รายงานให้เห็นก่อนใช้ |
| weak label error แบบ systematic ต่อ tag | High | G1 per-tag precision → ดรอป tag ที่พัง (fail fast), cap สัดส่วน weak ใน train |
| คลาสหายากใน test หาย/น้อย | Med | stratified split กัน, bias-sampling ตอน label |
| ยังไม่มีชุดมนุษย์ label ~1-2k | High | ถือเป็น dependency ภายนอก ต้องได้มาก่อน evaluate จริง (ถามมนุษย์) |
| GPU ไม่มี / VRAM ไม่พอ | Med | PhayaThaiBERT ขนาดเล็ก, mixed precision, grad accumulation |

## Open Questions
- [ ] ค่า macro-F1 เป้าหมายที่ยอมรับได้จริง = ? (ใช้ 0.70 เป็น default)
- [ ] Gemini รุ่นไหน (pro/flash) + งบ API สูงสุด?
- [ ] ชุดมนุษย์ label ~1-2k มาจากไหน ใครเป็นคน label? (เกต eval ทั้งหมด)
- [ ] คอลัมน์จริงของ `raw_posts.csv` / `weak_labels.csv` คืออะไร? (สมมติ `text` / `text, predicted_label`)
- [ ] สัดส่วน cap ของ weak label ใน train = 40% เหมาะไหม?

## Parallelization Notes
- **Parallel ได้:** Task 2, 3, 4 เขียน code พร้อมกันได้ (code ไม่พึ่งกัน) แต่ G1 runtime ต้องรอ referee จาก Task 3 ก่อน
- **Must sequential:** Task 5 → 6 → 7 → 8 (data → train → eval → deploy)
- **Needs coordination:** Task 3 กับ 4 แชร์ contract ของ label format (define schema ของ `*jsonl`/`csv` ก่อนแล้ว parallel)

## Definition of Done (ตาม spec Boundaries)
- [ ] ทุก task: `uv run pytest` ผ่านก่อน commit
- [ ] ใช้ uv เสมอ (ไม่ pip/conda)
- [ ] ไม่ commit secrets / `data/` / `models/`
- [ ] เปลี่ยน threshold gate (G1/G4) หรือชุดมนุษย์ test → ถามก่อน
- [ ] weak label เข้า train ได้เฉพาะที่ผ่าน G1
