# Implementation Plan: ระบบ Text Classification 18 หมวด (ไทยทวิต) — เทรนจาก Weak Label ทั้งหมด

> อ้างอิง: spec `docs/specs/text-classification-pipeline.md` · data contract `docs/notes/data-contract.md`
> **Decision (2026-08-27):** ยกเลิก LLM (Gemini) ออกจาก pipeline ทั้งหมด — ไม่มี `llm_label` (เดิม Task 3/G2), ไม่มี referee G1, ไม่มีชุดมนุษย์ label. เทรนจาก **weak label ทั้งหมด** โดยตรง, ประเมินบน **val ที่ hold-out จาก weak**.

## Overview

สร้าง local classifier จำแนกข้อความ Twitter ไทยเป็น 18 หมวด (single-label) พร้อม confidence เทรนจาก **`data/weak_labels.csv` (tag rule) ทั้งชุด** เพียงแหล่งเดียว:

```
weak_labels.csv → clean text → map taxonomy (weak 16 → 18 หมวด) → stratified split (train/val)
  → fine-tune → evaluate บน val (G3) → calibrate + threshold 0.6 (G4) → deploy (API + CLI)
```

`data/raw_posts.csv` (258k แถว) **เก็บไว้ก่อน (ไว้คราวหน้า)** — ถ้ากลับมาใช้ เช่น เพิ่ม LLM label/ground truth ค่อยเปิดขั้นตอน preprocess ใหม่

> **หมายเหตุ:** การรันตามคำสั่ง/Verification ในไฟล์นี้เป็นแค่ **smoke test แบบจำนวนน้อย** เท่านั้น (เช่น 1,000 ประโยค) เพื่อเช็คว่า pipeline ผ่านทั้ง chain — **ไม่ใช่** การเทรน/เทสด้วยข้อมูลทั้งชุด

**จุดที่เปลี่ยนจากแผนเดิม:**
- ~~Task 3 `llm_label.py`~~ — ลบ (logic ซ้ำซ้อน: weak labels มี label + confidence อยู่แล้ว ไม่ต้องให้ Gemini label ซ้ำ)
- ~~G1 per-tag precision gate~~ — ลบ (referee = LLM หายไปแล้ว → วัดไม่ได้; เทรนจาก weak ทั้งหมดตรงๆ)
- ~~G2 LLM agreement~~ — ลบ
- ~~test = ชุดมนุษย์ / LLM hold-out~~ — ลบ; ประเมินบน **val ที่ hold-out จาก weak**
- ~~A/B with/without weak (G3 เดิม)~~ — ลบ; ข้อมูลทั้งหมดเป็น weak เหมือนกัน A/B ที่เหลือคือเทียบโมเดล base ผ่าน `--model`

## Architecture Decisions

- **Weak label = single source ของ label** — `weak_labels.csv` (คอลัมน์ `content, flag, category, score, confidence, …`) ไม่มี label อื่นใน pipeline
- **ไม่ใช้ LLM (Gemini) เลย** — ไม่ label, ไม่ referee, ไม่ประเมิน
- **raw_posts ไว้คราวหน้า** — เก็บข้อมูลดิบไว้; preprocess (Task 2) เลื่อน; ถ้าอยากได้ ground truth จริง/คุณภาพดีขึ้น ค่อยเปิด LLM/มนุษย์ภายหลัง
- **ประเมินบน val ที่ hold-out จาก weak** — G3 วัด macro-F1 = agreement กับ label ของ tag rule (ไม่ใช่ความจริงสัมบูรณ์) — ต้องระบุข้อจำกัดในรายงานเสมอ
- **config.py เป็น single source of truth** — paths, 18 หมวด, thresholds (G4) อยู่ไฟล์เดียว
- **deploy 2 ช่องทาง** — API (FastAPI) + CLI batch

## Task List

### Phase 0: Scaffold

#### Task 1: Scaffold โปรเจกต์ + config.py
**Description:** โครงสร้าง uv + src-layout + `config.py` รวม paths/18 หมวด/threshold G4. ลบ constant ที่ไม่ใช้แล้ว (`PRECISION_THRESHOLD` ของ G1)

**Acceptance criteria:**
- [ ] `uv sync` + `uv run pytest` รันได้
- [ ] `textcls/config.py`: DATA_DIR, MODEL_DIR, CATEGORIES_PATH, CONFIDENCE_THRESHOLD=0.6 (G4)
- [ ] `.gitignore` มี `data/`, `models/`, `.env`

**Verification:** `uv run pytest` ผ่าน · `uv run python -c "import textcls.config"` ผ่าน
**Dependencies:** None · **Scope:** Small

### Phase 1: Data + Train

#### Task 2: preprocess — **เลื่อน (deferred)**
**Description:** ล้าง Thai text (URL/@, `555`→`[LAUGH]`, ตัวซ้ำ→`[CREP]`) สำหรับ raw posts — **ยังไม่ต้องใช้** (raw ไว้คราวหน้า). โค้ด `preprocess.py` + test มีอยู่แล้ว; dataset (Task 3) เรียก `clean_text` ซ้ำตรงๆ กับ weak content

**Acceptance criteria:**
- [ ] (deferred) `clean_text` ยังถูกใช้ใน Task 3 กับ weak content
- [ ] กลับมาใช้เมื่อเริ่มเทรนจาก raw

**Dependencies:** — · **Scope:** Deferred

#### Task 3: dataset.py — clean + map taxonomy + split + save
**Description:** อ่าน `data/weak_labels.csv` → กรองแถวที่มี label (`flag==AUTO`, `category` ไม่ว่าง) → clean text (reuse `clean_text`) → map taxonomy weak (16 คลาส) → 18 หมวดของ `categories.json` ผ่าน `configs/weak_label_map.json` → **stratified split → train/val** → คำนวณ class weight → **save ทั้ง merged (ก่อน split) + train.csv + val.csv**

**Acceptance criteria:**
- [ ] กรอง weak แถวที่ไม่มี label / `flag != AUTO` ออก
- [ ] text ถูก clean (URL/@/#/555/ตัวซ้ำ) ก่อนเข้าเทรน
- [ ] map taxonomy weak→18 หมวด (label ที่ map ไม่ได้ → ตัด + warn) ตาม `configs/weak_label_map.json`
- [ ] Stratified split ไม่ให้คลาสหายจาก val
- [ ] **save as: merged + train.csv + val.csv** (รวมถึง `class_weights.json`)
- [ ] Class weight คำนวณถูก (inverse frequency)

**Verification:**
- [ ] `uv run pytest tests/test_dataset.py` ผ่าน (ปรับ test: ไม่มี `--human-test`/`--llm`)
- [ ] `data/{merged,train,val}.csv` + `class_weights.json` ถูกเขียน

**Dependencies:** Task 1 · **Files likely touched:** `textcls/dataset.py`, `tests/test_dataset.py` · **Scope:** Medium

#### Task 4: train.py — fine-tune
**Description:** fine-tune `airesearch/wangchanberta-base-att-spm-uncased` (default) / PhayaThaiBERT (`--model`) ด้วย CE + class weight → checkpoint `models/`. **ไม่มี A/B weak vs no-weak แล้ว** (ทุกอย่างเป็น weak); A/B ที่เหลือ = เทียบโมเดล base ผ่าน `--model`

**Acceptance criteria:**
- [ ] เทรนบน GPU สำเร็จ, checkpoint ถูกบันทึก
- [ ] validation ใช้ `data/val.csv` (hold-out จาก weak)
- [ ] reproducible (seed + config บันทึก)
- [ ] ลบ flag `--no-weak` (ไม่มีความหมายแล้ว)

**Verification:** เทรนจบได้ + `models/` มี checkpoint + val loss/accuracy ใน log
**Dependencies:** Task 3 · **Files likely touched:** `textcls/train.py` · **Scope:** Medium

### Phase 2: Eval + Deploy

#### Task 5: evaluate.py + calibrate.py — G3 on val + G4 threshold
**Description:** evaluate บน `data/val.csv` → macro-F1, per-class recall, confusion matrix → รายงาน (**G3: agreement กับ weak label — ไม่ใช่ความจริงสัมบูรณ์, ระบุข้อจำกัดในรายงาน**) แล้ว temperature scaling + จูน threshold 0.6 (G4) → score ต่ำกว่า = "low confidence"

**Acceptance criteria:**
- [ ] รายงาน macro-F1 + per-class recall + confusion matrix บน val
- [ ] temperature scale ถูก fit บน val + threshold 0.6
- [ ] ขึ้น label "low confidence" เมื่อ score < threshold

**Verification:** `uv run python -m textcls.evaluate --model models/… --test data/val.csv` ผ่าน · พิมพ์ macro-F1 (baseline ตั้งต้น 0.70 — วัด agreement กับ weak)
**Dependencies:** Task 4 · **Files likely touched:** `textcls/evaluate.py`, `textcls/calibrate.py` · **Scope:** Medium

#### Task 6: serve.py + predict.py — Deploy 2 ช่องทาง
**Description:** API FastAPI `POST /classify` → `{category, confidence}` + CLI batch `predict.py` (CSV → CSV พร้อมคอลัมน์ category, confidence) ใช้โมเดลที่ผ่าน calibrate

**Acceptance criteria:**
- [ ] `POST /classify` คืน `{"category": ..., "confidence": ...}` โดย confidence ∈ [0,1]
- [ ] CLI `python -m textcls.predict --model --input --output` ผลิต CSV พร้อมคอลัมน์ category + confidence
- [ ] Score ต่ำกว่า threshold ถูก mark เป็น "low confidence"

**Verification:** `uv run pytest tests/test_serve.py` ผ่าน (TestClient) · smoke: curl POST + predict.py กับไฟล์ตัวอย่าง
**Dependencies:** Task 4 (model), 5 (threshold) · **Scope:** Medium

### Checkpoints

- **Data ครบ:** `data/weak_labels.csv` ผ่าน Task 3 → `merged/train/val.csv` + `class_weights.json`
- **Trained:** `models/` มี checkpoint (เทรนจาก weak จริง ไม่ใช่แค่ smoke)
- **Complete:** รายงาน G3 (val macro-F1) + G4 (threshold) + deploy 2 ช่องทาง + `uv run pytest` ผ่าน

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| weak label error สูงโดยไม่มี referee วัด | High | decision ยอมรับ — ประเมินบน val (agreement กับ weak เอง) ระบุข้อจำกัดทุกครั้ง; ถ้าต้องการ ground truth จริง → เปิด LLM/มนุษย์ภายหลัง (raw เก็บไว้) |
| mapping weak 16→18 หมวด ผิด | Med | mapping อยู่ `configs/weak_label_map.json` (มนุษย์แก้ได้) — review ก่อนเทรนจริง |
| คลาส imbalance (rule บาง tag เยอะเกิน) | Med | stratified split + class weight |
| ไม่มีชุดอ้างอิง → ตัวเลขประเมินตีความยาก | Med | รายงาน macro-F1 เป็น "agreement" ไม่ใช่ accuracy; บันทึก distribution ของ train/val ทุก run |

## Open Questions

- [ ] ค่า macro-F1 เป้าหมายที่รับได้ (ตอนนี้วัด agreement กับ weak — ตั้ง baseline ใหม่, 0.70 เดิมตั้งไว้บนมนุษย์ test)?
- [ ] mapping weak→18 หมวดใน `configs/weak_label_map.json` — confirm หรือแก้?
- [ ] ถอดโค้ดที่ตายแล้วออก: ลบ `llm_label.py` + test, `weak_label.py` + test, flag `--no-weak`, `--human-test`?
- [ ] ถ้ากลับมาใช้ raw: ขั้นตอน LLM label / ground truth ต้องคุยกันใหม่ (ไว้คราวหน้า)

## Definition of Done (ตาม spec Boundaries)

- [ ] ทุก task: `uv run pytest` ผ่านก่อน commit
- [ ] ใช้ uv เสมอ (ไม่ pip/conda)
- [ ] ไม่ commit secrets / `data/` / `models/`
- [ ] เปลี่ยน threshold gate (G4) หรือชุดประเมิน (val) → ถามก่อน
- [ ] เทรนจาก weak ทั้งหมดโดยตรง (ไม่มี gate G1)
