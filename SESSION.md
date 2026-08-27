# Session Log — text_classification

อัปเดตล่าสุด: 2026-08-27 · โหมด build: `/build auto` (autonomous) ตาม
`docs/specs/text-classification-pipeline.md` + `docs/plans/implementation-plan.md`

## สรุปสั้น

กำลังสร้าง pipeline 8 ขั้น → **ทำเสร็จ 6** (scaffold → train) พร้อม smoke train
พิสูจน์บน 100 ประโยค · หยุดก่อน Task 7 เพราะ**รอข้อมูลจริง** (LLM labels + human
labels) ยังไม่มี · mapping weak→18 หมวดยังรอการ confirm

## ทำอะไรไปแล้ว ✅

| Task | ผลงาน | Commit |
|------|-------|--------|
| 1 | scaffold: uv project + `textcls/config.py` + tests | `81d63ed` |
| — | spec + plan + interviews/ideas + data contract | `2829d29` |
| — | dev tooling (jupyter) + EDA notebook + data prep scripts | `bb7de84` |
| 2 | `preprocess.py` — ล้างภาษาไทย (URL/@/#/555/อักษรซ้ำ) | `f5a93e2` |
| 3 | `llm_label.py` — Gemini label + G2 agreement | `463c4b0` |
| 4 | `weak_label.py` — G1 per-tag precision gate (≥0.85) | `50bc84d` |
| 5 | `dataset.py` — merge + stratified split + class weight + weak cap 40% | `b9f0d17` |
| — | mapping weak→18 หมวด แยกเป็นไฟล์ `configs/weak_label_map.json` (มนุษย์แก้ได้) | `011b78d` |
| 6 | `train.py` — fine-tune WangchanBERTa/PhayaThaiBERT + A/B (--with-weak/--no-weak) | `df983cc` |

**ทดสอบ:** 50 tests ผ่าน · **Smoke train** 100 ประโยคบน GPU (RTX 4050 6GB):
`train_loss 2.457 → eval_loss 2.147`, ~8 it/s, 10 steps → checkpoint เซฟ/โหลด/
predict ได้จริง (`models/smoke/with_weak/`)

**Deps ที่เพิ่มระหว่างทาง:** sentencepiece, protobuf, tiktoken, accelerate
(ทั้งหมดมาจาก transformers v5 ต้องการตอน runtime)

## อยู่ขั้นตอนไหน 📍

ระหว่าง **Task 6 กับ Task 7** — pipeline ครบครึ่งหลัง (label → train) แต่ยังไม่
มีข้อมูลจริงเข้า G3/G4. ข้อมูลที่ใช้เทรนตอนนี้ = smoke 100 ประโยคจาก weak เท่านั้น

## ยังค้างอยู่ / ถูกบล็อก ⛔

1. **mapping weak→18 หมวด — ยังไม่ confirm**  
   default อยู่ที่ `configs/weak_label_map.json` (scam→fraud, e-cigarettes→ecig,
   forged documents→forged_docs, kratom drink→kratom, royal institution &
   royal family→royal, alcohol advertising→alcohol, copyright infringement→copyright)
   → ผู้ใช้เลือก "อยากแก้ mapping" ไว้ แต่ยังไม่ได้ระบุรายการที่แก้

2. **`data/llm_labels/*.jsonl` — ยังไม่มี**  
   ต้องรัน `textcls.llm_label` กับ Gemini จริง (ต้อง `GEMINI_API_KEY`, เสียค่าใช้จ่าย
   ต่อการเรียก) + ใช้ referee นี่แหละเป็น input ของ G1/G2 ด้วย

3. **`data/human_labels.csv` (~1-2k แถว) — ยังไม่มี**  
   ชุดมนุษย์ label ใช้เป็น **test set** ใน Task 5/7 (ตาม acceptance: test = human เท่านั้น)

## เหลืออะไร ⏭️

| # | งาน | ต้องมีก่อน |
|---|-----|-----------|
| 7 | `evaluate.py` (G3: A/B macro-F1 + confusion matrix) + `calibrate.py` (G4: temperature scaling + threshold 0.6 → `low confidence`) | llm_labels + human_labels + mapping confirm |
| 8 | `serve.py` (FastAPI POST /classify → `{category, confidence}`) + `predict.py` (CLI batch CSV→CSV) | model เทรนจริง (ผล Task 6) |

**ของจริงยังต้องทำ:**
- รัน LLM label บน sample ใหญ่ (10k+ ตามแผน) → weak_gated → dataset จริง → เทรน A/B จริง
- ตัดสินผล G3 (A/B) แล้ว lock ว่าใช้ weak หรือไม่
- calibrate G4 แล้วตั้ง threshold → deploy

## วิธีดำเนินต่อ 🔀

1. ตอบ mapping: แก้ `configs/weak_label_map.json` หรือยืนยัน default
2. รัน Gemini label (`uv run python -m textcls.llm_label …` — ดูตัวอย่างใน README)
3. เตรียม `data/human_labels.csv`
4. กลับมา `/build` ต่อ Task 7 → 8

*บันทึกเก่า: transcript เต็มอยู่ที่
`~/.claude/projects/-home-nummmm-Document-Code-Work-text-classification/`*
