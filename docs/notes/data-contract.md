# Data Contract — ไฟล์ data ทั้งหมดของ pipeline text classification 18 หมวด

> สร้าง: 2026-08-27 · ตรวจสอบข้อมูลจริงก่อนเขียน (ไฟล์ `data/` + spec `docs/specs/text-classification-pipeline.md`)

## ภาพรวม data flow

```
[input]                    [pipeline สร้าง]                          [input/eval]
raw_posts.csv ──T2──▶ preprocessed.parquet ──T3──▶ llm_labels/*.jsonl ──┐
                                                                    ├─T5─▶ train/val/test.csv ──T6▶ เทรน
weak_labels.csv ──────────────T4──▶ weak_gated.csv ─────────────────┘     └─▶ eval บน human_labels
categories.json (ให้ prompt ใช้) · human_labels.csv (test set)
```

## ตารางไฟล์ทั้งหมด

| # | ไฟล์ | แหล่ง | รูปแบบ | คอลัมน์ | สถานะ |
|---|------|------|--------|---------|-------|
| 1 | `data/raw_posts.csv` | ข้อมูลจริง | CSV | `content` | ✅ มี (`data/alltime_25_26_content_dedup.csv`) |
| 2 | `data/weak_labels.csv` | tag rule (ระบบนอก) | CSV | `content, predicted_label` | ✅ มี (`data/weak_labels_auto.csv`) |
| 3 | `data/categories.json` | **มนุษย์ supply** | JSON | 18 หมวด + ชื่อไทย + description |  ✅ มี (`data/categories.json`) |
| 4 | `data/human_labels.csv` | **มนุษย์ label ~1-2k** | CSV | `content, label` | ❌ ยังไม่มี (เกต eval) |
| 5 | `data/preprocessed.parquet` | Task 2 | parquet | `content_clean` | 🔜 สร้าง |
| 6 | `data/llm_labels/*.jsonl` | Task 3 | JSONL | `content, category, confidence` | 🔜 สร้าง |
| 7 | `data/llm_labels/agreement.json` | Task 3 (G2) | JSON | `{agreement_rate, ...}` | 🔜 สร้าง |
| 8 | `data/weak_gated.csv` | Task 4 (G1) | CSV | `content, predicted_label, source` | 🔜 สร้าง |
| 9 | `data/train.csv` / `val.csv` / `test.csv` | Task 5 | CSV | `content, label, source` | 🔜 สร้าง |

## ข้อมูลจริง (checked 2026-08-27)

- `data/alltime_25_26_content_dedup.csv` — **258,136 แถว**, คอลัมน์เดียว `content` (มี BOM `﻿` นำหน้าหัวคอลัมน์ → ตอนอ่านต้องใช้ `utf-8-sig`)
- ตัวอย่าง: `#SCKCatalogueรวบรวมรายการสินค้าประเภทเมนเบรกเกอร์2P/ลูกย่อย1...`
- จำนวนแถวมากกว่าเป้า ~100k ของแผน → budget พอ

## สถานะข้อมูล (สิ่งที่ยังต้อง supply)

### ✅ มีแล้ว
- `content` ข้อมูลดิบ 258k แถว (dedup แล้ว)

### ❌ ยังไม่มี — ต้องได้จากมนุษย์/ระบบนอก ก่อน dev ถึงขั้นนั้น
- **`data/categories.json`** — 18 หมวด ยังไม่นิยามที่ไหน ต้อง supply ก่อน Task 3 (llm_label ใช้ใน prompt)
- **`data/weak_labels.csv`** — ผลจาก tag rule (ระบบนอกเครื่อง) ต้อง supply ก่อน Task 4; ต้องยืนยันคอลัมน์จริงจากระบบนั้น
- **`data/human_labels.csv`** — ชุดมนุษย์ label ~1-2k สำหรับ test (เกต evaluate ทั้งหมด ตาม spec ห้ามข้าม) ต้อง supply ก่อน Task 7

## Schema ที่เสนอ (ยังต้อง confirm)

### 1. ชื่อคอลัมน์มาตรฐาน: `content` (ไม่ใช่ `text`)
spec เดิมสมมติ `text` แต่ข้อมูลจริงเป็น `content` → เสนอใช้ `content` เป็นชื่อมาตรฐานทุกไฟล์ pipeline

### 2. `data/categories.json` (มนุษย์ supply)
```json
[
  {"id": "sport", "name": "กีฬา", "description": "ข่าวกีฬา คะแนน แมตช์ นัด..."},
  {"id": "politics", "name": "การเมือง", "description": "..."}
]
```
(ต้องได้ 18 หมวดจริงของโปรเจกต์)

### 3. `data/weak_labels.csv` (จาก tag rule)
```
content, predicted_label
"แชมป์โลกแน่แล้วทีมนี้", sport
```

### 4. `data/human_labels.csv` (มนุษย์ label)
```
content, label
"วันนี้กองเชียร์คึกคักสุดๆ", sport
```

## Pipeline ผลิต (ไม่ต้อง supply) — ตัวอย่าง

### `data/preprocessed.parquet` (Task 2)
คอลัมน์ `content_clean` — `#SCKCatalogue...` → ตัด URL/@, `555`→`[LAUGH]`, ตัวซ้ำ→`[CREP]` (เก็บ `#` ไว้ก่อน ตัดตอน train)

### `data/llm_labels/*.jsonl` (Task 3)
```json
{"content": "แชมป์โลกแน่แล้วทีมนี้", "category": "sport", "confidence": 0.92}
```
พร้อมรายงาน agreement: `data/llm_labels/agreement.json`

### `data/weak_gated.csv` (Task 4, G1)
เฉพาะ tag ที่ precision ≥ เกณฑ์ (ค่า default 0.85 จาก config) + รายงานตาราง per-tag precision

### `data/train.csv` / `val.csv` / `test.csv` (Task 5)
มีคอลัมน์ `source` บอกที่มาของ label (`llm` / `weak` / `human`) — ใช้ assert ว่า test ไม่มี weak ปน (ตาม acceptance ของ Task 5)

## TODO / Open Questions
- [ ] 18 หมวดจริงของ `categories.json` มาจากไหน?
- [ ] `weak_labels.csv` จากระบบ tag rule มีคอลัมน์จริงชื่ออะไร?
- [ ] ชุดมนุษย์ label ใครเป็นคน label?
