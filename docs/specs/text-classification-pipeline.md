# Spec: ระบบ Text Classification 18 หมวด (ไทยทวิต) — เทรนจาก Weak Label ทั้งหมด

> ต้นทางแนวทาง: `docs/ideas/text-classification-pipeline.md` · บทสัมภาษณ์: `docs/interviews/text-classification-options-summary.md`
> **Decision (2026-08-27):** ไม่ใช้ LLM (Gemini) / ไม่มีชุดมนุษย์ label — เทรนจาก weak label (tag rule) ทั้งหมดตรงๆ, ประเมินบน val ที่ hold-out จาก weak

## Objective

สร้าง **local classifier** จำแนกข้อความ Twitter ไทย เข้า **18 หมวด** (single-label) พร้อมค่า **confidence** เพื่อนำไปแสดงผลต่อ โดยเทรนจาก **weak label (tag rule) ทั้งชุด** โดยตรง:

```
weak_labels.csv → Clean (URL/@/555/ซ้ำ) → Map taxonomy (weak 16 → 18 หมวด)
  → Stratified split (train/val) → Fine-tune → Eval บน val (G3) → Confidence (G4) → Deploy (API + CLI)
```

**User:** ทีมที่มี weak label (ผล tag rule) ครอบคลุมข้อมูลแล้ว ต้องการโมเดล local ที่ใช้งานได้เร็ว
**Success:** ได้โมเดล local + confidence ใช้แสดงผลได้จริง; **raw posts เก็บไว้ก่อน (ไว้คราวหน้า)** ถ้าต้องการ ground truth/คุณภาพดีขึ้นค่อยเปิด LLM/มนุษย์ภายหลัง

### Flow อ้างอิง

```
weak_labels.csv ──clean──▶ ──map 16→18──▶ ──stratified split──▶ train.csv / val.csv
   ──fine-tune──▶ ──eval บน val (G3)──▶ ──calibrate + threshold 0.6 (G4)──▶ deploy (API + CLI)
```

### Gates ที่เหลือ

- **G3 — ประเมิน macro-F1 บน val (hold-out จาก weak)** — วัด agreement กับ label ของ tag rule (ไม่ใช่ความจริงสัมบูรณ์) — ระบุข้อจำกัดในรายงานเสมอ
- **G4 — temperature scaling + threshold 0.6** — score ต่ำกว่า = `low confidence`

*(เดิมมี G1 per-tag precision gate + G2 LLM agreement — ลบทั้งคู่ เนื่องจาก referee (LLM) และมนุษย์ label ไม่มีใน pipeline แล้ว)*

---

## Tech Stack

| ส่วน | เทคโนโลยี |
|---|---|
| Python venv | **uv** (สร้าง/จัดการ `pyproject.toml`) |
| Model | `airesearch/wangchanberta-base-att-spm-uncased` หรือ `PhayaThaiBERT` (กำหนดได้จาก config/`--model`) |
| Deep learning | PyTorch + HuggingFace `transformers` |
| Data | pandas / pyarrow (CSV/parquet) · scikit-learn (metrics, split) |
| Serve | FastAPI + uvicorn (HTTP) + CLI แบบ batch |
| Test | pytest |

---

## Commands

```bash
# ── Environment (uv) ──
uv sync
uv run pytest

# ── Pipeline ──
# 1. dataset: clean + map taxonomy + stratified split + save (merged/train/val)
uv run python -m textcls.dataset \
    --weak data/weak_labels.csv --categories data/categories.json --out data/

# 2. train (fine-tune จาก weak ทั้งชุด)
uv run python -m textcls.train --train data/train.csv --val data/val.csv \
    --categories data/categories.json --out models/

# 3. eval (G3: บน val) + calibrate (G4: threshold)
uv run python -m textcls.evaluate --model models/model/ --test data/val.csv
uv run python -m textcls.calibrate --model models/model/ --val data/val.csv

# 4. deploy
uv run uvicorn textcls.serve:app --host 0.0.0.0 --port 8000   # HTTP API
uv run python -m textcls.predict --model models/model/ \
    --input data/new_posts.csv --output results.csv           # CLI batch
```

---

## Project Structure

```
text_classification/
├── pyproject.toml            # uv-managed dependencies
├── README.md
├── docs/
│   ├── ideas/                # one-pager แนวทาง
│   ├── interviews/           # บันทึกบทสัมภาษณ์
│   ├── specs/                # spec ฉบับนี้
│   └── notes/data-contract.md  # โครงสร้างคอลัมน์ทุกไฟล์
├── configs/
│   └── weak_label_map.json   # weak taxonomy → 18 หมวด (มนุษย์แก้ได้)
├── data/                     # ⚠ gitignored
│   ├── raw_posts.csv         # 258k โพสต์ดิบ (เก็บไว้คราวหน้า — ยังไม่ใช้)
│   ├── weak_labels.csv       # ผล tag rule (content, flag, category, score, confidence, …)
│   ├── categories.json       # 18 หมวด + รายละเอียด
│   ├── merged.csv / train.csv / val.csv + class_weights.json
├── models/                   # ⚠ gitignored — checkpoints
├── src/textcls/
│   ├── config.py             # paths, 18 หมวด, thresholds (G4)
│   ├── preprocess.py         # clean Thai text (URL/@/555/ซ้ำ) — reuse จาก dataset
│   ├── dataset.py            # clean + map taxonomy + stratified split + save
│   ├── train.py              # fine-tune (CE + class weight)
│   ├── evaluate.py           # macro-F1, per-class recall, confusion matrix (G3)
│   ├── calibrate.py          # temperature scaling + threshold (G4)
│   ├── serve.py              # FastAPI: POST /classify → {category, confidence}
│   └── predict.py            # CLI batch scoring
└── tests/
    ├── test_preprocess.py
    ├── test_dataset.py
    └── test_serve.py
```

*(ถอดออกจาก pipeline แล้ว: `llm_label.py`, `weak_label.py` + test — รอการลบจากโค้ด)*

---

## Code Style

Python + type hints ตลอด, ฟังก์ชันเล็กๆ หนึ่งหน้าที่, ชื่อตรงไปตรงมา, config ทั้งหมดอยู่ที่ `config.py`

```python
# textcls/dataset.py — ตัวอย่างสไตล์
from pathlib import Path
import pandas as pd
from textcls.preprocess import clean_text

def load_weak(path: str | Path) -> pd.DataFrame:
    """โหลด weak label → [content_clean, label] (กรองแถว flag==AUTO, label ไม่ว่าง)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[(df["flag"] == "AUTO") & df["category"].notna()].copy()
    df["content_clean"] = df["content"].astype(str).map(clean_text)
    return df[["content_clean", "category"]].rename(columns={"category": "label"})
```

**กฎ:** type hints ทุกฟังก์ชัน · ไม่มี magic number เปล่า (ย้ายขึ้น constant/config) · ฟังก์ชันเล็กหนึ่งหน้าที่

---

## Testing Strategy

- **Framework:** pytest, เก็บที่ `tests/`
- **ระดับ:** unit test เป็นหลัก
- **Coverage ที่ต้องมี (จุดเสี่ยง):**
  - `preprocess`: ตัวซ้ำ → `[CREP]`, `555` → `[LAUGH]`, ตัด `#` เก็บคำ, ตัด URL/@
  - `dataset`: กรอง weak ที่ไม่มี label, map taxonomy (weak→18) ถูกต้อง, stratified split กันคลาสหายจาก val, save merged/train/val
  - `serve`: POST /classify คืน `{category, confidence}` อยู่ในช่วงที่ถูกต้อง
- **ไม่ทดสอบ:** ความแม่นของโมเดล (เป็นขั้น evaluate ไม่ใช่ unit test)

---

## Boundaries

- **Always:**
  - รัน `uv run pytest` ก่อน commit ทุกครั้ง
  - ใช้ uv เป็น venv manager เสมอ (ไม่ใช้ pip/conda เอง)
  - `data/` และ `models/` อยู่ใน `.gitignore`
- **Ask first:**
  - เปลี่ยนโมเดล base (WangchanBERTa → ตัวอื่น)
  - เพิ่ม dependency ใหม่
  - เปลี่ยนชุดประเมิน (val) หรือ threshold gate (G4)
  - **กลับไปใช้ LLM / เปิดขั้นตอน preprocess จาก raw**
- **Never:**
  - commit secrets / API key ลง git
  - commit `data/`, `models/`, หรือ checkpoint

---

## Success Criteria

1. **เทรนโมเดลได้** — fine-tune สำเร็จบน GPU จาก weak ทั้งชุด, `models/` มี checkpoint
2. **dataset.py ทำงานครบ** — `data/{merged,train,val}.csv` + `class_weights.json` ถูกเขียน (stratified, label map ถูก)
3. **G3 ทำจริง** — รายงาน macro-F1 + per-class recall + confusion matrix บน val (ระบุว่าเป็น agreement กับ weak label)
4. **G4 ใช้งานได้** — calibration + threshold 0.6 → โพสต์ confidence ต่ำ ขึ้น label "low confidence"
5. **คุณภาพ baseline:** macro-F1 ≥ 0.70 (agreement กับ weak บน val — ค่าตั้งต้นที่ต้อง confirm)
6. **Deploy 2 ช่องทาง:**
   - CLI: `predict.py` รับ CSV → คืน CSV พร้อมคอลัมน์ category + confidence
   - API: `POST /classify` รับ text → `{"category": ..., "confidence": 0.87}`
7. **Reproducible:** `uv sync` ขึ้นโปรเจกต์ใหม่แล้วรัน pipeline ต่อได้จาก `pyproject.toml` ตัวเดียว

---

## Open Questions

- [ ] ค่า macro-F1 เป้าหมายที่รับได้ (วัด agreement กับ weak — 0.70 เดิมตั้งบนมนุษย์ test)?
- [ ] mapping weak→18 หมวดใน `configs/weak_label_map.json` — confirm หรือแก้?
- [ ] จะถอดโค้ดที่ตายแล้ว (llm_label/weak_label/--no-weak) เมื่อไหร่?
