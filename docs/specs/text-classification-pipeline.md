# Spec: ระบบ Text Classification 18 หมวด (ไทยทวิต) + ใช้ Weak Label อย่างปลอดภัย

> ต้นทางแนวทาง: `docs/ideas/text-classification-pipeline.md` · บทสัมภาษณ์: `docs/interviews/text-classification-options-summary.md`

## Objective

สร้าง **local classifier** จำแนกข้อความ Twitter ไทย เข้า **18 หมวด** (single-label) พร้อมค่า **confidence** เพื่อนำไปแสดงผลต่อ โดยเทรนจากข้อมูล **100k unlabeled** ผ่าน pipeline:
LLM (Gemini) สร้าง label + weak label (tag rule) ที่ผ่าน **gating** (วัด → ปลดล็อกเฉพาะส่วนที่แม่น) แล้ว fine-tune บน PhayaThaiBERT

**User:** ทีมที่มีคลังโพสต์ 1 แสน อันที่ยังไม่มี label
**Success ในเชิงภาพรวม:** ได้โมเดล local ที่แม่นพอใช้งาน พร้อม confidence ใช้แสดงผลได้จริง โดย weak label ไม่กัดกร่อนคุณภาพ

### Flow อ้างอิง (จาก idea doc)

```
[100k unlabeled] → Preprocess → Weak label (G1: per-tag precision gate)
   → LLM label Gemini (G2: agreement check) → Merge → Split (test = มนุษย์ label)
   → Fine-tune (G3: A/B test) → Confidence (G4: calibration+threshold) → Deploy (API + CLI)
```

---

## Tech Stack

| ส่วน | เทคโนโลยี |
|---|---|
| Python venv | **uv** (สร้าง/จัดการ `pyproject.toml`) |
| Model | `airesearch/wangchanberta-base-att-spm-uncased` หรือ `PhayaThaiBERT` (กำหนดได้จาก config) |
| Deep learning | PyTorch + HuggingFace `transformers` |
| LLM label | **Google Gemini API** ผ่าน `google-genai` SDK |
| Data | pandas / polars (อ่าน CSV) · scikit-learn (metrics, split) |
| Serve | FastAPI + uvicorn (HTTP) + CLI แบบ batch |
| Deploy | export ONNX/TorchScript + quantization (option) |
| Test | pytest |

---

## Commands

```bash
# ── Environment (uv) ──
uv init                                # สร้าง pyproject.toml + .python-version
uv add torch transformers pandas scikit-learn datasets \
       google-genai fastapi uvicorn pydantic onnxruntime \
       sentencepiece tokenizers pytest

# ── Pipeline (Stage โดย Stage) ──
# 1. Preprocess ข้อมูลดิบ
uv run python -m textcls.preprocess \
    --input data/raw_posts.csv --output data/preprocessed.parquet

# 2. Weak label gating (G1): วัด per-tag precision เทียบ referee (LLM) แล้ว gate
uv run python -m textcls.weak_label \
    --weak data/weak_labels.csv --referee data/llm_labels/sample.jsonl \
    --precision-threshold 0.85 --output data/weak_gated.csv

# 3. LLM label ด้วย Gemini (Stage 3) + agreement check (G2)
uv run python -m textcls.llm_label \
    --input data/preprocessed.parquet --output data/llm_labels/batch_1.jsonl \
    --model gemini-2.5-pro --category-file data/categories.json \
    --sample 5000 --bias-sampling

# 4. Merge + build train/val/test (stratified; test = ชุดมนุษย์ label)
uv run python -m textcls.dataset \
    --weak data/weak_gated.csv --llm data/llm_labels/ \
    --human-test data/human_labels.csv --out data/

# 5. Fine-tune (Stage 5) — baseline (LLM only) และ A/B (มี weak label)
uv run python -m textcls.train --train data/train.csv --val data/val.csv \
    --model phayathaibert --out models/ --with-weak
uv run python -m textcls.train --train data/train.csv --val data/val.csv \
    --model phayathaibert --out models/ --no-weak           # สำหรับ G3 A/B

# 6. Eval + A/B compare (G3) + calibration (G4)
uv run python -m textcls.evaluate --model models/with_weak/ --test data/test.csv
uv run python -m textcls.evaluate --model models/no_weak/  --test data/test.csv
uv run python -m textcls.calibrate --model models/with_weak/ --val data/val.csv

# 7. Deploy
uv run uvicorn textcls.serve:app --host 0.0.0.0 --port 8000   # HTTP API
uv run python -m textcls.predict --model models/with_weak/ \
    --input data/new_posts.csv --output results.csv           # CLI batch

# ── Test ──
uv run pytest
```

---

## Project Structure

```
text_classification/
├── pyproject.toml            # uv-managed dependencies
├── README.md
├── docs/
│   ├── interviews/           # บันทึกบทสัมภาษณ์
│   ├── ideas/                # one-pager แนวทาง
│   └── specs/                # spec ฉบับนี้
├── data/                     # ⚠ gitignored
│   ├── raw_posts.csv         # 100k โพสต์ (คอลัมน์: text, ...)
│   ├── weak_labels.csv       # ผลจาก tag rule (text, predicted_label)
│   ├── categories.json       # 18 หมวด + รายละเอียด (ใช้ใน prompt)
│   ├── llm_labels/           # ผล label จาก Gemini (*.jsonl)
│   ├── human_labels.csv      # ชุดมนุษย์ label ~1-2k (สำหรับ test)
│   ├── train.csv / val.csv / test.csv
├── models/                   # ⚠ gitignored — checkpoints
├── textcls/                  # source (src-layout เขียนตรงๆ ตาม uv init)
│   ├── __init__.py
│   ├── config.py             # constants: paths, 18 หมวด, thresholds
│   ├── preprocess.py         # Thai social: [CREP], [LAUGH], strip #, ตัด URL/@
│   ├── weak_label.py         # โหลดผล rule, คำนวณ per-tag precision (G1), gate
│   ├── llm_label.py          # Gemini labeling (Stage 3) + agreement (G2)
│   ├── dataset.py            # merge, stratified split, imbalance handling
│   ├── train.py              # fine-tune PhayaThaiBERT + CE/class-weight (G3 A/B)
│   ├── evaluate.py           # macro-F1, per-class recall, confusion matrix
│   ├── calibrate.py          # temperature scaling + threshold (G4)
│   ├── serve.py              # FastAPI: POST /classify → {category, confidence}
│   └── predict.py            # CLI batch scoring
└── tests/
    ├── test_preprocess.py
    ├── test_weak_label.py
    ├── test_dataset.py
    └── test_serve.py
```

---

## Code Style

Python + type hints ตลอด, ฟังก์ชันเล็กๆ หนึ่งหน้าที่, ชื่อตรงไปตรงมา

```python
# textcls/weak_label.py — ตัวอย่างสไตล์
from pathlib import Path
import pandas as pd

PRECISION_THRESHOLD = 0.85  # G1: tag แม่นกว่าขีดนี้ → ปลดล็อกเป็น label ฝึก

def per_tag_precision(weak: pd.DataFrame, referee: pd.DataFrame) -> pd.Series:
    """วัดความแม่นของ tag rule ต่อ tag (เทียบกับ referee/LLM)"""
    merged = weak.merge(referee, on="text", suffixes=("_rule", "_ref"))
    return merged.groupby("tag")["label_rule"].apply(
        lambda x: (x == merged.loc[x.index, "label_ref"]).mean()
    )

def gate_weak_labels(weak: pd.DataFrame, precision: pd.Series,
                     threshold: float = PRECISION_THRESHOLD) -> pd.DataFrame:
    """tag แม่น ≥ threshold → เป็น label ฝึกได้; ต่ำกว่า → routing-only (ตัดทิ้งจาก train)"""
    trusted_tags = precision[precision >= threshold].index
    return weak[weak["tag"].isin(trusted_tags)]
```

**กฎ:** type hints ทุกฟังก์ชัน · ไม่มี magic number เปล่า (ย้ายขึ้น constant) · คลาส/ไฟล์เดียวหนึ่งความรับผิดชอบ · config อยู่ที่ `config.py` เดียว

---

## Testing Strategy

- **Framework:** pytest, เก็บที่ `tests/`
- **ระดับ:** unit test เป็นหลัก (ไม่ต้อง e2e framework ใหญ่)
- **Coverage ที่ต้องมี (จุดเสี่ยงของ pipeline):**
  - `preprocess`: ตัวซ้ำ → `[CREP]`, `555` → `[LAUGH]`, ตัด `#` เก็บคำ, ตัด URL/@
  - `weak_label`: การคำนวณ per-tag precision, gate ถูกต้อง (threshold แบ่ง tag ได้ถูก)
  - `dataset`: stratified split กันคลาสหายไปจาก test, class weight ถูก
  - `serve`: POST /classify คืน `{category, confidence}` อยู่ในช่วงที่ถูกต้อง
- **ไม่ทดสอบ:** ตัว Gemini API เอง (ใช้ mock), ความแม่นของโมเดล (เป็นขั้น evaluate ไม่ใช่ unit test)

---

## Boundaries

- **Always:**
  - รัน `uv run pytest` ก่อน commit ทุกครั้ง
  - ใช้ uv เป็น venv manager เสมอ (ไม่ใช้ pip/conda เอง)
  - `data/` และ `models/` อยู่ใน `.gitignore`
  - API key (Gemini) ผ่าน env var เท่านั้น
- **Ask first:**
  - เปลี่ยนโมเดล base (PhayaThaiBERT → ตัวอื่น) หรือ provider LLM
  - เพิ่ม dependency ใหม่
  - เปลี่ยนชุด test (มนุษย์ label) หรือ threshold gate (G1/G4)
- **Never:**
  - commit secrets / API key ลง git
  - commit `data/`, `models/`, หรือ checkpoint
  - ลบ/แก้ชุดมนุษย์ label โดยไม่ได้รับอนุมัติ
  - ใส่ weak label เข้า train โดยไม่ผ่าน G1 gate

---

## Success Criteria

1. **เทรนโมเดลได้** — fine-tune PhayaThaiBERT สำเร็จบน GPU, `models/` มี checkpoint (with_weak + no_weak)
2. **G1 ทำจริง** — มีรายงาน per-tag precision (ตาราง: tag, precision, gate verdict) และ `weak_gated.csv` ที่ผ่าน gate
3. **G2 ทำจริง** — มีรายงาน agreement ของ Gemini label (label ซ้ำ 2 รอบ)
4. **G3 ตอบได้** — เปรียบเทียบ macro-F1 ระหว่าง with_weak vs no_weak บนชุดมนุษย์ label แล้วบอกได้ว่า "weak label ช่วย/ถ่วง/ไม่ต่าง" พร้อมตัวเลข
5. **คุณภาพ baseline:** macro-F1 ≥ **0.70** บนชุดมนุษย์ label (สำหรับตัว LLM-only; เป็นค่าตั้งต้นที่ต้อง confirm กันอีกที)
6. **G4 ใช้งานได้** — calibration ปรับแล้ว, threshold ตั้งแล้ว, โพสต์ confidence ต่ำ ขึ้น label "low confidence"
7. **Deploy 2 ช่องทาง:**
   - CLI: `predict.py` รับ CSV → คืน CSV พร้อมคอลัมน์ category + confidence
   - API: `POST /classify` รับ text → `{"category": ..., "confidence": 0.87}`
8. **Reproducible:** `uv sync` ขึ้นโปรเจกต์ใหม่แล้วรัน pipeline ต่อได้จาก `pyproject.toml` ตัวเดียว

---

## Open Questions

- [ ] ค่า macro-F1 เป้าหมายที่ยอมรับได้จริง = ? (ใช้ 0.70 เป็น default)
- [ ] Gemini ใช้รุ่นไหน (gemini-2.5-pro / flash) + งบ API สูงสุดต่อการ label ~10-20k?
- [ ] คอลัมน์ของ `raw_posts.csv` มีชื่ออะไรบ้าง (สมมติ `text`)
- [ ] ไฟล์ `weak_labels.csv` ของ tag rule มีคอลัมน์อะไรบ้าง (สมมติ `text, predicted_label`)
- [ ] ขนาด label set สุดท้าย + สัดส่วน cap ของ weak label ใน train (สมมติ ~30-40%)
