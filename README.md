# Thai Twitter Text Classification — 18 หมวด

Classifier ข้อความทวิตเตอร์ภาษาไทยเป็น 18 หมวด (single-label) พร้อม confidence
โดยใช้ **weak supervision**: 100k+ โพสต์ไม่มี label → LLM (Gemini) กำกับ label
+ tag rule (weak) → กรองด้วย per-tag precision (G1) → merge → fine-tune
WangchanBERTa/PhayaThaiBERT (A/B: กับ weak / ไม่มี weak) → calibrate + threshold → deploy.

สถานะปัจจุบันดูที่ **[SESSION.md](./SESSION.md)** · ดีไซน์เต็มดูที่
**[docs/specs/text-classification-pipeline.md](./docs/specs/text-classification-pipeline.md)**

## สารบัญ

- [ภาพรวม pipeline](#ภาพรวม-pipeline)
- [เทคโนโลยี](#เทคโนโลยี)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [เริ่มต้นใช้งาน](#เริ่มต้นใช้งาน)
- [รันทีละสเตจ](#รันทีละสเตจ)
  - [T2 — Preprocess](#t2--preprocess)
  - [T3 — LLM label (Gemini) + G2](#t3--llm-label-gemini--g2)
  - [T4 — Weak gate (G1)](#t4--weak-gate-g1)
  - [T5 — Dataset (merge + split)](#t5--dataset-merge--split)
  - [T6 — Train (A/B)](#t6--train-ab)
- [Smoke test — พิสูจน์ pipeline](#smoke-test--พิสูจน์-pipeline)
- [Test](#test)
- [แผนงานที่เหลือ](#แผนงานที่เหลือ)

## ภาพรวม pipeline

```
raw posts ──T2──▶ preprocessed.parquet ──T3──▶ llm_labels/*.jsonl ──┐
                                            (Gemini + G2)          ├─T5──▶ train/val/test.csv ──T6▶ เทรน A/B
weak_labels ──────────────────T4──▶ weak_gated.csv ────────────────┘    └─▶ G3 eval · G4 calibrate+threshold
                                    (G1: per-tag precision)
```

Gates:
- **G1** — weak label แต่ละ tag ต้องมี precision ≥ 0.85 (วัดเทียบ LLM referee)
- **G2** — LLM ซ้ำ ~10% เพื่อวัด agreement rate
- **G3** — A/B เทียบ macro-F1 (กับ weak vs ไม่มี weak)
- **G4** — temperature scaling + threshold 0.6 → ผล confidence ต่ำ = `low confidence`

## เทคโนโลยี

- Python ≥ 3.12 + [uv](https://docs.astral.sh/uv/)
- pandas / pyarrow / scikit-learn
- PyTorch + Hugging Face `transformers`
- โมเดล: `airesearch/wangchanberta-base-att-spm-uncased` (default, ตั้งเป็น
  PhayaThaiBERT ได้ที่ `--model`)
- LLM labeling: `google-genai` (Gemini)

## โครงสร้างโปรเจกต์

```
src/textcls/
  config.py      # paths, GEMINI_API_KEY_ENV, thresholds
  preprocess.py  # T2 clean ภาษาไทย (URL, @, #, 555, ซ้ำ)
  llm_label.py   # T3 Gemini label + G2 agreement
  weak_label.py  # T4 G1 per-tag precision gate
  dataset.py     # T5 merge + stratified split + class weight + weak cap
  train.py       # T6 fine-tune + A/B
configs/
  weak_label_map.json   # weak taxonomy → 18 หมวด (มนุษย์แก้ได้)
data/                   # (gitignored) ข้อมูลจริง + outputs
models/                 # (gitignored) checkpoints
scripts/
  build_smoke_dataset.py   # สร้าง smoke set 100 ประโยค
docs/specs/               # spec · docs/plans/ แผน · docs/notes/ data contract
tests/                    # pytest
notebooks/                # EDA
```

## เริ่มต้นใช้งาน

```bash
uv sync                # ติดตั้ง dependencies
uv run pytest          # ตรวจว่าทุกอย่างผ่าน
```

ต้องมี Gemini API key สำหรับ T3 (ตั้งค่า env ชื่อตาม `config.py`):

```bash
export GEMINI_API_KEY="..."
```

ข้อมูล: `data/categories.json` (18 หมวด) + `data/weak_labels_auto.csv` (weak label จริง,
คอลัมน์ `content, flag, category, …`) · แผนผังคอลัมน์ทุกไฟล์ดู
[`docs/notes/data-contract.md`](./docs/notes/data-contract.md)

## รันทีละสเตจ

### T2 — Preprocess

ล้างข้อความ (URL/mention/hashtag → ช่องว่าง, `5555…`→`5`, อักษรซ้ำ → ตัวเดียว):

```bash
uv run python -m textcls.preprocess \
  --input data/alltime_25_26_content_dedup.csv \
  --output data/preprocessed.parquet
```

Output: `data/preprocessed.parquet` (คอลัมน์ `content_clean`)

### T3 — LLM label (Gemini) + G2

Label sample + ซ้ำ `--dup-fraction` เพื่อวัด G2:

```bash
uv run python -m textcls.llm_label \
  --input data/preprocessed.parquet \
  --output data/llm_labels/batch1.jsonl \
  --category-file data/categories.json \
  --sample 10000 --dup-fraction 0.1 --seed 42
```

- `--bias-sampling` = โอเวอร์แซมป์คลาสหายาก (ถ้า input มีคอลัมน์ `pred_label`)
- Output: `batch1.jsonl` (`content, category, confidence`) + `agreement.json`

### T4 — Weak gate (G1)

กรอง weak label ทิ้ง tag ที่ precision ต่ำกว่า threshold (default 0.85):

```bash
uv run python -m textcls.weak_label \
  --weak data/weak_labels_auto.csv \
  --referee data/llm_labels/batch1.jsonl \
  --output data/weak_gated.csv
```

Output: `weak_gated.csv` + `weak_gated.report.csv` (tag / precision / verdict)

### T5 — Dataset (merge + split)

รวม weak + LLM + human → split แบบ stratified; weak ใน train ถูกจำกัด ≤ 40%:

```bash
uv run python -m textcls.dataset \
  --weak data/weak_gated.csv \
  --llm data/llm_labels \
  --human-test data/human_labels.csv \
  --categories data/categories.json \
  --out data/
```

Output: `data/{train,val,test}.csv` + `class_weights.json`
(`--label-map` override mapping weak→18 หมวด ได้; default จาก `configs/weak_label_map.json`)

### T6 — Train (A/B)

```bash
# A: กับ weak (default)
uv run python -m textcls.train \
  --train data/train.csv --val data/val.csv \
  --categories data/categories.json --out models/

# B: LLM อย่างเดียว (baseline)
uv run python -m textcls.train \
  --train data/train.csv --val data/val.csv \
  --categories data/categories.json --out models/ --no-weak
```

Output: `models/{with_weak,no_weak}/` (checkpoint + `run_config.json` บันทึก
model/tag/seed/hyperparams)

## Smoke test — พิสูจน์ pipeline

ข้อมูลจริงยังไม่ครบ (llm_labels/human_labels) จึงเทรนพิสูจน์บน **100 ประโยค**
จาก weak labels จริงก่อน:

```bash
uv run python scripts/build_smoke_dataset.py   # สร้าง data/smoke/{train,val}.csv
uv run python -m textcls.train \
  --train data/smoke/train.csv --val data/smoke/val.csv \
  --categories data/categories.json --out models/smoke
```

Output โดยประมาณ (RTX 4050 6GB, ~8 it/s, 10 steps):

```
train=80 val=20
...
{'loss': 2.457, 'eval_loss': 2.147, 'eval_runtime': 4.12, ...}
done → models/smoke/with_weak/ (val loss ใน log)
```

พิสูจน์ว่า pipeline เทรน–เซฟ–โหลด–predict ครบวงจรบน GPU ได้จริง

## Test

```bash
uv run pytest
```

50 tests — ครอบคลุม config, preprocess, llm_label (mock Gemini), weak_label,
dataset, train (mock tokenizer) · เน้น unit test ไม่โหลดโมเดลจริง/ไม่เรียก API

## แผนงานที่เหลือ

| Task | งาน | สถานะ |
|------|-----|-------|
| 1 | scaffold (uv + config) | ✅ |
| 2 | preprocess | ✅ |
| 3 | llm_label + G2 | ✅ |
| 4 | weak_label + G1 | ✅ |
| 5 | dataset | ✅ |
| 6 | train + A/B | ✅ (smoke เท่านั้น) |
| 7 | evaluate (G3) + calibrate (G4) | ⏳ รอข้อมูลจริง |
| 8 | serve (FastAPI) + predict (CLI) | ⏳ |

Blockers: `data/llm_labels/` ยังไม่รัน Gemini (ต้อง API key) ·
`data/human_labels.csv` ยังไม่มี · mapping weak→18 หมวดยังไม่ confirm —
ดูรายละเอียดใน **[SESSION.md](./SESSION.md)**
