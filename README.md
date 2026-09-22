# Thai Twitter Text Classification — 19 หมวด

Classifier ข้อความทวิตเตอร์ภาษาไทยเป็น **19 หมวด (single-label — 18 หมวดเนื้อหา + `no_match`)**
พร้อม confidence เทรนจาก **weak label (tag rule) ทั้งหมด** โดยตรง — ไม่ใช้ LLM และไม่มีชุดมนุษย์
label:

```
weak_labels.csv → clean text → map taxonomy (weak 16 → 19 หมวด) → stratified split
→ fine-tune WangchanBERTa → calibrate (temperature scaling) → predict + confidence
```

> [!NOTE]
> **Decision (2026-08-27):** ยกเลิก LLM ออกจาก pipeline — เดิม `llm_label.py` (Gemini
> label/G2) และ G1 per-tag precision gate (referee = LLM) ถูกถอด เพราะ weak labels มี
> label + confidence อยู่แล้ว และไม่มี referee/มนุษย์ label มาเป็น ground truth
>
> **Decision (2026-09-22):** ถอด G4 (confidence threshold 0.6) ออก — `no_match`
> เปลี่ยนเป็น **คลาสเทรนที่ 19** ใน `categories.json` (predict = argmax 19 คลาส ไม่มี
> flag `low_confidence` แล้ว) · temperature scaling ยังอยู่ (ปรับค่า confidence
> เท่านั้น ไม่ตัดสิน label)

สถานะปัจจุบัน + รายละเอียดเริ่มจาก **[CLAUDE.md](./CLAUDE.md)** · ดีไซน์เต็มดูที่
**[docs/specs/text-classification-pipeline.md](./docs/specs/text-classification-pipeline.md)**
· โครงสร้างคอลัมน์ทุกไฟล์ที่ **[docs/notes/data-contract.md](./docs/notes/data-contract.md)**

## สารบัญ

- [ภาพรวม pipeline](#ภาพรวม-pipeline)
- [หมวดทั้ง 19](#หมวดทั้ง-19)
- [เทคโนโลยี](#เทคโนโลยี)
- [ความต้องการระบบ](#ความต้องการระบบ)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [เริ่มต้นใช้งาน](#เริ่มต้นใช้งาน)
- [รันทีละสเตจ](#รันทีละสเตจ)
  - [T3 — Dataset](#t3--dataset)
  - [T4 — Train](#t4--train)
  - [T5 — Evaluate (G3) + Calibrate](#t5--evaluate-g3--calibrate)
  - [T6 — Predict (CLI)](#t6--predict-cli)
- [Monitoring ตอนเทรน](#monitoring-ตอนเทรน)
- [Smoke test — พิสูจน์ pipeline](#smoke-test--พิสูจน์-pipeline)
- [Test](#test)
- [ข้อจำกัดที่ควรรู้](#ข้อจำกัดที่ควรรู้)
- [แผนงานที่เหลือ](#แผนงานที่เหลือ)

## ภาพรวม pipeline

```mermaid
flowchart LR
    WL["data/weak_labels.csv"] -->|"T3 dataset"| SPLIT["merged/train.csv + val.csv"]
    SPLIT -->|"T4 train"| M["models/"]
    M -->|"T5"| G3["evaluate — G3 macro-F1"]
    M -->|"T5"| CAL["calibrate — temperature scaling"]
    G3 --> P["predict"]
    CAL -. "calib.json" .-> P
    RAW["data/raw_posts.csv (258k)"] -.->|"T2 preprocess — เก็บไว้ก่อน<br>ไว้คราวหน้า เมื่อต้องการ ground truth"| WL
```

Gates:
- **G3** — ประเมิน macro-F1 บน `val` ที่ hold-out จาก weak (agreement กับ weak rule —
  ไม่ใช่ความจริงสัมบูรณ์)

## หมวดทั้ง 19

รายละเอียดเต็ม (คำอธิบาย + ตัวอย่าง) อยู่ที่ `data/categories.json` — 18 หมวดเนื้อหา
fraud · ecig · forged_docs · kratom · royal · alcohol · copyright · … + `no_match`
(ไม่เข้าหมวดใด)

> [!WARNING]
> `religion` / `child_sexual_content` / `no_match` **ยังไม่มี weak source** → ไม่มี
> train data (class weight = 0) — ต้องมี weak row ของหมวดนั้นก่อนจึงจะเทรนได้จริง

## เทคโนโลยี

- Python ≥ 3.12 + [uv](https://docs.astral.sh/uv/)
- pandas / pyarrow / scikit-learn
- PyTorch + Hugging Face `transformers` (v5)
- TensorBoard — monitoring ตอนเทรน
- โมเดล: `airesearch/wangchanberta-base-att-spm-uncased` (default, เปลี่ยนเป็น
  PhayaThaiBERT ได้ที่ `--model`)
- deps เสริมที่ใช้ตอน runtime: `openpyxl` (อ่าน .xlsx), `tqdm` (progress bar
  interactive), `sentencepiece`/`protobuf`/`tiktoken` (สกัด tokenizer WangchanBERTa)

## ความต้องการระบบ

- Python 3.12+ (จัดการด้วย [uv](https://docs.astral.sh/uv/) เสมอ — ไม่ใช้ pip/conda เอง)
- GPU (CUDA) สำหรับ T4 train — สเตจอื่นรันบน CPU ได้
- ไฟล์ข้อมูล (gitignored — ต้องเตรียมเอง):
  - `data/weak_labels.csv` — weak label จริง (คอลัมน์ `content, flag, category, …`)
  - `data/categories.json` — นิยาม 19 หมวด
  - `configs/weak_label_map.json` — mapping weak taxonomy → หมวด (มนุษย์แก้ได้)

## โครงสร้างโปรเจกต์

```
src/textcls/
  config.py      # ทุก path รวมที่เดียว
  preprocess.py  # T2 clean ภาษาไทย (ยังไม่ใช้ เก็บไว้ก่อน)
  dataset.py     # T3 clean + map weak→19 + stratified split + class weight
  train.py       # T4 fine-tune
  infer.py       # shared: โหลดโมเดล + predict logits
  evaluate.py    # T5 G3 macro-F1
  calibrate.py   # T5 temperature scaling
  predict.py     # T6 predict batch CLI (csv/xlsx + interactive)
  serve.py       # (ยังไม่สร้าง — เลื่อน)
configs/
  weak_label_map.json   # weak taxonomy → หมวดใน categories.json (มนุษย์แก้ได้)
data/                   # (gitignored) ข้อมูลจริง + outputs
models/                 # (gitignored) checkpoints
docs/
  specs/          # spec ดีไซน์ pipeline
  plans/          # implementation plan
  notes/          # data contract — โครงสร้างคอลัมน์ทุกไฟล์
  ideas/ interviews/  # แนวคิดต้นทาง
notebooks/ scripts/     # งานสำรวจ/ยูทิล
tests/                  # pytest
```

## เริ่มต้นใช้งาน

```bash
uv sync                # ติดตั้ง dependencies
uv run pytest          # ตรวจว่าทุกอย่างผ่าน
```

แผนผังคอลัมน์ทุกไฟล์ (input/output ของแต่ละสเตจ) ดูที่
[`docs/notes/data-contract.md`](./docs/notes/data-contract.md)

## รันทีละสเตจ

### T3 — Dataset

Clean text + map taxonomy (weak 16 → 19 หมวด) + stratified split + class weight:

```bash
uv run python -m textcls.dataset \
  --weak data/weak_labels.csv \
  --categories data/categories.json \
  --out data/
```

| Option | Default | คำอธิบาย |
|--------|---------|----------|
| `--weak` | (บังคับ) | `weak_labels.csv` (คอลัมน์ `content, flag, category`) |
| `--categories` | (บังคับ) | `categories.json` (19 หมวด) |
| `--out` | (บังคับ) | โฟลเดอร์ output |
| `--val-frac` | จาก config | สัดส่วน val ใน stratified split |
| `--seed` | จาก config | seed ของการ split |

Output: `data/{merged,train,val}.csv` + `class_weights.json`

### T4 — Train

```bash
uv run python -m textcls.train \
  --train data/train.csv --val data/val.csv \
  --categories data/categories.json --out models/
```

| Option | Default | คำอธิบาย |
|--------|---------|----------|
| `--train` / `--val` | (บังคับ) | `train.csv` / `val.csv` (คอลัมน์ `content, label`) |
| `--categories` | (บังคับ) | `categories.json` |
| `--model` | WangchanBERTa | HF model id (เช่น PhayaThaiBERT) |
| `--out` | `models` | โฟลเดอร์ root ของ checkpoint |
| `--tag` | `model` | ชื่อโฟลเดอร์ checkpoint (`models/<tag>/`) |
| `--epochs` | 3 | จำนวน epoch |
| `--lr` | 2e-5 | learning rate |
| `--batch-size` | 16 | batch size |
| `--seed` | 42 | seed |
| `--logging-steps` | 50 | ความถี่ log ไป TensorBoard |

Output: `models/<tag>/` — checkpoint + `run_config.json` (บันทึก
model/seed/hyperparams ทั้งหมด เอาไว้เทียบรันอื่น) + `metrics.json` + `runs/`
(TensorBoard)

### T5 — Evaluate (G3) + Calibrate

```bash
uv run python -m textcls.evaluate --model models/model/ --test data/val.csv
uv run python -m textcls.calibrate --model models/model/ --val data/val.csv
```

- **evaluate** → macro-F1 บน `--test` (G3) — ดู agreement กับ weak rule
- **calibrate** → fit temperature `T` บน `--val` เขียน `calib.json` ลงโฟลเดอร์โมเดล
  — predict ใช้ปรับค่า confidence (ไม่ตัดสิน label)

ทั้งสองคำสั่งรับแค่ `--model` (โฟลเดอร์ checkpoint) + ไฟล์ test/val เท่านั้น

### T6 — Predict (CLI)

```bash
# แบบ pipeline (non-interactive): ระบุคอลัมน์ข้อความเอง
uv run python -m textcls.predict \
  --model models/model/ --input in.csv --output out.csv --text-column content

# แบบ interactive: เลือก model (จาก models/) + sheet/คอลัมน์ใน terminal,
# preview 5 แถว + ยืนยัน, progress bar + bar chart กระจาย label
# (ใช้ได้กับ .xlsx ด้วย — ต้องมี openpyxl)
uv run python -m textcls.predict --input in.xlsx
```

| Option | Default | คำอธิบาย |
|--------|---------|----------|
| `--input` | (บังคับ) | ไฟล์ `.csv` หรือ `.xlsx` |
| `--model` | (interactive: เลือกจาก `models/`) | checkpoint dir ที่มี `run_config.json` — โหมด non-interactive ต้องระบุเอง |
| `--output` | `<input>_predicted.<ext>` | path ไฟล์ผลลัพธ์ (ต้นฉบับไม่ถูกแก้) |
| `--text-column` | (interactive: เลือกใน terminal) | คอลัมน์ข้อความ — การระบุ option นี้ = โหมดเงียบ |
| `--no-header` | ปิด | ไฟล์ไม่มีแถว header (คอลัมน์ชื่อ `col_1..col_N`) |
| `--batch-size` | 32 | batch size ตอน infer |

พฤติกรรม:
- ไม่ระบุ `--text-column` = **interactive** — เลือก model + sheet/คอลัมน์, preview
  5 แถว + ยืนยันก่อนรัน, progress bar + bar chart กระจาย label ตอนจบ
- ไฟล์ `--no-header` → output ก็ไม่มี header เหมือน input (non-interactive ระบุ
  `--text-column col_1`)
- แถวที่คอลัมน์ข้อความว่าง → `category`/`confidence` เว้นว่าง
- score ผ่าน temperature scaling จาก `calib.json` **เสมอ** (ไม่มีไฟล์ → T=1.0)

Output: ไฟล์เดิม + คอลัมน์ `category` + `confidence`

(`serve.py` API ยังเลื่อน — ถ้าต้องการค่อยสร้าง)

## Monitoring ตอนเทรน

- `models/<tag>/metrics.json` — eval loss + macro-F1 ต่อ epoch (macro-F1 คิดแบบเดียว
  กับ G3: เฉพาะคลาสที่มีใน val) ผ่าน `MetricsCallback`
- TensorBoard — loss/learning_rate ทุก `--logging-steps`:

```bash
uv run tensorboard --logdir models/model/runs
```

## Smoke test — พิสูจน์ pipeline

Smoke ครบ chain บน GPU: train 736 แถว (`models/model/`) → calibrate (T≈0.22) →
evaluate (macro-F1 ≈ 0.48) → predict — **ยังไม่เทรนเต็ม 128k**

## Test

```bash
uv run pytest
```

42 tests — ครอบคลุม config, preprocess, dataset, train (mock tokenizer), evaluate,
calibrate, predict · เน้น unit test ไม่โหลดโมเดลจริง/ไม่เรียก API · รันก่อน commit
ทุกครั้ง

## ข้อจำกัดที่ควรรู้

- **G3 วัด agreement กับ weak rule ไม่ใช่ความจริงสัมบูรณ์** — val hold-out มาจาก weak
  label เดียวกับ train ถ้า rule ผิด ตัวเลขก็ผิดตาม
- 3 หมวด (`religion`, `child_sexual_content`, `no_match`) ยังไม่มี weak source →
  argmax จะไม่ออกหมวดนี้จนกว่าจะมี train data
- mapping weak→หมวด (`configs/weak_label_map.json`) ยังใช้ default — ยังรอ confirm
- เทรนเต็ม 128k แถวยังไม่เคยรัน (ตัวเลข smoke มาจาก 736 แถว)

## แผนงานที่เหลือ

| Task | งาน | สถานะ |
|------|-----|-------|
| 1 | scaffold (uv + config) | ✅ |
| 2 | preprocess (จาก raw_posts 258k) | ⏳ deferred — เปิดใช้เมื่อต้องการ ground truth (ถามก่อน) |
| 3 | dataset (clean + map weak→19 + split) | ✅ |
| 4 | train | ✅ |
| 5 | evaluate (G3) + calibrate | ✅ |
| 6 | predict CLI | ✅ |
| — | serve (API) | ⏳ เลื่อน |
| — | เทรนเต็ม 128k แถว | ⏳ ยังไม่รัน |
