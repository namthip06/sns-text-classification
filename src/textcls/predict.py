"""Task 6 — predict: classify batch (CLI) ด้วยโมเดลที่ calibrate แล้ว.

อ่าน CSV/Excel → คำนวณ category + confidence ต่อแถวด้วย temperature scaling
(ถ้ามี calib.json) → score ต่ำกว่า threshold = low_confidence.

ทำงาน 2 โหมด:
- ระบุ --text-column → ทำงานเงียบ (pipeline / test) เหมือนเดิม
- ไม่ระบุ → interactive: เลือก sheet/คอลัมน์ใน terminal, preview 5 แถว, ยืนยัน,
  progress bar + bar chart กระจาย label, เขียนไฟล์ `<ชื่อเดิม>_predicted.<ext>`

แถวที่คอลัมน์ข้อความว่าง → category/confidence เว้นว่าง (ไม่ predict).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from textcls.config import MODEL_DIR
from textcls.evaluate import load_calib, softmax
from textcls.infer import load_model_dir, predict_logits

XLSX_EXTS = (".xlsx", ".xls")


def choose_model(models_dir: Path) -> Path:
    """interactive: สแกน models/ หา checkpoint dir (มี run_config.json) → ให้ผู้ใช้เลือก."""
    found = sorted(d for d in models_dir.iterdir() if d.is_dir() and (d / "run_config.json").exists())
    if not found:
        raise SystemExit(f"ไม่พบโมเดลใน {models_dir}/ (ต้องมี run_config.json) — หรือระบุ --model เอง")
    if len(found) == 1:
        print(f"ใช้โมเดล: {found[0]}")
        return found[0]
    return found[pick("เลือกโมเดล", [str(d) for d in found])]


def pick(prompt: str, options: list) -> int:
    """ถามผู้ใช้เลือก 1 อย่างจาก options ด้วยหมายเลข → index."""
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        try:
            n = int(input(f"{prompt} [1-{len(options)}]: "))
            if 1 <= n <= len(options):
                return n - 1
        except ValueError:
            pass


def read_input(path: Path, interactive: bool, no_header: bool = False) -> pd.DataFrame:
    """อ่าน .csv หรือ .xlsx (หลาย sheet + interactive → ให้เลือก sheet).

    no_header=True → ทุกบรรทัดเป็นข้อมูล ตั้งชื่อคอลัมน์เองเป็น col_1..col_N.
    """
    if path.suffix.lower() in XLSX_EXTS:
        sheets = pd.ExcelFile(path).sheet_names
        name = sheets[pick("เลือก sheet", sheets)] if len(sheets) > 1 and interactive else sheets[0]
        df = pd.read_excel(path, sheet_name=name, header=None if no_header else 0)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", header=None if no_header else "infer")
    if no_header:
        df.columns = [f"col_{i + 1}" for i in range(len(df.columns))]
    return df


def choose_column(df: pd.DataFrame) -> str:
    """interactive: ให้ผู้ใช้เลือกคอลัมน์ข้อความ (ไม่ hardcode ชื่อ)."""
    return str(df.columns[pick("เลือกคอลัมน์ข้อความ", list(df.columns))])


def confirm_preview(df: pd.DataFrame, col: str) -> bool:
    """interactive: preview ข้อความ 5 แถวแรก → ยืนยันก่อนรันจริง."""
    print(f"\n--- preview '{col}' (5 แถวแรก) ---")
    for i, t in enumerate(df[col].head(5).fillna(""), 1):
        print(f"{i}. {str(t)[:120]}")
    return input("\nรันจริง? [y/N]: ").strip().lower() == "y"


def classify_probs(probs: np.ndarray, id2label: dict[int, str]) -> list[dict]:
    """argmax + confidence ต่อแถว → [{category, confidence}]; แถว NaN = ว่าง (ไม่ predict)."""
    rows = []
    for p in probs:
        if np.isnan(p).any():
            rows.append({"category": "", "confidence": float("nan")})
            continue
        i = int(p.argmax())
        rows.append({"category": id2label[i], "confidence": float(p[i])})
    return rows


def build_output(df: pd.DataFrame, rows: list[dict], threshold: float) -> pd.DataFrame:
    """ต่อคอลัมน์ category/confidence/low_confidence เข้ากับ input frame."""
    out = df.copy()
    out["category"] = [r["category"] for r in rows]
    out["confidence"] = [r["confidence"] for r in rows]
    out["low_confidence"] = out["confidence"] < threshold  # NaN (แถวว่าง) → False
    return out


def bar_chart(counts: pd.Series, total: int, width: int = 30) -> str:
    """bar chart แนวนอนของการกระจาย label (เรียงมาก→น้อย)."""
    label_w = max(len(str(c)) for c in counts.index)
    lines = ["\n=== การกระจาย label ==="]
    for lab, n in counts.items():
        bar = "█" * max(1, round(n / counts.max() * width))
        lines.append(f"{lab:<{label_w}} │{bar} {n} ({n / total:.1%})")
    return "\n".join(lines)


def write_output(out: pd.DataFrame, src: Path, output: str | None,
                 no_header: bool = False) -> Path:
    """เขียนไฟล์ผลลัพธ์ (นามสกุลเดียวกับ input) — ต้นฉบับไม่ถูกแก้.

    no_header=True → เขียนแบบไม่มีแถว header เหมือน input เดิม.
    """
    dest = Path(output) if output else src.with_name(f"{src.stem}_predicted{src.suffix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() in XLSX_EXTS:
        out.to_excel(dest, index=False, header=not no_header)
    else:
        out.to_csv(dest, index=False, header=not no_header, encoding="utf-8-sig")
    return dest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="predict batch: CSV/Excel → CSV/Excel (category, confidence)")
    p.add_argument("--model", help="checkpoint dir (มี run_config.json + calib.json); ไม่ระบุ = เลือกจาก models/ (interactive เท่านั้น)")
    p.add_argument("--input", required=True, help="input .csv หรือ .xlsx")
    p.add_argument("--output", help="output path (default: <input>_predicted.<ext>)")
    p.add_argument("--text-column", help="คอลัมน์ข้อความ (ไม่ระบุ = interactive เลือกใน terminal)")
    p.add_argument("--no-header", action="store_true",
                   help="ไฟล์ไม่มีแถว header — ทุกบรรทัดเป็นข้อมูล (คอลัมน์ชื่อ col_1..col_N)")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"ไม่พบไฟล์: {src}")
    interactive = args.text_column is None
    if args.model:
        model_dir = Path(args.model)
    elif interactive:
        model_dir = choose_model(MODEL_DIR)
    else:
        raise SystemExit("โหมดเงียบ (--text-column) ต้องระบุ --model")
    df = read_input(src, interactive, no_header=args.no_header)
    print(f"โหลด {len(df):,} แถว, {len(df.columns)} คอลัมน์")

    if interactive:
        col = choose_column(df)
        if not confirm_preview(df, col):
            raise SystemExit("ยกเลิก")
    else:
        col = args.text_column
        if col not in df.columns:
            raise SystemExit(f"ไม่มีคอลัมน์ '{col}' ในไฟล์ (มีแค่: {', '.join(df.columns)})")

    model, tokenizer, id2label, _ = load_model_dir(model_dir)
    calib = load_calib(model_dir)

    texts = df[col].fillna("").astype(str).str.strip()
    mask = (texts != "").to_numpy()  # แถวว่าง → เว้นว่างในผลลัพธ์ ไม่ predict
    probs = np.full((len(df), len(id2label)), np.nan)
    if mask.any():
        logits = predict_logits(model, tokenizer, texts[mask].tolist(),
                                batch_size=args.batch_size, progress=interactive)
        probs[mask] = softmax(logits / calib["temperature"])

    out = build_output(df, classify_probs(probs, id2label), calib["threshold"])

    if interactive:
        counts = out.loc[mask, "category"].value_counts()
        print(bar_chart(counts, total=int(mask.sum())))

    dest = write_output(out, src, args.output, no_header=args.no_header)
    print(f"predict {int(mask.sum()):,}/{len(out):,} แถว → {dest}")
    print(f"low confidence (<{calib['threshold']}): {int(out['low_confidence'].sum()):,}")


if __name__ == "__main__":
    main()
