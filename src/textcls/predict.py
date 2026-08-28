"""Task 6 — predict: classify batch (CLI) ด้วยโมเดลที่ calibrate แล้ว.

อ่าน CSV (คอลัมน์ `content`) → คำนวณ category + confidence ต่อแถว
ด้วย temperature scaling (ถ้ามี calib.json) → score ต่ำกว่า threshold = low_confidence.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from textcls.evaluate import load_calib, softmax
from textcls.infer import load_model_dir, predict_logits


def classify_probs(probs: np.ndarray, id2label: dict[int, str]) -> list[dict]:
    """argmax + confidence ต่อแถว → [{category, confidence}]."""
    pred = probs.argmax(axis=1)
    conf = probs[np.arange(len(probs)), pred]
    return [{"category": id2label[int(p)], "confidence": float(c)} for p, c in zip(pred, conf)]


def build_output(df: pd.DataFrame, rows: list[dict], threshold: float) -> pd.DataFrame:
    """ต่อคอลัมน์ category/confidence/low_confidence เข้ากับ input frame."""
    out = df.copy()
    out["category"] = [r["category"] for r in rows]
    out["confidence"] = [r["confidence"] for r in rows]
    out["low_confidence"] = out["confidence"] < threshold
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="predict batch: CSV → CSV (category, confidence)")
    p.add_argument("--model", required=True, help="checkpoint dir (มี run_config.json + calib.json)")
    p.add_argument("--input", required=True, help="input CSV (คอลัมน์ content)")
    p.add_argument("--output", required=True, help="output CSV path")
    args = p.parse_args(argv)

    model, tokenizer, id2label, _ = load_model_dir(args.model)
    calib = load_calib(args.model)
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    texts = df["content"].astype(str).tolist()
    probs = softmax(predict_logits(model, tokenizer, texts) / calib["temperature"])

    out = build_output(df, classify_probs(probs, id2label), calib["threshold"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"predict {len(out):,} rows → {args.output}")
    print(f"low confidence (<{calib['threshold']}): {int(out['low_confidence'].sum()):,}")


if __name__ == "__main__":
    main()
