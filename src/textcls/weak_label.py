"""Task 4 — weak_label: G1 per-tag precision gate.

เทียบผล tag rule (`weak_labels.csv`) กับ referee (LLM label) → per-tag precision
→ tag ที่แม่น ≥ threshold (default 0.85) ปลดล็อกเป็น label ฝึก, ต่ำกว่า → ตัดออก.

รองรับข้อมูลจริง: weak label อยู่ในคอลัมน์ `category` (พร้อม `flag=AUTO`),
normalize label เป็น lowercase/trim กัน case ต่างกัน (ข้อมูลจริง "Gambling" vs "gambling").

⚠ taxonomy: weak rule ใช้ 16 คลาสของตัวเอง ("scam", "alcohol advertising", ...)
   ขณะที่ categories.json ใช้ 18 คลาส ("fraud", "alcohol", ...) — G1 วัด precision
   ใน vocabulary ของ weak เอง; การ map ไป 18 คลาส ต้อง human-review ที่ checkpoint ก่อนเทรน.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from textcls.config import PRECISION_THRESHOLD


def load_weak(path: str | Path) -> pd.DataFrame:
    """โหลดผล tag rule → [content, predicted_label].

    ถ้าไม่มีคอลัมน์ `predicted_label` ใช้ `category` (ข้อมูลจริง),
    ตัดแถวที่ flag != AUTO (ถ้ามี) และแถวที่ label ว่าง.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "predicted_label" not in df.columns:
        df["predicted_label"] = df["category"]
    df = df[df["predicted_label"].notna() & (df["predicted_label"].astype(str).str.strip() != "")]
    if "flag" in df.columns:
        df = df[df["flag"] == "AUTO"]
    return df[["content", "predicted_label"]].reset_index(drop=True)


def load_referee(path: str | Path) -> pd.DataFrame:
    """โหลด LLM label (*.jsonl) → [content, category]."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)[["content", "category"]].reset_index(drop=True)


def per_tag_precision(weak: pd.DataFrame, referee: pd.DataFrame) -> pd.Series:
    """precision ต่อ tag (predicted label ของ rule) เทียบ referee — merge ด้วย content.

    normalize label (lowercase/trim) กัน case ต่างกัน; tag ที่ไม่มี referee ตรงกัน → NaN.
    """
    w = weak.assign(
        _content=weak["content"].astype(str).str.strip(),
        _tag=weak["predicted_label"].astype(str).str.strip().str.lower(),
    )
    r = referee.assign(
        _content=referee["content"].astype(str).str.strip(),
        _cat=referee["category"].astype(str).str.strip().str.lower(),
    )
    all_tags = pd.Index(w["_tag"].dropna().unique())
    merged = w.merge(r[["_content", "_cat"]], on="_content", how="inner")
    if merged.empty:
        return pd.Series(float("nan"), index=all_tags)  # tag ไม่มี referee → NaN
    merged["_ok"] = merged["_tag"] == merged["_cat"]
    return merged.groupby("_tag")["_ok"].mean().reindex(all_tags)


def gate_weak_labels(weak: pd.DataFrame, precision: pd.Series,
                     threshold: float = PRECISION_THRESHOLD) -> pd.DataFrame:
    """เก็บเฉพาะแถวของ tag ที่ precision ≥ threshold (NaN → ตก)."""
    trusted = set(precision[precision >= threshold].index)
    tag_n = weak["predicted_label"].astype(str).str.strip().str.lower()
    return weak[tag_n.isin(trusted)].reset_index(drop=True)


def build_report(precision: pd.Series, threshold: float = PRECISION_THRESHOLD) -> pd.DataFrame:
    """ตาราง `tag, precision, verdict` เรียง precision น้อยลง (gate/drop)."""
    report = pd.DataFrame({"tag": precision.index, "precision": precision.values})
    ok = report["precision"].fillna(-1.0) >= threshold
    report["verdict"] = ok.map({True: "gate", False: "drop"})
    return report.sort_values("precision", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="G1: per-tag precision gate ของ weak label")
    p.add_argument("--weak", required=True, help="ผล tag rule (CSV: content, predicted_label)")
    p.add_argument("--referee", required=True, help="LLM label (*.jsonl: content, category)")
    p.add_argument("--precision-threshold", type=float, default=PRECISION_THRESHOLD)
    p.add_argument("--output", required=True, help="weak_gated.csv")
    args = p.parse_args(argv)

    weak = load_weak(args.weak)
    referee = load_referee(args.referee)
    precision = per_tag_precision(weak, referee)
    gated = gate_weak_labels(weak, precision, args.precision_threshold)
    gated["source"] = "weak"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    gated.to_csv(out, index=False, encoding="utf-8-sig")
    report = build_report(precision, args.precision_threshold)
    report.to_csv(out.with_suffix(".report.csv"), index=False)

    print(f"tags: {len(precision)} · gated rows: {len(gated):,} → {out}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
