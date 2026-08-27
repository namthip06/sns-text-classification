"""Task 2 — preprocess ข้อมูลดิบ → preprocessed.parquet.

แปลง Thai Twitter text: ตัด URL/@mention, `555`→[LAUGH], ตัวซ้ำ→[CREP],
ตัด `#` เก็บคำไว้ (ตาม spec Testing Strategy "ตัด # เก็บคำ").
"""

import re
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w-]+")
HASHTAG_RE = re.compile(r"#(\S+)")  # ตัด # แต่เก็บคำ
LAUGH_RE = re.compile(r"5{3,}")  # 555… → [LAUGH]
CREP_RE = re.compile(r"(\S)\1{2,}")  # ตัวซ้ำ 3+ → เก็บตัวเดียว + [CREP] (ไม่จับ whitespace)
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """ทำความสะอาดข้อความเดียว ตามลำดับ: URL → @ → # → 555 → ตัวซ้ำ → whitespace."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = LAUGH_RE.sub("[LAUGH]", text)
    text = CREP_RE.sub(r"\1[CREP]", text)
    return WS_RE.sub(" ", text).strip()


def preprocess(input_csv: Path, output_parquet: Path) -> int:
    """อ่าน CSV (คอลัมน์ `content`), ล้างทุกแถว, เขียน parquet คอลัมน์ `content_clean`.

    อ่านด้วย utf-8-sig เพราะข้อมูลจริงมี BOM นำหน้า (data-contract.md)
    """
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df["content_clean"] = df["content"].astype(str).apply(clean_text)
    df[["content_clean"]].to_parquet(output_parquet, index=False)
    return len(df)


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Preprocess raw Thai Twitter posts")
    p.add_argument("--input", required=True, help="raw CSV (คอลัมน์ content)")
    p.add_argument("--output", required=True, help="output parquet path")
    args = p.parse_args(argv)

    n = preprocess(Path(args.input), Path(args.output))
    print(f"preprocessed {n:,} rows -> {args.output}")


if __name__ == "__main__":
    main()
