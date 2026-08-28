"""Task 3 — dataset: weak-only → clean text + map taxonomy + stratified split + save.

อ่าน `weak_labels.csv` (tag rule) → กรอง flag==AUTO / label ว่าง → clean text
(reuse `clean_text`) → map taxonomy weak (16 คลาส) → 18 หมวดของ categories.json
ผ่าน `configs/weak_label_map.json` → stratified split (train/val) → class weight
→ save merged (ก่อน split) + train.csv + val.csv + class_weights.json

weak taxonomy (16 คลาส) ต่างจาก 18 หมวดของ categories.json —
mapping อยู่ที่ configs/weak_label_map.json (มนุษย์แก้ได้).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from textcls.preprocess import clean_text

VAL_FRAC = 0.1
SEED = 42

# weak rule taxonomy → categories.json id — human-editable ที่ configs/weak_label_map.json
DEFAULT_LABEL_MAP = Path(__file__).resolve().parents[2] / "configs" / "weak_label_map.json"


def default_label_map() -> dict:
    """mapping ปัจจุบัน — ถ้า configs/weak_label_map.json มี ให้ใช้ไฟล์นั้น (มนุษย์แก้ได้)."""
    return json.loads(DEFAULT_LABEL_MAP.read_text(encoding="utf-8"))


def _norm_label(label) -> str:
    return str(label).lower().strip().replace(" ", "_")


def load_weak(df: pd.DataFrame) -> pd.DataFrame:
    """โหลด weak (tag rule) → [content, label] — ตัด flag != AUTO และ label ว่าง.

    คอลัมน์ label: ใช้ `predicted_label` ถ้ามี ไม่มี → `category` (ข้อมูลจริง).
    """
    if "flag" in df.columns:
        df = df[df["flag"].fillna("") == "AUTO"]
    if "predicted_label" in df.columns:
        df = df.rename(columns={"predicted_label": "label"})
    elif "category" in df.columns:
        df = df.rename(columns={"category": "label"})
    else:
        raise ValueError("weak CSV ต้องมีคอลัมน์ `category` หรือ `predicted_label`")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["label"] != ""][["content", "label"]].reset_index(drop=True)


def map_labels(df: pd.DataFrame, category_ids: list[str]) -> pd.DataFrame:
    """map คอลัมน์ label ไป taxonomy 18 หมวด; label ที่ map ไม่ได้ → ตัดทิ้ง + warn."""
    alias = {_norm_label(k): v for k, v in default_label_map().items()}
    norm = df["label"].map(_norm_label)
    mapped = norm.map(lambda l: alias.get(l, l))
    valid = mapped.isin(category_ids)
    if (~valid).sum():
        print(f"warn: ตัด {int((~valid).sum()):,} แถว label ไม่ตรง taxonomy")
    out = df.copy()
    out["label"] = mapped
    return out[valid].reset_index(drop=True)


def train_val_split(df: pd.DataFrame, val_frac: float = VAL_FRAC, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split ตาม label — กันคลาสหายจาก val.

    (ponytail: คลาส <2 แถว stratified ไม่ได้ → ไป train ทั้งหมด;
     ข้อมูลเล็กเกิน (test_size < n_classes) → fallback random split)
    """
    counts = df["label"].value_counts()
    singles = set(counts[counts < 2].index)
    df_single = df[df["label"].isin(singles)]
    df_multi = df[~df["label"].isin(singles)]
    if len(df_multi) < 2:
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)
    try:
        tr_multi, va = train_test_split(df_multi, test_size=val_frac,
                                        stratify=df_multi["label"], random_state=seed)
    except ValueError:
        print("warn: stratified split เป็นไปไม่ได้ → random split")
        tr_multi, va = train_test_split(df_multi, test_size=val_frac, random_state=seed)
    tr = pd.concat([tr_multi, df_single])
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def class_weights(label_series: pd.Series, classes: list[str]) -> dict:
    """Inverse-frequency weight (normalized ให้ผลรวม = 1); คลาสไม่มีข้อมูล → 0."""
    counts = label_series.value_counts()
    n = len(label_series)
    k = len(classes)
    w = {c: (n / (k * counts.get(c, 0))) if counts.get(c, 0) else 0.0 for c in classes}
    total = sum(w.values())
    return {c: v / total for c, v in w.items()} if total else w


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="weak-only: clean + map + stratified split + save")
    p.add_argument("--weak", required=True, help="weak_labels.csv (คอลัมน์ content, flag, category)")
    p.add_argument("--categories", required=True, help="categories.json (18 หมวด)")
    p.add_argument("--out", required=True, help="output dir (merged/train/val.csv + class_weights.json)")
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    categories = json.loads(Path(args.categories).read_text(encoding="utf-8"))
    ids = [c["id"] for c in categories]

    df = load_weak(pd.read_csv(args.weak, encoding="utf-8-sig"))
    df["content"] = df["content"].astype(str).map(clean_text)
    df = df[df["content"].str.strip() != ""].reset_index(drop=True)  # content ว่าง (ล้างแล้ว) → ตัด
    merged = map_labels(df, ids)
    train, val = train_val_split(merged, args.val_frac, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "merged.csv", index=False, encoding="utf-8-sig")
    train.to_csv(out / "train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(out / "val.csv", index=False, encoding="utf-8-sig")
    (out / "class_weights.json").write_text(
        json.dumps(class_weights(train["label"], ids), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"merged={len(merged):,} train={len(train):,} val={len(val):,} → {out}/")
    missing = [c for c in ids if c not in set(merged["label"])]
    if missing:
        print(f"warn: หมวดไม่มี train data: {missing}")


if __name__ == "__main__":
    main()
