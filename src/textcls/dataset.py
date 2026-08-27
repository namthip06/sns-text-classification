"""Task 5 — dataset: merge labels + stratified split + class weight + weak cap.

รวม weak (ผ่าน G1) + LLM + human → train/val/test
- test = ชุดมนุษย์ label เท่านั้น (ตาม acceptance)
- val = stratified จาก pool ที่เหลือ (กันคลาสหายจาก val)
- weak ใน train ≤ cap (default 40%)
- class weight = inverse frequency

⚠ weak taxonomy (16 คลาสของ rule) ต้อง map ไป 18 หมวดของ categories.json —
  alias default อยู่ที่ WEAK_LABEL_ALIASES (ตรวจโดยมนุษย์ที่ checkpoint ก่อนเทรน),
  override ได้ผ่าน --label-map.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

WEAK_CAP = 0.4  # สัดส่วน weak label สูงสุดใน train
VAL_FRAC = 0.1
SEED = 42

# weak rule taxonomy → categories.json id (mapping ที่ยังต้อง human confirm)
WEAK_LABEL_ALIASES = {
    "scam": "fraud",
    "e-cigarettes": "ecig",
    "forged_documents": "forged_docs",
    "kratom_drink": "kratom",
    "royal_institution_&_royal_family": "royal",
    "alcohol_advertising": "alcohol",
    "copyright_infringement": "copyright",
}


def _norm_label(label) -> str:
    return str(label).lower().strip().replace(" ", "_")


def merge_labels(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """รวม frame (content, label, source) — content ซ้ำ ให้ human > llm > weak."""
    priority = {"human": 0, "llm": 1, "weak": 2}
    df = pd.concat(frames)
    df["_p"] = df["source"].map(priority).fillna(9)
    df = df.sort_values("_p").drop_duplicates(subset="content", keep="first")
    return df[["content", "label", "source"]].reset_index(drop=True)


def map_labels(df: pd.DataFrame, category_ids: list[str], label_map: dict | None = None) -> pd.DataFrame:
    """map คอลัมน์ label ไป taxonomy 18 หมวด; label ที่ map ไม่ได้ → ตัดทิ้ง + warn."""
    alias = {**WEAK_LABEL_ALIASES, **(label_map or {})}
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


def cap_weak_fraction(train: pd.DataFrame, cap: float = WEAK_CAP, seed: int = SEED) -> pd.DataFrame:
    """ลด weak ใน train ให้สัดส่วน ≤ cap (weak ≤ cap/(1-cap) × จำนวน non-weak)."""
    non_weak = train[train["source"] != "weak"]
    max_weak = int((cap / (1 - cap)) * len(non_weak))
    weak = train[train["source"] == "weak"]
    if len(weak) <= max_weak:
        return train.reset_index(drop=True)
    kept = weak.sample(n=max_weak, random_state=seed)
    return pd.concat([non_weak, kept]).reset_index(drop=True)


def class_weights(label_series: pd.Series, classes: list[str]) -> dict:
    """Inverse-frequency weight (normalized ให้ผลรวม = 1); คลาสไม่มีข้อมูล → 0."""
    counts = label_series.value_counts()
    n = len(label_series)
    k = len(classes)
    w = {c: (n / (k * counts.get(c, 0))) if counts.get(c, 0) else 0.0 for c in classes}
    total = sum(w.values())
    return {c: v / total for c, v in w.items()} if total else w


def load_weak_gated(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"predicted_label": "label"})
    if "source" not in df.columns:
        df["source"] = "weak"
    return df[["content", "label", "source"]]


def load_llm_labels(llm_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(llm_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows.append({"content": d["content"], "label": d["category"], "source": "llm"})
    return pd.DataFrame(rows, columns=["content", "label", "source"])


def load_human(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source"] = "human"
    return df[["content", "label", "source"]]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="merge + stratified split + class weight")
    p.add_argument("--weak", required=True, help="weak_gated.csv (จาก Task 4)")
    p.add_argument("--llm", required=True, help="dir ของ llm_labels/*.jsonl (จาก Task 3)")
    p.add_argument("--human-test", required=True, help="human_labels.csv (ชุดมนุษย์ label)")
    p.add_argument("--categories", required=True, help="categories.json (18 หมวด)")
    p.add_argument("--out", required=True, help="output dir (train/val/test.csv)")
    p.add_argument("--label-map", default=None, help="JSON override weak→category id")
    p.add_argument("--cap-weak", type=float, default=WEAK_CAP)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    categories = json.loads(Path(args.categories).read_text(encoding="utf-8"))
    ids = [c["id"] for c in categories]
    label_map = json.loads(Path(args.label_map).read_text()) if args.label_map else None

    df = merge_labels([load_weak_gated(args.weak), load_llm_labels(args.llm), load_human(args.human_test)])
    df = map_labels(df, ids, label_map)

    test = df[df["source"] == "human"].copy()
    trainval = df[df["source"] != "human"]
    train, val = train_val_split(trainval, args.val_frac, args.seed)
    train = cap_weak_fraction(train, args.cap_weak, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(out / "val.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out / "test.csv", index=False, encoding="utf-8-sig")
    (out / "class_weights.json").write_text(
        json.dumps(class_weights(train["label"], ids), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"train={len(train):,} val={len(val):,} test={len(test):,} → {out}/")
    print(f"weak ใน train = {(train['source']=='weak').mean():.0%} (cap {args.cap_weak:.0%})")


if __name__ == "__main__":
    main()
