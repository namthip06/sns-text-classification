"""Task 3 — dataset: weak-only → clean + map taxonomy + stratified split + save."""

import json

import pandas as pd
import pytest

from textcls import dataset


CATEGORIES = ["gambling", "pornography", "fraud", "ecig", "prostitution", "kratom"]


def _d(rows):
    return pd.DataFrame(rows)


# ── load_weak ───────────────────────────────────────────────────────────

def test_load_weak_filters_non_auto_and_empty_label():
    df = _d({"content": ["a", "b", "c", "d", "e"],
             "flag": ["AUTO", "AUTO", "MANUAL", "AUTO", "AUTO"],
             "category": ["Gambling", "", "Pornography", None, "Scam"]})
    out = dataset.load_weak(df)
    assert out["label"].tolist() == ["Gambling", "Scam"]  # MANUAL flag + label ว่าง ตัด
    assert list(out.columns) == ["content", "label"]


def test_load_weak_prefers_predicted_label_column():
    df = _d({"content": ["a"], "flag": ["AUTO"],
             "category": ["Gambling"], "predicted_label": ["Scam"]})
    out = dataset.load_weak(df)
    assert out["label"].tolist() == ["Scam"]


def test_load_weak_missing_label_column_raises():
    df = _d({"content": ["a"], "flag": ["AUTO"]})
    with pytest.raises(ValueError):
        dataset.load_weak(df)


# ── label mapping (weak taxonomy → 18 หมวด) ─────────────────────────────

def test_map_labels_aliases_and_drop_unmapped():
    df = _d({"content": ["a", "b", "c", "d"],
             "label": ["scam", "kratom drink", "e-cigarettes", "not_a_category"]})
    out = dataset.map_labels(df, CATEGORIES)
    assert out["label"].tolist() == ["fraud", "kratom", "ecig"]  # unmapped ถูกตัด


def test_map_labels_identity_passthrough():
    df = _d({"content": ["a"], "label": ["Gambling"]})
    out = dataset.map_labels(df, CATEGORIES)
    assert out["label"].tolist() == ["gambling"]


# ── split ───────────────────────────────────────────────────────────────

def test_train_val_split_stratified_keeps_all_classes():
    rows = []
    for cls in CATEGORIES:
        rows += [{"content": f"{cls}-{i}", "label": cls} for i in range(20)]
    df = pd.DataFrame(rows)
    tr, va = dataset.train_val_split(df, val_frac=0.2, seed=42)
    assert set(va["label"]) == set(CATEGORIES)  # ไม่มีคลาสหายจาก val
    assert set(tr["label"]) == set(CATEGORIES)
    assert len(va) == pytest.approx(0.2 * len(df), rel=0.15)


# ── class weight ────────────────────────────────────────────────────────

def test_class_weights_inverse_frequency():
    labels = pd.Series(["a"] * 4 + ["b"] * 2 + ["c"] * 2)
    w = dataset.class_weights(labels, ["a", "b", "c"])
    # inverse frequency: a เยอะสุด (4/8=0.5) → weight น้อยสุด; b/c เท่ากัน
    assert w["b"] == w["c"]
    assert w["a"] < w["b"]
    assert sum(w.values()) == pytest.approx(1.0)  # normalized


def test_class_weights_empty_class_zero():
    labels = pd.Series(["a"] * 4 + ["b"] * 2)
    w = dataset.class_weights(labels, ["a", "b", "c"])
    assert w["c"] == 0.0  # คลาสไม่มีข้อมูล → 0


# ── CLI end-to-end ──────────────────────────────────────────────────────

def test_main_writes_splits_and_weights(tmp_path):
    weak = tmp_path / "weak.csv"
    cats_rows = [{"content": f"{c}-{i}", "flag": "AUTO", "category": c} for c in
                 ["Gambling", "Scam", "Kratom Drink", "E-Cigarettes", "Pornography", "Prostitution"]
                 for i in range(10)]
    _d(cats_rows).to_csv(weak, index=False, encoding="utf-8-sig")

    cats = tmp_path / "categories.json"
    cats.write_text(json.dumps([{"id": c, "name": c} for c in CATEGORIES], ensure_ascii=False))

    out = tmp_path / "out"
    dataset.main(["--weak", str(weak), "--categories", str(cats), "--out", str(out)])

    for name in ["merged.csv", "train.csv", "val.csv"]:
        assert (out / name).exists()
    assert not (out / "test.csv").exists()

    train = pd.read_csv(out / "train.csv")
    assert list(train.columns) == ["content", "label"]  # ไม่มี source
    val = pd.read_csv(out / "val.csv")
    assert set(train["label"]) | set(val["label"]) == set(CATEGORIES)

    merged = pd.read_csv(out / "merged.csv")
    assert len(merged) == 60  # ทุกแถว map ได้ → merged ครบก่อน split

    w = json.loads((out / "class_weights.json").read_text())
    assert set(w) >= set(CATEGORIES)


def test_main_drops_unmapped_and_cleans_text(tmp_path):
    weak = tmp_path / "weak.csv"
    # แถวที่ 3 = เนื้อหาเป็น URL ล้วน → clean กลายเป็น "" → ต้องถูกตัด (กัน NaN ตอนอ่านกลับ)
    _d({"content": ["ดู https://x.com @user #สล็อต 555555", "zzz", "https://only-url.com"],
        "flag": ["AUTO", "AUTO", "AUTO"],
        "category": ["Gambling", "NoSuchCategory", "Gambling"]}).to_csv(weak, index=False, encoding="utf-8-sig")

    cats = tmp_path / "categories.json"
    cats.write_text(json.dumps([{"id": c, "name": c} for c in CATEGORIES], ensure_ascii=False))

    out = tmp_path / "out"
    dataset.main(["--weak", str(weak), "--categories", str(cats), "--out", str(out)])

    merged = pd.read_csv(out / "merged.csv")
    assert len(merged) == 1  # NoSuchCategory + content ว่าง ถูกตัด
    text = merged["content"].iloc[0]
    assert "https" not in text and "@" not in text and "#" not in text
    assert "555" not in text and "[LAUGH]" in text  # 555 → [LAUGH]
