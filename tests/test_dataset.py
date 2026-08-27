"""Task 5 — dataset: merge labels + stratified split + class weight + weak cap."""

import json

import pandas as pd
import pytest

from textcls import dataset


CATEGORIES = ["gambling", "pornography", "fraud", "ecig", "prostitution", "kratom"]


def _d(rows):
    return pd.DataFrame(rows)


# ── merge / priority ────────────────────────────────────────────────────

def test_merge_labels_priority_human_wins():
    weak = _d({"content": ["x", "y"], "label": ["gambling", "kratom"], "source": ["weak", "weak"]})
    llm = _d({"content": ["x", "z"], "label": ["gambling", "pornography"], "source": ["llm", "llm"]})
    human = _d({"content": ["x"], "label": ["fraud"], "source": ["human"]})
    merged = dataset.merge_labels([weak, llm, human])
    by_content = merged.set_index("content")
    assert by_content.loc["x", "source"] == "human"  # human ชนะ
    assert len(merged) == 3  # x, y, z


# ── label mapping (weak taxonomy → 18 หมวด) ─────────────────────────────

def test_map_labels_aliases_and_drop_unmapped():
    df = _d({"content": ["a", "b", "c", "d"],
             "label": ["scam", "kratom drink", "e-cigarettes", "not_a_category"],
             "source": ["weak"] * 4})
    out = dataset.map_labels(df, CATEGORIES)
    assert out["label"].tolist() == ["fraud", "kratom", "ecig"]  # unmapped ถูกตัด


def test_map_labels_identity_passthrough():
    df = _d({"content": ["a"], "label": ["gambling"], "source": ["llm"]})
    out = dataset.map_labels(df, CATEGORIES)
    assert out["label"].tolist() == ["gambling"]


# ── split ───────────────────────────────────────────────────────────────

def test_train_val_split_stratified_keeps_all_classes():
    rows = []
    for cls in CATEGORIES:
        rows += [{"content": f"{cls}-{i}", "label": cls, "source": "llm"} for i in range(20)]
    df = pd.DataFrame(rows)
    tr, va = dataset.train_val_split(df, val_frac=0.2, seed=42)
    assert set(va["label"]) == set(CATEGORIES)  # ไม่มีคลาสหายจาก val
    assert set(tr["label"]) == set(CATEGORIES)
    assert len(va) == pytest.approx(0.2 * len(df), rel=0.15)


# ── class weight / weak cap ─────────────────────────────────────────────

def test_class_weights_inverse_frequency():
    labels = pd.Series(["a"] * 4 + ["b"] * 2 + ["c"] * 2)
    w = dataset.class_weights(labels, ["a", "b", "c"])
    # inverse frequency: a หายากสุด (4/8=0.5) → weight น้อยสุด; กลับกัน c เยอะสุด
    assert w["b"] == w["c"]  # count เท่ากัน → weight เท่ากัน
    assert w["a"] < w["b"]
    assert sum(w.values()) == pytest.approx(1.0)  # normalized


def test_cap_weak_fraction_bounds():
    rows = [{"content": f"w{i}", "label": "gambling", "source": "weak"} for i in range(7)]
    rows += [{"content": f"l{i}", "label": "gambling", "source": "llm"} for i in range(3)]
    train = pd.DataFrame(rows)
    out = dataset.cap_weak_fraction(train, cap=0.4, seed=42)
    weak_frac = (out["source"] == "weak").mean()
    assert weak_frac <= 0.4
    assert (out["source"] == "weak").sum() == 2  # 3 non-weak → weak ≤ 2


def test_cap_weak_fraction_noop_when_under_cap():
    rows = [{"content": f"l{i}", "label": "gambling", "source": "llm"} for i in range(8)]
    rows += [{"content": f"w{i}", "label": "gambling", "source": "weak"} for i in range(2)]
    train = pd.DataFrame(rows)
    assert len(dataset.cap_weak_fraction(train, cap=0.4)) == 10


# ── CLI end-to-end ──────────────────────────────────────────────────────

def test_main_writes_splits_and_weights(tmp_path):
    weak = tmp_path / "weak_gated.csv"
    _d({"content": ["w1", "w2", "w3"], "predicted_label": ["scam", "gambling", "kratom drink"],
        "source": ["weak"] * 3}).to_csv(weak, index=False, encoding="utf-8-sig")

    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    with open(llm_dir / "b1.jsonl", "w", encoding="utf-8") as f:
        for content, cat in [("l1", "gambling"), ("l2", "gambling"), ("l3", "pornography"),
                             ("l4", "fraud"), ("l5", "kratom"), ("l6", "ecig"),
                             ("l7", "prostitution"), ("l8", "gambling"), ("l9", "fraud"), ("l10", "ecig")]:
            f.write(json.dumps({"content": content, "category": cat, "confidence": 0.9}) + "\n")

    human = tmp_path / "human_labels.csv"
    _d({"content": ["h1", "h2"], "label": ["gambling", "fraud"]}).to_csv(human, index=False, encoding="utf-8-sig")

    cats = tmp_path / "categories.json"
    cats.write_text(json.dumps([{"id": c, "name": c} for c in CATEGORIES], ensure_ascii=False))

    out = tmp_path / "out"
    dataset.main(["--weak", str(weak), "--llm", str(llm_dir), "--human-test", str(human),
                  "--categories", str(cats),
                  "--out", str(out), "--cap-weak", "0.4"])

    test = pd.read_csv(out / "test.csv")
    assert set(test["source"]) == {"human"}  # test = ชุดมนุษย์เท่านั้น

    train = pd.read_csv(out / "train.csv")
    assert list(train.columns) == ["content", "label", "source"]
    assert (train["source"] == "weak").mean() <= 0.4

    val = pd.read_csv(out / "val.csv")
    assert set(train["label"]) | set(val["label"]) | set(test["label"]) >= set(CATEGORIES)

    w = json.loads((out / "class_weights.json").read_text())
    assert set(w) >= set(CATEGORIES)
