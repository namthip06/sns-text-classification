"""Task 4 — weak_label: G1 per-tag precision gate.

tag = predicted label ของ rule (อ้างอิง data contract: content, predicted_label)
referee = LLM label (jsonl: content, category)
"""

import json

import pandas as pd
import pytest

from textcls import weak_label


# ── per-tag precision ───────────────────────────────────────────────────

def test_per_tag_precision_math():
    weak = pd.DataFrame({
        "content": ["a", "b", "c", "d", "e"],
        "predicted_label": ["sport", "sport", "sport", "politics", "politics"],
    })
    referee = pd.DataFrame({
        "content": ["a", "b", "c", "d", "e"],
        "category": ["sport", "sport", "politics", "politics", "politics"],
    })
    prec = weak_label.per_tag_precision(weak, referee)
    assert prec["sport"] == pytest.approx(2 / 3)
    assert prec["politics"] == pytest.approx(1.0)


def test_per_tag_precision_normalizes_case():
    # weak ใช้ "Gambling" แต่ referee ใช้ "gambling" — ต้อง match หลัง normalize
    weak = pd.DataFrame({"content": ["a"], "predicted_label": ["Gambling"]})
    referee = pd.DataFrame({"content": ["a"], "category": ["gambling"]})
    assert weak_label.per_tag_precision(weak, referee)["gambling"] == 1.0


def test_tag_without_referee_is_nan():
    weak = pd.DataFrame({"content": ["a"], "predicted_label": ["sport"]})
    referee = pd.DataFrame(columns=["content", "category"])  # ไม่มี referee ตรงกัน
    prec = weak_label.per_tag_precision(weak, referee)
    assert pd.isna(prec["sport"])


# ── gate ────────────────────────────────────────────────────────────────

def test_gate_keeps_only_tags_above_threshold():
    weak = pd.DataFrame({
        "content": ["a", "b", "c", "d"],
        "predicted_label": ["sport", "sport", "sport", "politics"],
    })
    precision = pd.Series({"sport": 2 / 3, "politics": 1.0})
    gated = weak_label.gate_weak_labels(weak, precision, threshold=0.85)
    assert gated["predicted_label"].tolist() == ["politics"]


def test_gate_drops_nan_precision_tags():
    weak = pd.DataFrame({"content": ["a"], "predicted_label": ["sport"]})
    precision = pd.Series({"sport": float("nan")})
    assert weak_label.gate_weak_labels(weak, precision).empty


# ── report ──────────────────────────────────────────────────────────────

def test_build_report_verdict():
    precision = pd.Series({"sport": 1.0, "politics": 0.5})
    report = weak_label.build_report(precision, threshold=0.85)
    assert list(report.columns) == ["tag", "precision", "verdict"]
    assert report.set_index("tag").loc["sport", "verdict"] == "gate"
    assert report.set_index("tag").loc["politics", "verdict"] == "drop"


# ── load (รองรับข้อมูลจริง: คอลัมน์ category + flag AUTO) ─────────────────

def test_load_weak_uses_category_and_filters_non_auto():
    path = "/tmp/_t4_weak.csv"
    pd.DataFrame({
        "content": ["a", "b", "c"],
        "flag": ["AUTO", "NOT_THAI", "NO_LABEL"],
        "category": ["gambling", "gambling", None],
    }).to_csv(path, index=False, encoding="utf-8-sig")
    df = weak_label.load_weak(path)
    assert df["predicted_label"].tolist() == ["gambling"]


def test_load_referee_from_jsonl():
    path = "/tmp/_t4_ref.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"content": "a", "category": "sport", "confidence": 0.9}\n')
        f.write('{"content": "b", "category": "politics", "confidence": 0.7}\n')
    df = weak_label.load_referee(path)
    assert list(df.columns) == ["content", "category"]
    assert len(df) == 2


# ── CLI end-to-end ──────────────────────────────────────────────────────

def test_main_writes_gated_and_report(tmp_path):
    weak_path = tmp_path / "weak.csv"
    pd.DataFrame({
        "content": ["a", "b", "c", "d"],
        "predicted_label": ["sport", "sport", "sport", "politics"],
    }).to_csv(weak_path, index=False, encoding="utf-8-sig")

    ref_path = tmp_path / "ref.jsonl"
    with open(ref_path, "w", encoding="utf-8") as f:
        for content, cat in [("a", "sport"), ("b", "sport"), ("c", "politics"), ("d", "politics")]:
            f.write(json.dumps({"content": content, "category": cat, "confidence": 0.9}) + "\n")

    out = tmp_path / "weak_gated.csv"
    weak_label.main(["--weak", str(weak_path), "--referee", str(ref_path),
                     "--precision-threshold", "0.85", "--output", str(out)])

    gated = pd.read_csv(out, encoding="utf-8-sig")
    assert list(gated.columns) == ["content", "predicted_label", "source"]
    assert gated["predicted_label"].tolist() == ["politics"]
    assert set(gated["source"]) == {"weak"}

    report = pd.read_csv(out.with_suffix(".report.csv"))
    assert "verdict" in report.columns
    assert report["tag"].tolist() == ["politics", "sport"]  # เรียง precision
