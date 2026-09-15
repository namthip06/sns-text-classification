"""Task 6 — predict: batch classify CSV/Excel → CSV/Excel."""

import numpy as np
import pandas as pd

from textcls import predict


def test_classify_probs_argmax_and_confidence():
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8],
    ])
    id2label = {0: "gambling", 1: "fraud", 2: "kratom"}
    rows = predict.classify_probs(probs, id2label)
    assert rows == [{"category": "gambling", "confidence": 0.7},
                    {"category": "kratom", "confidence": 0.8}]


def test_classify_probs_blank_rows_stay_empty():
    probs = np.array([
        [0.7, 0.2, 0.1],
        [np.nan, np.nan, np.nan],  # แถวข้อความว่าง — ไม่ได้ predict
        [0.1, 0.1, 0.8],
    ])
    id2label = {0: "gambling", 1: "fraud", 2: "kratom"}
    rows = predict.classify_probs(probs, id2label)
    assert rows[0] == {"category": "gambling", "confidence": 0.7}
    assert rows[1] == {"category": "", "confidence": rows[1]["confidence"]}  # nan
    assert np.isnan(rows[1]["confidence"])
    assert rows[2] == {"category": "kratom", "confidence": 0.8}


def test_build_output_marks_low_confidence():
    df = pd.DataFrame({"content": ["a", "b", "c"]})
    rows = [{"category": "gambling", "confidence": 0.9},
            {"category": "fraud", "confidence": 0.5},
            {"category": "kratom", "confidence": 0.6}]
    out = predict.build_output(df, rows, threshold=0.6)
    assert out["category"].tolist() == ["gambling", "fraud", "kratom"]
    assert out["confidence"].tolist() == [0.9, 0.5, 0.6]
    assert out["low_confidence"].tolist() == [False, True, False]  # < threshold เท่านั้น


def test_build_output_blank_row_not_low_confidence():
    df = pd.DataFrame({"content": ["a", ""]})
    rows = [{"category": "gambling", "confidence": 0.9},
            {"category": "", "confidence": float("nan")}]
    out = predict.build_output(df, rows, threshold=0.6)
    assert out["category"].tolist() == ["gambling", ""]
    assert out["low_confidence"].tolist() == [False, False]  # nan < threshold → False


def test_bar_chart_shares_and_sorting():
    counts = pd.Series({"gambling": 8, "fraud": 2})
    chart = predict.bar_chart(counts, total=10)
    assert "gambling" in chart and "fraud" in chart
    assert "80.0%" in chart and "20.0%" in chart
    gi, fi = chart.index("gambling"), chart.index("fraud")
    assert gi < fi  # เรียงมาก → น้อย (value_counts ให้มาแล้ว)


def test_read_input_no_header_names_and_keeps_rows(tmp_path):
    src = tmp_path / "nohead.csv"
    src.write_text("ข้อความแรก\nข้อความที่สอง\n", encoding="utf-8")
    df = predict.read_input(src, interactive=False, no_header=True)
    assert df.columns.tolist() == ["col_1"]  # ตั้งชื่อเอง ไม่กินแถวแรกเป็น header
    assert df["col_1"].tolist() == ["ข้อความแรก", "ข้อความที่สอง"]


def test_read_input_with_header_normal(tmp_path):
    src = tmp_path / "head.csv"
    src.write_text("content\nข้อความแรก\n", encoding="utf-8")
    df = predict.read_input(src, interactive=False)
    assert df.columns.tolist() == ["content"]


def test_choose_model(monkeypatch, tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "run_config.json").write_text("{}", encoding="utf-8")
    # โมเดลเดียว = เลือกเองโดยไม่ถาม (ข้ามไฟล์ที่ไม่ใช่ dir)
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    assert predict.choose_model(tmp_path) == tmp_path / "b"
    # หลายโมเดล = ถาม → ตอบ 2 → ตัวที่สอง (เรียงตามชื่อ: a, b)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "run_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert predict.choose_model(tmp_path) == tmp_path / "b"
