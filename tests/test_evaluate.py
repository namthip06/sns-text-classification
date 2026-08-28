"""Task 5 — evaluate: G3 macro-F1 on val (agreement กับ weak rule)."""

import numpy as np
import pytest

from textcls import evaluate


def test_softmax_normalizes_rows():
    x = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 3.0]])
    p = evaluate.softmax(x)
    assert p.shape == x.shape
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)


def test_compute_report_metrics():
    true = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 1, 2])
    id2label = {0: "a", 1: "b", 2: "c"}
    r = evaluate.compute_report(true, pred, id2label)
    # pred == true ตรง 5/6 (ตำแหน่งที่ 2: 0≠1)
    assert r["accuracy"] == pytest.approx(5 / 6)
    # a: F1=0.8 (P=1,R=2/3) · b: 0.8 (P=2/3,R=1) · c: 1.0
    assert r["macro_f1"] == pytest.approx((0.8 + 0.8 + 1.0) / 3)
    assert r["per_class"]["a"]["recall"] == pytest.approx(2 / 3)
    assert r["per_class"]["b"]["recall"] == 1.0
    assert r["per_class"]["c"]["support"] == 1


def test_compute_report_skips_classes_absent_from_true():
    true = np.array([0, 0, 1])
    pred = np.array([0, 0, 2])  # class 2 ไม่มีใน true → ไม่ฉุด macro-F1
    id2label = {0: "a", 1: "b", 2: "c"}
    r = evaluate.compute_report(true, pred, id2label)
    assert set(r["per_class"]) == {"a", "b"}  # c ถูกข้าม


def test_load_calib_defaults_when_missing(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    calib = evaluate.load_calib(d)
    assert calib["temperature"] == 1.0
    assert calib["threshold"] == 0.6
