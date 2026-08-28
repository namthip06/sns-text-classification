"""Task 5 — calibrate: temperature scaling (G4)."""

import numpy as np

from textcls import calibrate


def test_nll_penalizes_wrong_prediction():
    logits = np.array([[3.0, 0.0, 0.0]])
    assert calibrate.nll(logits, np.array([0]), 1.0) < calibrate.nll(logits, np.array([1]), 1.0)


def test_fit_temperature_lowers_nll():
    # overconfident separable logits — NLL หลัง fit ไม่ควรแย่กว่า T=1
    logits = np.array([
        [4.0, 1.0, 0.0],
        [0.5, 4.0, 0.2],
        [0.1, 0.2, 4.0],
    ])
    labels = np.array([0, 1, 2])
    t = calibrate.fit_temperature(logits, labels)
    assert t > 0
    assert calibrate.nll(logits, labels, t) <= calibrate.nll(logits, labels, 1.0) + 1e-6


def test_fit_temperature_returns_positive_scalar():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    t = calibrate.fit_temperature(logits, np.array([0, 1]))
    assert isinstance(t, float)
    assert t > 0
