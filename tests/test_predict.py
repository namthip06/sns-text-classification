"""Task 6 — predict: batch classify CSV → CSV."""

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


def test_build_output_marks_low_confidence():
    df = pd.DataFrame({"content": ["a", "b", "c"]})
    rows = [{"category": "gambling", "confidence": 0.9},
            {"category": "fraud", "confidence": 0.5},
            {"category": "kratom", "confidence": 0.6}]
    out = predict.build_output(df, rows, threshold=0.6)
    assert out["category"].tolist() == ["gambling", "fraud", "kratom"]
    assert out["confidence"].tolist() == [0.9, 0.5, 0.6]
    assert out["low_confidence"].tolist() == [False, True, False]  # < threshold เท่านั้น
