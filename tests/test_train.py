"""Task 6 — train: หน่วยที่ test ได้โดยไม่ต้องโหลดโมเดลจริง/ดาวน์โหลด weights."""

import json
import types

import pytest
import torch

from textcls import train

CATEGORIES = [
    {"id": "gambling", "name": "พนัน"},
    {"id": "fraud", "name": "หลอกลวง"},
    {"id": "kratom", "name": "กระท่อม"},
]


def test_label2id_index_order():
    lid = train.label2id(CATEGORIES)
    assert lid == {"gambling": 0, "fraud": 1, "kratom": 2}


def test_loss_weights_aligned_to_label2id():
    weights = {"gambling": 0.5, "fraud": 0.25, "kratom": 0.25}
    t = train.loss_weights(weights, train.label2id(CATEGORIES))
    assert t.tolist() == [0.5, 0.25, 0.25]
    assert t.dtype == torch.float32


def test_training_kwargs_defaults():
    args = types.SimpleNamespace(out="models", tag="model", lr=2e-5, epochs=3,
                                 batch_size=16, seed=42, logging_steps=50)
    kw = train.training_kwargs(args)
    assert kw["output_dir"] == "models/model"
    assert kw["learning_rate"] == 2e-5
    assert kw["num_train_epochs"] == 3
    assert kw["seed"] == 42
    assert kw["report_to"] == ["tensorboard"]
    assert kw["fp16"] == torch.cuda.is_available()


def test_compute_metrics_macro_f1_skips_absent_classes():
    """คลาสที่ไม่มีใน true ไม่ฉุด macro-F1 — คิดแบบเดียวกับ G3 (evaluate.py)."""
    import numpy as np

    # 3 คลาส แต่ true มีแค่คลาส 0,1 → macro-F1 คิดเฉพาะสองคลาสนี้
    logits = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    labels = np.array([0, 1])
    m = train.compute_metrics((logits, labels))
    assert m["macro_f1"] == pytest.approx(1.0)

    pred_wrong = np.array([[9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    m2 = train.compute_metrics((pred_wrong, labels))
    assert 0.0 <= m2["macro_f1"] < 1.0


def test_metrics_callback_persists_per_epoch(tmp_path):
    """on_evaluate แต่ละครั้ง append และเขียน metrics.json ทันที (กระโดดออกกลางทางก็เหลือข้อมูล)."""
    cb = train.MetricsCallback(tmp_path)
    cb.on_evaluate(None, None, None, metrics={"eval_loss": 2.1, "eval_macro_f1": 0.4, "epoch": 1.0})
    cb.on_evaluate(None, None, None, metrics={"eval_loss": 1.7, "eval_macro_f1": 0.5, "epoch": 2.0})
    saved = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    history = json.loads(saved)
    assert [h["eval_macro_f1"] for h in history] == [0.4, 0.5]
    assert history[1]["epoch"] == 2.0


def test_metrics_callback_noop_without_metrics(tmp_path):
    cb = train.MetricsCallback(tmp_path)
    cb.on_evaluate(None, None, None)
    assert not (tmp_path / "metrics.json").exists()


class FakeTokenizer:
    """จำลอง tokenizer — ไม่โหลดโมเดลจริง"""

    def __call__(self, text, truncation=True, max_length=None):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_text_dataset_maps_labels(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"content": ["a", "b"], "label": ["gambling", "fraud"]})
    ds = train.TextDataset(df, FakeTokenizer(), train.label2id(CATEGORIES))
    assert len(ds) == 2
    item = ds[0]
    assert item["input_ids"] == [1, 2, 3]
    assert item["labels"].item() == 0  # gambling → id 0
