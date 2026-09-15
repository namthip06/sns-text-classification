"""Shared inference: โหลด classifier จากโฟลเดอร์ checkpoint + predict logits.

ใช้ร่วมโดย calibrate / evaluate / predict — โหลด model+tokenizer จากโฟลเดอร์
ที่ train.py เขียน (ต้องมี run_config.json ที่เก็บ label2id).
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 256


def load_model_dir(model_dir: str | Path):
    """โหลด model + tokenizer + id→label + label→id จาก checkpoint dir."""
    d = Path(model_dir)
    cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    label2id: dict[str, int] = cfg["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(d)
    tokenizer = AutoTokenizer.from_pretrained(d)
    return model, tokenizer, id2label, label2id


def predict_logits(model, tokenizer, texts: list[str], batch_size: int = 32,
                   progress: bool = False) -> np.ndarray:
    """รันข้อความผ่าน model → logits [n, k] (no_grad, batch ทีละ batch_size).

    progress=True → แสดง tqdm progress bar (predict interactive เท่านั้น).
    """
    model.eval()
    outs = []
    with torch.no_grad(), tqdm(total=len(texts), desc="predicting", unit="row",
                               disable=not progress) as pbar:
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LENGTH,
                            padding=True, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            outs.append(model(**enc).logits.cpu())
            pbar.update(min(i + batch_size, len(texts)) - i)
    return torch.cat(outs).numpy()
