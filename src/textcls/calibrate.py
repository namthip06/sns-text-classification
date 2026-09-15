"""Task 5 — calibrate: temperature scaling (G4) fit บน val → save calib.json.

`calib.json` = {"temperature": T, "threshold": CONFIDENCE_THRESHOLD} เก็บในโฟลเดอร์
model — evaluate / predict ใช้ปรับ confidence; score ต่ำกว่า threshold → "no_match".
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from textcls.config import CONFIDENCE_THRESHOLD
from textcls.evaluate import softmax
from textcls.infer import load_model_dir, predict_logits


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """NLL ของ temperature-scaled softmax ที่ temperature หนึ่ง (ใช้ตรวจ calibration)."""
    logits_t = torch.tensor(logits, dtype=torch.float32) / temperature
    return float(torch.nn.functional.cross_entropy(logits_t, torch.tensor(labels)).item())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """หาค่า scalar T ที่ minimize NLL ของ softmax(logits/T) — LBFGS บนตัวแปรเดียว."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    t = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / t, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    # ponytail: บังคับ T>0 กันพลิกเครื่องหมาย (logits/T กลับด้าน)
    return max(float(t.item()), 1e-3)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="fit temperature scaling (G4) บน val")
    p.add_argument("--model", required=True, help="checkpoint dir (มี run_config.json)")
    p.add_argument("--val", required=True, help="val.csv (content, label)")
    args = p.parse_args(argv)

    model, tokenizer, _, label2id = load_model_dir(args.model)
    df = pd.read_csv(args.val, encoding="utf-8-sig")
    df = df[df["label"].isin(label2id)]
    logits = predict_logits(model, tokenizer, df["content"].astype(str).tolist())
    labels = df["label"].map(label2id).to_numpy()

    t = fit_temperature(logits, labels)
    no_match = int((softmax(logits / t).max(axis=1) < CONFIDENCE_THRESHOLD).sum())
    out = Path(args.model) / "calib.json"
    out.write_text(json.dumps({"temperature": t, "threshold": CONFIDENCE_THRESHOLD},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"temperature={t:.4f}  (nll T=1: {nll(logits, labels, 1.0):.4f} → T={t:.4f}: "
          f"{nll(logits, labels, t):.4f})  threshold={CONFIDENCE_THRESHOLD} → "
          f"no_match={no_match:,}/{len(df):,}  → {out}")


if __name__ == "__main__":
    main()
