"""Task 5 — evaluate: G3 macro-F1 บน val (agreement กับ weak rule — ไม่ใช่ความจริงสัมบูรณ์).

val เป็น hold-out จาก weak → metric วัดว่าโมเดลตรงกับ label ของ tag rule แค่ไหน
ไม่ใช่ความจริงสัมบูรณ์ — ต้องระบุข้อจำกัดนี้ในรายงานทุกครั้ง (ตาม spec).

pred = argmax ของ 19 คลาส (`no_match` รวมอยู่ใน taxonomy เป็นคลาสเทรน)
confidence ปรับด้วย temperature scaling จาก calib.json (ถ้ามี).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from textcls.infer import load_model_dir, predict_logits


def softmax(x: np.ndarray) -> np.ndarray:
    """softmax ตาม row (stable)."""
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_calib(model_dir: str | Path) -> dict:
    """อ่าน calib.json (ถ้ามี) — ไม่มี → default T=1.0."""
    p = Path(model_dir) / "calib.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"temperature": 1.0}


def compute_report(true: np.ndarray, pred: np.ndarray, id2label: dict[int, str]) -> dict:
    """เมตริกบนคลาสที่ปรากฏใน true; คลาสที่ไม่มีใน true → ข้าม ไม่ฉุด macro-F1."""
    labels = sorted({int(i) for i in true})
    names = [id2label[i] for i in labels]
    r = recall_score(true, pred, labels=labels, average=None, zero_division=0)
    p = precision_score(true, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(true, pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(true, pred, labels=labels)
    per_class = {
        names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                   "support": int(cm[i].sum())}
        for i in range(len(labels))
    }
    return {
        "accuracy": float((np.asarray(true) == np.asarray(pred)).mean()),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="evaluate G3: macro-F1 บน val (agreement กับ weak)")
    p.add_argument("--model", required=True, help="checkpoint dir (มี run_config.json + calib.json)")
    p.add_argument("--test", required=True, help="val.csv (content, label)")
    args = p.parse_args(argv)

    model, tokenizer, id2label, label2id = load_model_dir(args.model)
    calib = load_calib(args.model)
    df = pd.read_csv(args.test, encoding="utf-8-sig")
    df = df[df["label"].isin(label2id)]
    logits = predict_logits(model, tokenizer, df["content"].astype(str).tolist())
    pred = softmax(logits / calib["temperature"]).argmax(axis=1)
    true = df["label"].map(label2id).to_numpy()

    report = compute_report(true, pred, id2label)
    report["n_test"] = int(len(df))
    report["temperature"] = calib["temperature"]

    out = Path(args.model) / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("evaluate บน val — G3: agreement กับ weak rule (ไม่ใช่ความจริงสัมบูรณ์)")
    print(f"accuracy={report['accuracy']:.4f}  macro_f1={report['macro_f1']:.4f}  n={report['n_test']:,}")
    print("per-class:")
    for name, m in report["per_class"].items():
        print(f"  {name:20s} recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
              f"f1={m['f1']:.3f}  support={m['support']}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
