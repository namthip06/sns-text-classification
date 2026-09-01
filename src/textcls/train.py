"""Task 4 — fine-tune BERT-style classifier (WangchanBERTa/PhayaThaiBERT).

เทรนจาก weak ทั้งหมดตรงๆ (ข้อมูลทุก row เป็น weak) → checkpoint ลง models/<tag>/
(default models/model/). A/B ที่เหลือ = เทียบโมเดล base ผ่าน `--model`.
val ใช้ data/val.csv (hold-out จาก weak) · reproducible (seed + config บันทึก).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments)

DEFAULT_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"
MAX_LENGTH = 256


def label2id(categories: list[dict]) -> dict:
    """category id → index (เรียงตามลำดับใน categories.json)."""
    return {c["id"]: i for i, c in enumerate(categories)}


def loss_weights(weights: dict, lid: dict) -> torch.Tensor:
    """class weights (จาก dataset.class_weights) → tensor เรียงตาม label2id."""
    return torch.tensor([weights[c] for c in lid], dtype=torch.float)


class TextDataset(Dataset):
    """Dataset สำหรับ Trainer — tokenize ทีละแถว (truncate + padding ที่ collator)."""

    def __init__(self, df: pd.DataFrame, tokenizer, lid: dict, max_length: int = MAX_LENGTH):
        self.texts = df["content"].astype(str).tolist()
        self.labels = df["label"].map(lid).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length)
        return {**enc, "labels": torch.tensor(self.labels[i], dtype=torch.long)}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred) -> dict:
    """macro-F1 บนคลาสที่ปรากฏใน true (zero_division=0) — คิดแบบเดียวกับ G3 ใน evaluate.py."""
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    pred = np.asarray(logits).argmax(axis=1)
    labels = np.asarray(labels)
    present = sorted(set(labels.tolist()))
    return {"macro_f1": float(f1_score(labels, pred, labels=present,
                                       average="macro", zero_division=0))}


class MetricsCallback(TrainerCallback):
    """เก็บ metrics ต่อ epoch ลง metrics.json — ดูย้อนหลังได้หลังจบเทรน (console log หายเอง)."""

    def __init__(self, out_dir: Path):
        self.path = out_dir / "metrics.json"
        self.history: list[dict] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        if metrics is None:
            return
        self.history.append({k: v for k, v in metrics.items()
                             if isinstance(v, (int, float, str))})
        self.path.write_text(json.dumps(self.history, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def training_kwargs(args) -> dict:
    """kwargs ของ TrainingArguments — แยกให้ test ได้โดยไม่ต้องสร้าง Trainer."""
    return {
        "output_dir": str(Path(args.out) / args.tag),
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "eval_strategy": "epoch",  # transformers v5 ใช้ชื่อนี้ (เดิม evaluation_strategy)
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),
        "logging_steps": args.logging_steps,
        # tensorboard reporter (transformers v5) เขียนลง <output_dir>/runs/ เอง (ไม่มี --logging_dir แล้ว)
        "report_to": ["tensorboard"],
    }


def main(argv: list[str] | None = None) -> None:
    from textcls.dataset import class_weights

    p = argparse.ArgumentParser(description="fine-tune classifier จาก weak ทั้งหมด")
    p.add_argument("--train", required=True, help="train.csv (content, label)")
    p.add_argument("--val", required=True, help="val.csv")
    p.add_argument("--categories", required=True, help="categories.json (18 หมวด)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (default WangchanBERTa)")
    p.add_argument("--out", default="models", help="output root dir (models/)")
    p.add_argument("--tag", default="model", help="ชื่อโฟลเดอร์ checkpoint (default model)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=50)
    args = p.parse_args(argv)

    categories = json.loads(Path(args.categories).read_text(encoding="utf-8"))
    ids = [c["id"] for c in categories]
    lid = label2id(categories)

    train_df = pd.read_csv(args.train, encoding="utf-8-sig")
    val_df = pd.read_csv(args.val, encoding="utf-8-sig")

    class_w_t = loss_weights(class_weights(train_df["label"], ids), lid)

    setup_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = TextDataset(train_df, tokenizer, lid)
    val_ds = TextDataset(val_df, tokenizer, lid)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(ids))
    # CE + class weight (แก้ loss_fct ของ AutoModelForSequenceClassification ตรงๆ)
    model.loss_fct = torch.nn.CrossEntropyLoss(weight=class_w_t)

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs(args)),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback(out_dir)],
    )

    trainer.train()

    model.loss_fct = torch.nn.CrossEntropyLoss()  # reset ก่อน save กัน class-weight หลุดเข้าร state_dict

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "run_config.json").write_text(json.dumps({
        "model": args.model, "tag": args.tag, "seed": args.seed,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "categories": ids, "label2id": lid,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done → {out_dir}/ · metrics: {out_dir / 'metrics.json'} · "
          f"TensorBoard: uv run tensorboard --logdir {out_dir / 'runs'}")


if __name__ == "__main__":
    main()
