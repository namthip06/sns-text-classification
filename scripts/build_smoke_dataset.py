#!/usr/bin/env python3
"""สร้าง smoke dataset (100 ประโยค) จาก weak labels จริง — พิสูจน์ pipeline เทรน.

output: data/smoke/{train,val}.csv (gitignored อยู่แล้ว)
"""
import json

import pandas as pd

from textcls import dataset, weak_label

SRC = "data/weak_labels_auto.csv"
N = 100

df = weak_label.load_weak(SRC).sample(n=N, random_state=42).reset_index(drop=True)
df = df.rename(columns={"predicted_label": "label"})
cats = json.load(open("data/categories.json", encoding="utf-8"))
df = dataset.map_labels(df, [c["id"] for c in cats])

train, val = dataset.train_val_split(df, val_frac=0.2, seed=42)
out = "data/smoke"
import pathlib

pathlib.Path(out).mkdir(parents=True, exist_ok=True)
train.to_csv(f"{out}/train.csv", index=False, encoding="utf-8-sig")
val.to_csv(f"{out}/val.csv", index=False, encoding="utf-8-sig")
print(f"train={len(train)} val={len(val)}")
print(train["label"].value_counts().to_string())
