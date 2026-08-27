#!/usr/bin/env python3
"""Split data/weak_labels.csv by flag into three files: AUTO, NO_LABEL, and the rest (excluding NOT_THAI)."""
import csv

SRC = "data/weak_labels.csv"
OUT = {
    "AUTO": "data/weak_labels_auto.csv",
    "NO_LABEL": "data/weak_labels_no_label.csv",
    "REST": "data/weak_labels_other.csv",
}

with open(SRC, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    buckets = {k: [] for k in OUT}
    total = dropped = 0
    for row in reader:
        total += 1
        flag = row["flag"]
        if flag in OUT:
            buckets[flag].append(row)
        elif flag != "NOT_THAI":
            buckets["REST"].append(row)
        else:
            dropped += 1

for key, path in OUT.items():
    rows = buckets[key]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else None)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows)}")

# ponytail: split must be lossless — only NOT_THAI rows may be dropped
assert sum(len(v) for v in buckets.values()) + dropped == total

