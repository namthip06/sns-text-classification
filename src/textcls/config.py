"""Central config: paths + ตัวชี้ไปยัง 19 หมวด (data/categories.json).

ทุกค่า path ที่ pipeline ใช้รวมอยู่ไฟล์เดียว (ตาม spec Code Style)
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CATEGORIES_PATH = DATA_DIR / "categories.json"  # 19 หมวด (18 + no_match) + รายละเอียด
