"""Central config: paths + thresholds (G4) + ตัวชี้ไปยัง 18 หมวด (data/categories.json).

ทุกค่า path/threshold ที่ pipeline ใช้รวมอยู่ไฟล์เดียว (ตาม spec Code Style)
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CATEGORIES_PATH = DATA_DIR / "categories.json"  # 18 หมวด + รายละเอียด

# ── thresholds ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.6  # G4: score ต่ำกว่านี้ → ขึ้น label "low confidence"
