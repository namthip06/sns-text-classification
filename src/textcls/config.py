"""Central config: paths, thresholds (G1/G4), และตัวชี้ไปยัง 18 หมวด (data/categories.json).

ทุกค่า path/threshold ที่ pipeline ใช้รวมอยู่ไฟล์เดียว (ตาม spec Code Style)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CATEGORIES_PATH = DATA_DIR / "categories.json"  # 18 หมวด + รายละเอียด (ใช้ใน prompt)

# ── thresholds ─────────────────────────────────────────────────────────
PRECISION_THRESHOLD = 0.85  # G1: tag rule แม่นกว่าเกณฑ์ → ปลดล็อกเป็น label ฝึก
CONFIDENCE_THRESHOLD = 0.6  # G4: score ต่ำกว่านี้ → ขึ้น label "low confidence"

# ── secrets ────────────────────────────────────────────────────────────
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


def get_gemini_api_key() -> str:
    """Gemini API key — ผ่าน env var เท่านั้น (ไม่เคย hardcode/commit)."""
    key = os.environ.get(GEMINI_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"ไม่พบ env var {GEMINI_API_KEY_ENV}")
    return key
