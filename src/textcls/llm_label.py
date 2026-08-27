"""Task 3 — llm_label: ให้ Gemini label ข้อความเป็น 1 ใน 18 หมวด + confidence.

Flow: sample (bias ได้) → label → ~10% label ซ้ำ 2 รอบ วัด agreement (G2)
→ เขียน `*.jsonl` + รายงาน `agreement.json`.
เรียก API จริงเฉพาะตอนรัน CLI; test ใช้ mock ตาม spec.
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

DUP_FRACTION = 0.1  # G2: สุ่ม label ซ้ำ 2 รอบ ~10%
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_SEED = 42

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


def build_prompt(categories: list[dict], few_shots: list[dict], text: str) -> str:
    """สร้าง prompt: 18 หมวด + รายละเอียด + few-shot + ข้อความ → สั่งตอบ JSON."""
    cat_lines = "\n".join(f'- {c["id"]}: {c["name"]} — {c["description"]}' for c in categories)
    shot_lines = "\n".join(
        f'ข้อความ: {s["text"]}\nคำตอบ: {json.dumps({"category": s["category"], "confidence": s["confidence"]}, ensure_ascii=False)}'
        for s in few_shots
    )
    prompt = (
        "จัดหมวดหมู่ข้อความ Twitter ไทยต่อไปนี้ให้เป็นหนึ่งในหมวดด้านล่าง "
        'ตอบเป็น JSON เท่านั้น: {"category": "<id>", "confidence": 0.0-1.0}\n\n'
        f"หมวด:\n{cat_lines}\n"
    )
    if few_shots:
        prompt += f"\nตัวอย่าง:\n{shot_lines}\n"
    prompt += f'\nข้อความ: {text}\nคำตอบ: '
    return prompt


def parse_label_response(response_text: str) -> dict:
    """แยก JSON ออกจากคำตอบ (ทน code fence) → {category, confidence}."""
    cleaned = JSON_FENCE_RE.sub("", response_text).strip()
    data = json.loads(cleaned)
    if "category" not in data:
        raise ValueError(f"response ไม่มี category: {cleaned[:100]}")
    return {"category": data["category"], "confidence": float(data.get("confidence", 1.0))}


def label_text(client, model: str, prompt: str) -> dict:
    """เรียก Gemini ผ่าน client ที่ได้มา → dict label. client เป็น dependency ที่ test mock ได้."""
    resp = client.models.generate_content(model=model, contents=[prompt])
    return parse_label_response(resp.text)


class GeminiLabeler:
    """ห่อ client + model + prompt builder ให้ label ทีละข้อความ"""

    def __init__(self, client, model: str, categories: list[dict], few_shots: list[dict]):
        self.client = client
        self.model = model
        self.categories = categories
        self.few_shots = few_shots

    def label(self, text: str) -> dict:
        return label_text(self.client, self.model, build_prompt(self.categories, self.few_shots, text))


def sample_texts(df: pd.DataFrame, n: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """สุ่ม n แถว reproducible; ถ้า n ≥ จำนวนแถว คืนทั้งชุด."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def bias_sample(df: pd.DataFrame, n: int, freq: dict | None = None,
                seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Oversample คลาสหายาก: เอาแถวของคลาสหายากที่สุด (inverse frequency) เข้าก่อน.

    `freq` มาจาก distribution แรก (label 2-3k) — ถ้าไม่มี คืนแบบสุ่มธรรมดา.
    (ponytail: deterministic แบบ take-rarest-n — ถ้าต้องการสุ่ม+กรอง ratio ของน้ำหนัก
     ให้ใช้ capped-weight sampling แทน; สองเฟส label 2-3k แรกใน CLI เพิ่มเมื่อมี budget API)
    """
    if len(df) <= n or freq is None:
        return df.sample(n=n, random_state=seed)
    inv = {k: 1.0 / v for k, v in freq.items()}
    weights = df["pred_label"].map(inv).fillna(min(inv.values()))
    return df.loc[weights.sort_values(ascending=False).head(n).index]


def measure_agreement(pairs: list[tuple[str, str]]) -> float:
    """G2: สัดส่วนที่ label รอบ 1 ตรงกับรอบ 2 (ว่าง → 0.0 = ไม่มีสัญญาณ G2)."""
    if not pairs:
        return 0.0
    return sum(a == b for a, b in pairs) / len(pairs)


def label_batch(df: pd.DataFrame, labeler: GeminiLabeler,
                dup_fraction: float = DUP_FRACTION,
                seed: int = DEFAULT_SEED) -> tuple[list[dict], list[tuple[str, str]]]:
    """Label ทุกแถว; ~dup_fraction label ซ้ำรอบ 2 → (records, dup_pairs)."""
    texts = df["content_clean"].tolist()
    n = len(texts)
    dup_idx = set(random.Random(seed).sample(range(n), k=int(dup_fraction * n)))

    records = [{"content": text, **labeler.label(text)} for text in texts]
    dup_pairs = [(records[i]["category"], labeler.label(texts[i])["category"]) for i in dup_idx]
    return records, dup_pairs


def make_client(api_key: str | None = None):
    """สร้าง genai.Client — import ช้า กัน test ต้องพึ่ง SDK จริง."""
    from google import genai
    from textcls.config import get_gemini_api_key

    return genai.Client(api_key=api_key or get_gemini_api_key())


def main(argv: list[str] | None = None) -> None:
    from textcls.config import GEMINI_API_KEY_ENV, get_gemini_api_key

    p = argparse.ArgumentParser(description="LLM label ด้วย Gemini + agreement (G2)")
    p.add_argument("--input", required=True, help="preprocessed.parquet (คอลัมน์ content_clean)")
    p.add_argument("--output", required=True, help="*.jsonl (batch record)")
    p.add_argument("--category-file", required=True, help="categories.json (18 หมวด)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sample", type=int, default=5000)
    p.add_argument("--bias-sampling", action="store_true",
                   help="oversample คลาสหายาก ถ้า input มีคอลัมน์ pred_label")
    p.add_argument("--dup-fraction", type=float, default=DUP_FRACTION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)

    if not get_gemini_api_key():  # check early (ยังเป็นค่า env ตาม spec)
        raise RuntimeError(f"ไม่พบ env var {GEMINI_API_KEY_ENV}")

    df = pd.read_parquet(args.input)
    if "content_clean" not in df.columns:
        raise ValueError(f"input ต้องมีคอลัมน์ content_clean (มี {list(df.columns)})")

    categories = json.loads(Path(args.category_file).read_text(encoding="utf-8"))
    freq = df["pred_label"].value_counts().to_dict() if args.bias_sampling and "pred_label" in df.columns else None
    sampled = bias_sample(df, args.sample, freq=freq, seed=args.seed) if args.bias_sampling else sample_texts(df, args.sample, seed=args.seed)

    labeler = GeminiLabeler(make_client(), args.model, categories, few_shots=[])
    records, dup_pairs = label_batch(sampled, labeler, dup_fraction=args.dup_fraction, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    report = {"agreement_rate": measure_agreement(dup_pairs), "n_dup_pairs": len(dup_pairs)}
    (out.parent / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"labeled {len(records):,} rows -> {out} (agreement={report['agreement_rate']:.2f})")


if __name__ == "__main__":
    main()
