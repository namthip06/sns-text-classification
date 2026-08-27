"""Task 3 — llm_label: Gemini labeling + agreement (G2). Mock API ตาม spec."""

import json

import pandas as pd
import pytest

from textcls import llm_label


class FakeResp:
    def __init__(self, text: str):
        self.text = text


class FakeClient:
    """Stub ของ genai.Client — ตอบ JSON ตามลำดับที่ร้องขอ"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    @property
    def models(self):
        return self

    def generate_content(self, model: str, contents: list[str]):
        assert self._responses, "no more canned responses"
        return FakeResp(self._responses.pop(0))


CATEGORIES = [
    {"id": "sport", "name": "กีฬา", "description": "ข่าวกีฬา คะแนน แมตช์"},
    {"id": "politics", "name": "การเมือง", "description": "ข่าวการเมือง ราชการ"},
    {"id": "entertainment", "name": "บันเทิง", "description": "ดารา ละคร เพลง"},
]


# ── prompt / parse ─────────────────────────────────────────────────────

def test_build_prompt_contains_categories_and_text():
    prompt = llm_label.build_prompt(CATEGORIES, [], "วันนี้ฟุตบอลนัดชิง")
    assert "sport: กีฬา" in prompt
    assert "politics: การเมือง" in prompt
    assert "entertainment: บันเทิง" in prompt
    assert "วันนี้ฟุตบอลนัดชิง" in prompt
    assert "category" in prompt  # สั่งให้ตอบ JSON


def test_parse_label_response_valid_json():
    out = llm_label.parse_label_response('{"category": "sport", "confidence": 0.9}')
    assert out == {"category": "sport", "confidence": 0.9}


def test_parse_label_response_with_code_fence():
    out = llm_label.parse_label_response('```json\n{"category": "sport", "confidence": 0.8}\n```')
    assert out["category"] == "sport"


def test_parse_label_response_missing_category_raises():
    with pytest.raises(ValueError):
        llm_label.parse_label_response('{"confidence": 0.9}')


def test_parse_label_response_confidence_defaults():
    out = llm_label.parse_label_response('{"category": "sport"}')
    assert out["confidence"] == 1.0


# ── API call (mock) ─────────────────────────────────────────────────────

def test_label_text_uses_fake_client():
    client = FakeClient(['{"category": "sport", "confidence": 0.9}'])
    out = llm_label.label_text(client, "gemini-2.5-pro", "prompt")
    assert out == {"category": "sport", "confidence": 0.9}


# ── sampling / bias ─────────────────────────────────────────────────────

def test_sample_texts_reproducible_with_seed():
    df = pd.DataFrame({"content_clean": [f"t{i}" for i in range(50)]})
    a = llm_label.sample_texts(df, 10, seed=42)
    b = llm_label.sample_texts(df, 10, seed=42)
    assert a["content_clean"].tolist() == b["content_clean"].tolist()
    assert len(a) == 10


def test_sample_texts_returns_all_when_n_exceeds():
    df = pd.DataFrame({"content_clean": ["a", "b"]})
    assert len(llm_label.sample_texts(df, 10)) == 2


def test_bias_sample_oversamples_rare_class():
    # 1 แถว rare + 99 common → inverse-frequency weight ดัน rare เข้า sample แน่นอน
    df = pd.DataFrame({"content_clean": [f"t{i}" for i in range(100)], "pred_label": ["common"] * 100})
    df.loc[0, "pred_label"] = "rare"
    freq = {"rare": 1, "common": 99}
    out = llm_label.bias_sample(df, 10, freq=freq, seed=42)
    assert df.loc[0, "content_clean"] in out["content_clean"].tolist()


def test_bias_sample_falls_back_when_no_freq():
    df = pd.DataFrame({"content_clean": [f"t{i}" for i in range(10)]})
    out = llm_label.bias_sample(df, 5)
    assert len(out) == 5


# ── agreement (G2) / batch ──────────────────────────────────────────────

def test_measure_agreement():
    pairs = [("sport", "sport"), ("sport", "politics"), ("politics", "politics")]
    assert llm_label.measure_agreement(pairs) == pytest.approx(2 / 3)


def test_measure_agreement_empty_is_zero():
    assert llm_label.measure_agreement([]) == 0.0


def test_label_batch_returns_records_and_dup_pairs():
    df = pd.DataFrame({"content_clean": [f"t{i}" for i in range(4)]})
    responses = ['{"category": "sport", "confidence": 0.9}'] * 4 + ['{"category": "politics", "confidence": 0.7}'] * 2
    client = FakeClient(responses)
    labeler = llm_label.GeminiLabeler(client, "gemini-2.5-pro", CATEGORIES, [])
    records, pairs = llm_label.label_batch(df, labeler, dup_fraction=0.5, seed=42)

    assert len(records) == 4
    assert records[0]["content"] == "t0"
    assert records[0]["category"] == "sport"
    assert len(pairs) == 2  # dup 50% ของ 4 แถว
    assert all(isinstance(p, tuple) for p in pairs)


# ── CLI end-to-end (mock API + tmp files) ──────────────────────────────

def test_main_writes_jsonl_and_agreement(tmp_path, monkeypatch):
    data = pd.DataFrame({"content_clean": [f"t{i}" for i in range(4)]})
    src = tmp_path / "preprocessed.parquet"
    data.to_parquet(src, index=False)
    cat = tmp_path / "categories.json"
    cat.write_text(json.dumps(CATEGORIES, ensure_ascii=False))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # 4 รอบแรก = round1, 2 รอบหลัง = dup round2 (dup 50%)
    responses = ['{"category": "sport", "confidence": 0.9}'] * 4 + ['{"category": "sport", "confidence": 0.8}'] * 2
    monkeypatch.setattr(llm_label, "make_client", lambda key=None: FakeClient(responses))

    out = tmp_path / "batch.jsonl"
    llm_label.main([
        "--input", str(src), "--output", str(out),
        "--category-file", str(cat), "--sample", "4", "--dup-fraction", "0.5",
    ])

    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 4
    assert all(set(r) == {"content", "category", "confidence"} for r in lines)

    agreement = json.loads((tmp_path / "agreement.json").read_text())
    assert "agreement_rate" in agreement
    assert agreement["agreement_rate"] == 1.0
