"""Task 2 — preprocess: clean Thai Twitter text.

ครอบคลุมตาม spec Testing Strategy: ตัวซ้ำ→[CREP], 555→[LAUGH], ตัด # เก็บคำ, ตัด URL/@
"""

import pandas as pd
import pytest

from textcls.preprocess import clean_text, preprocess


def test_url_and_mention_stripped():
    assert clean_text("ดูคลิปนี้ https://t.co/xyz @someone") == "ดูคลิปนี้"


def test_www_url_stripped():
    assert clean_text("อ่านที่ www.example.com นะ") == "อ่านที่ นะ"


def test_repeated_char_to_crep():
    assert clean_text("ดีมากกกก") == "ดีมาก[CREP]"


def test_laugh_555_to_laugh():
    assert clean_text("55555 ขำ") == "[LAUGH] ขำ"


def test_hashtag_keeps_word():
    assert clean_text("#SCKCatalogue") == "SCKCatalogue"


def test_whitespace_collapsed_and_trimmed():
    assert clean_text("  มาก   ๆ  ") == "มาก ๆ"


def test_preprocess_writes_parquet_with_clean_column(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "preprocessed.parquet"
    pd.DataFrame({"content": ["ดีมากกกก https://t.co/x", "55555 #sport"]}).to_csv(
        raw, index=False, encoding="utf-8-sig"
    )

    n = preprocess(raw, out)

    assert n == 2
    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == ["content_clean"]
    assert df["content_clean"].tolist() == ["ดีมาก[CREP]", "[LAUGH] sport"]


def test_empty_text_ok():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
