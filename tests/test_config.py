import pytest

import textcls.config as cfg


def test_data_dir_points_to_project_data():
    assert cfg.DATA_DIR.name == "data"
    assert cfg.CATEGORIES_PATH == cfg.DATA_DIR / "categories.json"


def test_model_dir_points_to_models():
    assert cfg.MODEL_DIR.name == "models"


def test_g1_precision_threshold_is_0_85():
    assert cfg.PRECISION_THRESHOLD == 0.85


def test_g4_confidence_threshold_default():
    assert cfg.CONFIDENCE_THRESHOLD == 0.6


def test_gemini_api_key_reads_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert cfg.get_gemini_api_key() == "test-key"


def test_gemini_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        cfg.get_gemini_api_key()
