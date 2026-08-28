import textcls.config as cfg


def test_data_dir_points_to_project_data():
    assert cfg.DATA_DIR.name == "data"
    assert cfg.CATEGORIES_PATH == cfg.DATA_DIR / "categories.json"


def test_model_dir_points_to_models():
    assert cfg.MODEL_DIR.name == "models"


def test_g4_confidence_threshold_default():
    assert cfg.CONFIDENCE_THRESHOLD == 0.6
