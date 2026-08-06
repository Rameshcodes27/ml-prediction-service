from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(__file__).resolve().parents[1] / "saved_models"
MODEL_FILE = MODEL_DIR / "random_forest_model.pkl"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"


def load_artifacts() -> tuple[RandomForestClassifier, LabelEncoder]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model artifact not found. Run training first.")
    if not ENCODER_FILE.exists():
        raise FileNotFoundError("Label encoder artifact not found. Run training first.")

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    return model, encoder


model, label_encoder = load_artifacts()
