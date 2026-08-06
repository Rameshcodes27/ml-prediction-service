from __future__ import annotations

import pathlib
import sys

repository_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

from training.train import main as train_main

model_file = repository_root / "saved_models" / "random_forest_model.pkl"
encoder_file = repository_root / "saved_models" / "label_encoder.pkl"

if not model_file.exists() or not encoder_file.exists():
    train_main()

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Instructor Purchase Prediction"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


SAMPLE_PAYLOAD = {
    "login_count": 11,
    "avg_session_time": 57.95,
    "assignments_created": 8,
    "student_count": 101,
    "engagement_score": 0.4,
    "institution_type": "HigherEd",
}


def test_predict_endpoint() -> None:
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["confidence"], float)


def test_batch_predict_endpoint() -> None:
    payload = {"instructors": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert all("prediction" in item and "confidence" in item for item in data["predictions"])


def test_explain_endpoint() -> None:
    response = client.post("/explain", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["feature_importance"], list)
    assert len(data["feature_importance"]) == 6


def test_invalid_institution_type_validation() -> None:
    invalid_payload = SAMPLE_PAYLOAD.copy()
    invalid_payload["institution_type"] = "College"

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]
    assert any(
        "institution_type" in str(item)
        for item in data["detail"]
        if isinstance(item, dict)
    )
