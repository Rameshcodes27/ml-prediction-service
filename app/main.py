from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from app.explain import explain_instructor
from app.predict import predict_batch, predict_single
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    InstructorInput,
    PredictionResponse,
)

app = FastAPI(
    title="Instructor Purchase Prediction Service",
    version="1.0.0",
    description="API for predicting premium package purchase likelihood for instructors.",
)


@app.get("/", response_model=dict)
def root() -> dict[str, str]:
    return {"service": "Instructor Purchase Prediction", "status": "running"}


@app.get("/health", response_model=dict)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single instructor purchase prediction",
    response_description="Predicted purchase label and confidence score",
)
def predict(
    input_data: InstructorInput = Body(
        ...,
        examples={
            "example": {
                "summary": "Single prediction",
                "description": "Predict whether one instructor will purchase the premium package.",
                "value": {
                    "login_count": 11,
                    "avg_session_time": 57.95,
                    "assignments_created": 8,
                    "student_count": 101,
                    "engagement_score": 0.4,
                    "institution_type": "HigherEd",
                },
            }
        },
    )
) -> PredictionResponse:
    try:
        prediction, confidence = predict_single(input_data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"prediction": prediction, "confidence": confidence}


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch instructor purchase prediction",
    response_description="Predicted purchase labels and confidence scores for multiple instructors",
)
def predict_batch_endpoint(
    request: BatchPredictionRequest = Body(
        ...,
        examples={
            "example": {
                "summary": "Batch prediction",
                "description": "Predict purchase likelihood for multiple instructors in one request.",
                "value": {
                    "instructors": [
                        {
                            "login_count": 11,
                            "avg_session_time": 57.95,
                            "assignments_created": 8,
                            "student_count": 101,
                            "engagement_score": 0.4,
                            "institution_type": "HigherEd",
                        },
                        {
                            "login_count": 9,
                            "avg_session_time": 65.21,
                            "assignments_created": 1,
                            "student_count": 46,
                            "engagement_score": 0.02,
                            "institution_type": "K12",
                        },
                    ]
                },
            }
        },
    )
) -> BatchPredictionResponse:
    try:
        predictions = predict_batch(request.instructors)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"predictions": predictions}


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Explainable prediction using SHAP",
    response_description="Prediction, confidence, and feature importance values",
)
def explain(
    input_data: InstructorInput = Body(
        ...,
        examples={
            "example": {
                "summary": "Explain prediction",
                "description": "Return SHAP explanations for a single instructor prediction.",
                "value": {
                    "login_count": 11,
                    "avg_session_time": 57.95,
                    "assignments_created": 8,
                    "student_count": 101,
                    "engagement_score": 0.4,
                    "institution_type": "HigherEd",
                },
            }
        },
    )
) -> ExplanationResponse:
    try:
        explanation = explain_instructor(input_data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return explanation
