from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class InstitutionType(str, Enum):
    HigherEd = "HigherEd"
    K12 = "K12"
    Tutoring = "Tutoring"
    Unknown = "Unknown"


class InstructorInput(BaseModel):
    login_count: int = Field(..., ge=0)
    avg_session_time: float = Field(..., ge=0)
    assignments_created: int = Field(..., ge=0)
    student_count: int = Field(..., ge=0)
    engagement_score: float = Field(..., ge=0, le=1)
    institution_type: InstitutionType


class BatchPredictionRequest(BaseModel):
    instructors: List[InstructorInput]


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


class PredictionResult(BaseModel):
    prediction: int
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]


class ExplanationResponse(PredictionResponse):
    feature_importance: List[dict[str, float | str]]
