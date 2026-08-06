from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from app.model import label_encoder, model
from app.schemas import InstructorInput

FEATURE_COLUMNS = [
    "login_count",
    "avg_session_time",
    "assignments_created",
    "student_count",
    "engagement_score",
    "institution_type",
]

NUMERIC_COLUMNS = [
    "login_count",
    "avg_session_time",
    "assignments_created",
    "student_count",
    "engagement_score",
]


def _prepare_features(inputs: List[InstructorInput]) -> pd.DataFrame:
    records = []
    for instructor in inputs:
        records.append(
            {
                "login_count": instructor.login_count,
                "avg_session_time": instructor.avg_session_time,
                "assignments_created": instructor.assignments_created,
                "student_count": instructor.student_count,
                "engagement_score": instructor.engagement_score,
                "institution_type": instructor.institution_type.value,
            }
        )

    frame = pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if frame[NUMERIC_COLUMNS].isnull().any().any():
        raise ValueError("Numeric fields must contain valid values.")

    try:
        frame["institution_type"] = label_encoder.transform(frame["institution_type"].astype(str))
    except ValueError as exc:
        raise ValueError(
            "Unknown institution_type encountered during prediction. "
            "Use one of: %s." % ", ".join(label_encoder.classes_)
        ) from exc

    return frame.astype(float)


def predict_single(instructor: InstructorInput) -> tuple[int, float]:
    features = _prepare_features([instructor])
    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0, 1])
    return prediction, confidence


def predict_batch(instructors: List[InstructorInput]) -> list[dict[str, float | int]]:
    features = _prepare_features(instructors)
    predictions = model.predict(features)
    confidences = model.predict_proba(features)[:, 1]
    return [
        {"prediction": int(prediction), "confidence": float(confidence)}
        for prediction, confidence in zip(predictions, confidences)
    ]
