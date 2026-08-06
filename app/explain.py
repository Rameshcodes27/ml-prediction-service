from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from shap import TreeExplainer

from app.model import model
from app.predict import FEATURE_COLUMNS, _prepare_features
from app.schemas import InstructorInput


def explain_instructor(instructor: InstructorInput) -> dict[str, object]:
    features = _prepare_features([instructor])
    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0, 1])

    feature_frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(feature_frame)

    if isinstance(shap_values, list):
        shap_values_for_positive = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif shap_values.ndim == 2 and shap_values.shape[0] == len(FEATURE_COLUMNS) and shap_values.shape[1] == 2:
        shap_values_for_positive = shap_values[:, 1]
    elif shap_values.ndim == 2 and shap_values.shape[0] == 1:
        shap_values_for_positive = shap_values[0]
    elif shap_values.ndim == 3 and shap_values.shape[0] == 1 and shap_values.shape[2] == 2:
        shap_values_for_positive = shap_values[0, :, 1]
    else:
        shap_values_for_positive = shap_values.flatten()

    importance: List[dict[str, float | str]] = [
        {
            "feature": FEATURE_COLUMNS[index],
            "value": float(shap_values_for_positive[index]),
        }
        for index in range(len(FEATURE_COLUMNS))
    ]

    importance.sort(key=lambda item: abs(item["value"]), reverse=True)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "feature_importance": importance,
    }
