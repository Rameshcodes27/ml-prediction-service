from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

NUMERIC_COLUMNS = [
    "login_count",
    "avg_session_time",
    "assignments_created",
    "student_count",
    "engagement_score",
]

CATEGORICAL_COLUMN = "institution_type"
TARGET_COLUMN = "purchased_package"


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = df.copy()
    if "instructor_id" in df.columns:
        df = df.drop(columns=["instructor_id"])

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[CATEGORICAL_COLUMN] = df[CATEGORICAL_COLUMN].fillna("Unknown").astype(str)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    imputer = SimpleImputer(strategy="mean")
    df[NUMERIC_COLUMNS] = imputer.fit_transform(df[NUMERIC_COLUMNS])

    label_encoder = LabelEncoder()
    df[CATEGORICAL_COLUMN] = label_encoder.fit_transform(df[CATEGORICAL_COLUMN])

    X = df[NUMERIC_COLUMNS + [CATEGORICAL_COLUMN]]
    y = df[TARGET_COLUMN]
    return X, y, label_encoder
