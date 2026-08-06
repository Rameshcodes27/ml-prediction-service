from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from training.evaluate import evaluate_model
from training.preprocess import load_data, preprocess_data


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    data_path = repository_root / "data" / "instructor_behavior_dataset.csv"
    models_dir = repository_root / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X, y, label_encoder = preprocess_data(df)

    split_args = {
        "test_size": 0.2,
        "random_state": 42,
    }
    if len(y.unique()) > 1:
        split_args["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_args)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = evaluate_model(y_test.to_numpy(), y_pred, y_proba)

    mlflow.set_experiment("Instructor Purchase Prediction")
    with mlflow.start_run(run_name="random_forest_classifier"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, models_dir / "random_forest_model.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")

    print("Training complete. Model and encoder saved to saved_models.")


if __name__ == "__main__":
    main()
