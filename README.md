# Instructor Purchase Prediction Service

A production-ready Machine Learning Prediction API built with FastAPI. The service predicts whether an instructor is likely to purchase a premium package based on behavioral metrics and institution type.

## Features

- Train a Random Forest classifier on instructor behavior data
- Single and batch prediction endpoints
- SHAP explainability for model decisions
- MLflow experiment tracking
- Docker-ready deployment

## Architecture

- `training/`: data preprocessing, model training, evaluation
- `app/`: API, prediction logic, model loading, explanation logic
- `tests/`: API contract tests
- `data/`: dataset source file
- `saved_models/`: persisted model and encoder artifacts

## Installation

1. Clone the repository:

```bash
git clone <repo-url>
cd ml-prediction-service
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

Train the model and save artifacts:

```bash
python -m training.train
```

## Running the API

Start the FastAPI service locally:

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

## Docker

Build and run with Docker:

```bash
docker build -t instructor-purchase-service .
docker run -p 8000:8000 instructor-purchase-service
```

Or use Docker Compose:

```bash
docker compose up --build
```

## MLflow

Training logs are stored in the local `mlruns/` directory by default. Use MLflow tracking to inspect parameters, metrics, and model artifacts.

## API Endpoints

- `GET /` - service metadata
- `GET /health` - health check
- `POST /predict` - single prediction
- `POST /predict/batch` - batch predictions
- `POST /explain` - SHAP-based explanation

### Prediction Payload

```json
{
  "login_count": 11,
  "avg_session_time": 57.95,
  "assignments_created": 8,
  "student_count": 101,
  "engagement_score": 0.4,
  "institution_type": "HigherEd"
}
```

## Testing

Run tests with:

```bash
pytest
```
