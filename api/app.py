from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model_data = joblib.load(MODEL_PATH)


# ============================================================
# SUPPORT BOTH MODEL FORMATS
# ============================================================

if isinstance(model_data, dict):
    model = model_data.get("model")
    vectorizer = model_data.get("vectorizer")
    model_name = model_data.get("model_name", "best_model")
else:
    model = model_data
    vectorizer = None
    model_name = "best_model"


if model is None:
    raise ValueError("Model could not be loaded.")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Sentiment prediction using the best trained ML model",
    version="1.0.0",
)


# ============================================================
# PROMETHEUS
# ============================================================

Instrumentator().instrument(app).expose(app)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    text: str


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class PredictionResponse(BaseModel):
    text: str
    prediction: int
    sentiment: str
    model: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "message": "IMDB Sentiment Analysis API",
        "status": "running",
        "model": model_name,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": model_name,
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    try:

        # ----------------------------------------
        # TF-IDF model
        # ----------------------------------------

        if vectorizer is not None:

            transformed_text = vectorizer.transform([text])

            prediction = model.predict(
                transformed_text
            )[0]

        # ----------------------------------------
        # Pipeline model
        # ----------------------------------------

        else:

            prediction = model.predict([text])[0]

        prediction = int(prediction)

        sentiment = (
            "positive"
            if prediction == 1
            else "negative"
        )

        return PredictionResponse(
            text=text,
            prediction=prediction,
            sentiment=sentiment,
            model=model_name,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )



