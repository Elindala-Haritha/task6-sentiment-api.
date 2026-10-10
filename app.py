
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = tf.keras.models.load_model(
    MODEL_DIR / "sentiment_model.keras"
)
vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

app = FastAPI(
    title="Movie Review Sentiment API",
    description="Predicts whether a movie review is positive or negative",
    version="1.0.0"
)


class ReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: ReviewRequest):
    review = request.text.strip()

    if not review:
        raise HTTPException(
            status_code=422,
            detail="Review cannot be empty"
        )

    features = vectorizer.transform([review]).toarray()
    output = np.asarray(model.predict(features, verbose=0))

    if output.shape[-1] == 1:
        positive = float(output[0][0])
        probabilities = {
            "negative": 1.0 - positive,
            "positive": positive
        }
    elif output.shape[-1] == 2:
        probabilities = {
            "negative": float(output[0][0]),
            "positive": float(output[0][1])
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Unexpected model output"
        )

    sentiment = max(probabilities, key=probabilities.get)

    return {
        "sentiment": sentiment,
        "probabilities": probabilities
    }
