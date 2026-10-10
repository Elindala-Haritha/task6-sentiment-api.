# task6-sentiment-api.

# Real-Time ML Inference REST API

## Project
A FastAPI service that predicts positive or negative sentiment from movie reviews using a trained TensorFlow model and TF-IDF vectorizer.

## Architecture
Review text -> TF-IDF vectorizer -> neural network -> sentiment probabilities.

## Endpoints
- GET / : API welcome message
- GET /health : health status
- POST /predict : predict sentiment from JSON review text

## Example request
```json
{
  "text": "This movie was amazing!"
}
```

## Install
```bash
pip install -r requirements.txt
```

## Run API
```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

## Run tests
```bash
pytest -v
```

## Docker
```bash
docker build -t task6-sentiment-api .
docker run --rm -p 8000:8000 task6-sentiment-api
```

## Model files
The model directory contains sentiment_model.keras and tfidf_vectorizer.joblib.

## Limitations
Predictions depend on the training data and model quality.
