from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    response = client.post(
        "/predict",
        json={"text": "This movie was amazing and wonderful!"}
    )

    assert response.status_code == 200

    result = response.json()
    assert result["sentiment"] in ["positive", "negative"]

    probabilities = result["probabilities"]
    assert set(probabilities.keys()) == {"positive", "negative"}
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1) < 1e-5


def test_empty_review():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
