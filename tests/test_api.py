from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Breast Cancer Classification API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict(monkeypatch):
    # Mock the ML prediction so the test does not depend on
    # MLflow/model availability during CI testing.
    def mock_predict(features):
        return 1, 0.95

    monkeypatch.setattr("src.app.predict", mock_predict)

    features = [0.0] * 30

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] == 1
    assert result["probability"] == 0.95