from fastapi.testclient import TestClient
from src.app.service import app

client = TestClient(app)


def test_status():
    assert client.get("/status").status_code == 200


def test_predict():
    id_request = "test_id"
    request_payload = {
        "id": id_request,
        "transaction": {
            "id": id_request,
            "amount": 1500,
            "transaction_type": "purchase",
            "code": "a43",
        },
    }
    response = client.post("/predict", json=request_payload)
    print(response)

    assert response.status_code == 200

    response_data = response.json()
    assert "id" in response_data
    assert "id_request" in response_data
    assert "is_fraud_prob" in response_data

    assert response_data["id_request"] == id_request
