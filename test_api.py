from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_minimal_payload():
    r = client.post("/predict", json={"Time": 0.0, "Amount": 0.0, "Hour": 0})
    assert r.status_code == 200
    data = r.json()
    assert set(["prob","pred","threshold","missing_features"]).issubset(data)
    assert isinstance(data["prob"], float)
    assert data["pred"] in (0, 1)
