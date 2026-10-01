import pandas as pd
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
row = pd.read_csv("data/GermanCredit.csv").drop(columns="credit_risk").iloc[0].to_dict()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json={"features": row})
    assert r.status_code == 200
    assert 0 <= r.json()["default_probability"] <= 1


def test_predict_missing_field():
    bad = dict(row)
    bad.pop("duration")
    assert client.post("/predict", json={"features": bad}).status_code == 422
