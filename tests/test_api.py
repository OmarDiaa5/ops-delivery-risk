import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["LOG_DB"] = str(tmp_path_factory.mktemp("log") / "predictions.db")
    from service.app import app
    return TestClient(app)


ORDER = {
    "promised_days": 20, "distance_km": 850, "same_state": 0, "n_items": 1, "n_sellers": 1,
    "total_price": 120, "total_freight": 25, "freight_ratio": 0.21, "total_weight_g": 900,
    "total_volume_cm3": 6000, "installments": 3, "purchase_dow": 2, "purchase_hour": 14,
    "purchase_month": 5, "seller_hist_orders": 40, "seller_hist_late_rate": 0.1,
    "customer_state": "RJ", "seller_state": "SP", "main_category": "housewares",
    "payment_type": "credit_card",
}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_bounded_score(client):
    r = client.post("/predict", json=ORDER).json()
    assert 0 <= r["late_risk"] <= 1
    assert r["tier"] in {"low", "medium", "high"}


def test_risky_seller_scores_higher(client):
    safe = client.post("/predict", json=ORDER).json()["late_risk"]
    risky = client.post("/predict", json={**ORDER, "seller_hist_late_rate": 0.6}).json()["late_risk"]
    assert risky > safe


@pytest.mark.parametrize("field,value", [("promised_days", -1), ("purchase_hour", 25),
                                         ("seller_hist_late_rate", 1.5), ("same_state", 2)])
def test_rejects_invalid_input(client, field, value):
    assert client.post("/predict", json={**ORDER, field: value}).status_code == 422


def test_unknown_category_still_scores(client):
    assert client.post("/predict", json={**ORDER, "main_category": "never_seen"}).status_code == 200


def test_predictions_are_logged_and_monitored(client):
    client.post("/predict", json={**ORDER, "event_month": "2018-05"})
    d = client.get("/monitoring/drift", params={"month": "2018-05"}).json()
    assert d["n"] >= 1 and {f["feature"] for f in d["features"]} >= {"score", "promised_days"}


def test_order_lookup_404(client):
    # 503 when the feature store is not built (e.g. in CI)
    assert client.get("/orders/does-not-exist/risk").status_code in {404, 503}
