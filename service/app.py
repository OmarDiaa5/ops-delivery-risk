"""Late-delivery risk API. Scores an order, explains the top drivers, logs
every prediction, and serves a drift report over the log."""
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES, NUMERIC  # noqa: E402
from monitor import drift_report  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "late_risk.joblib"))
PROFILE_PATH = Path(os.getenv("PROFILE_PATH", ROOT / "models" / "reference_profile.json"))
FEATURE_DB = Path(os.getenv("FEATURE_DB", ROOT / "data" / "ops.db"))
LOG_DB = Path(os.getenv("LOG_DB", ROOT / "data" / "predictions.db"))

model = joblib.load(MODEL_PATH)
profile = json.loads(PROFILE_PATH.read_text())
THRESHOLD = profile["flag_threshold"]

app = FastAPI(title="Late-delivery risk", version="1.0")


class Order(BaseModel):
    order_id: str | None = None
    event_month: str | None = Field(None, pattern=r"^\d{4}-\d{2}$",
                                    description="Month the order was placed (for monitoring windows)")
    promised_days: float = Field(..., gt=0, le=180)
    distance_km: float | None = Field(None, ge=0, le=5000)
    same_state: Literal[0, 1]
    n_items: int = Field(..., ge=1, le=50)
    n_sellers: int = Field(..., ge=1, le=10)
    total_price: float = Field(..., gt=0)
    total_freight: float = Field(..., ge=0)
    freight_ratio: float = Field(..., ge=0)
    total_weight_g: float = Field(..., ge=0)
    total_volume_cm3: float = Field(..., ge=0)
    installments: int = Field(..., ge=0, le=24)
    purchase_dow: int = Field(..., ge=0, le=6)
    purchase_hour: int = Field(..., ge=0, le=23)
    purchase_month: int = Field(..., ge=1, le=12)
    seller_hist_orders: float = Field(..., ge=0)
    seller_hist_late_rate: float = Field(..., ge=0, le=1)
    customer_state: str = Field(..., min_length=2, max_length=2)
    seller_state: str = Field(..., min_length=2, max_length=2)
    main_category: str
    payment_type: str


class Risk(BaseModel):
    order_id: str | None
    late_risk: float
    flagged: bool
    tier: Literal["low", "medium", "high"]
    top_drivers: list[str]


def _log_conn():
    con = sqlite3.connect(LOG_DB, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""CREATE TABLE IF NOT EXISTS prediction_log (
        logged_at TEXT, order_id TEXT, event_month TEXT,
        score REAL, flagged INTEGER, model_version TEXT, features TEXT)""")
    return con


PREP, CLF = model.named_steps["prep"], model.named_steps["clf"]
FEATURE_NAMES = [n.split("__", 1)[1] for n in PREP.get_feature_names_out()]
_log = _log_conn()
_log_lock = threading.Lock()


def _drivers(x):
    """Top features pushing the score up, from the logistic model's
    per-feature contributions (coef x transformed value)."""
    if not hasattr(CLF, "coef_"):
        return []
    contrib = CLF.coef_.ravel() * x.ravel()
    top = np.argsort(-contrib)[:3]
    # numeric drivers say which way they push: promised_days_low, distance_km_high
    return [FEATURE_NAMES[i] + (("_high" if x.ravel()[i] > 0 else "_low") if FEATURE_NAMES[i] in NUMERIC else "")
            for i in top if contrib[i] > 0]


def score(order: Order) -> Risk:
    row = pd.DataFrame([order.model_dump()])[FEATURES]
    x = np.asarray(PREP.transform(row))
    p = float(CLF.predict_proba(x)[:, 1][0])
    flagged = p >= THRESHOLD
    tier = "high" if flagged else "medium" if p >= THRESHOLD / 2 else "low"
    with _log_lock:
        _log.execute("INSERT INTO prediction_log VALUES (?,?,?,?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(), order.order_id, order.event_month,
                      p, int(flagged), app.version, json.dumps(row.iloc[0].to_dict(), default=float)))
        _log.commit()
    return Risk(order_id=order.order_id, late_risk=round(p, 4), flagged=flagged,
                tier=tier, top_drivers=_drivers(x))


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model.named_steps["clf"]).__name__,
            "flag_threshold": round(THRESHOLD, 4)}


@app.post("/predict", response_model=Risk)
def predict(order: Order):
    return score(order)


@app.get("/orders/{order_id}/risk", response_model=Risk)
def order_risk(order_id: str):
    try:
        con = sqlite3.connect(f"file:{FEATURE_DB.as_posix()}?mode=ro", uri=True)
        row = pd.read_sql("SELECT * FROM order_features WHERE order_id = ?", con, params=(order_id,))
        con.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        raise HTTPException(503, "feature store unavailable")
    if row.empty:
        raise HTTPException(404, "order not found")
    r = row.iloc[0]
    fields = {k: (None if pd.isna(r[k]) else r[k]) for k in FEATURES}
    try:
        order = Order(order_id=order_id, event_month=str(r["purchase_ts"])[:7], **fields)
    except ValidationError as e:
        raise HTTPException(422, f"stored order failed validation: {e.errors()[0]['loc']}")
    return score(order)


@app.get("/monitoring/drift")
def drift(month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$")):
    from monitor import load_log
    if not LOG_DB.exists():
        raise HTTPException(404, "no predictions logged yet")
    log = load_log(LOG_DB)
    if month:
        log = log[log.event_month == month]
    if log.empty:
        raise HTTPException(404, "no predictions in that window")
    return drift_report(log, profile)
