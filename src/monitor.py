"""Drift and performance monitoring over the API's prediction log.

PSI (population stability index) compares each feature's and the score's
distribution in a window of logged traffic against the training reference.
Rule of thumb: < 0.10 stable, 0.10-0.25 watch, > 0.25 drifted."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CATEGORICAL, NUMERIC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-4
# Calendar features shift every month by construction; report them, but don't
# raise them as drift alarms.
SEASONAL = {"purchase_month", "purchase_dow", "purchase_hour"}


def psi(expected, actual):
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value):
    return "stable" if value < 0.10 else "watch" if value < 0.25 else "drift"


def numeric_shares(values, edges):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    counts, _ = np.histogram(v, bins=np.r_[-np.inf, edges, np.inf])
    return counts / max(counts.sum(), 1)


def drift_report(window, profile):
    """window: DataFrame of logged rows (feature columns + 'score')."""
    rows = []
    for col in NUMERIC + ["score"]:
        ref = profile["numeric"][col]
        value = psi(ref["shares"], numeric_shares(window[col], ref["edges"]))
        rows.append({"feature": col, "psi": round(value, 4),
                     "status": "seasonal" if col in SEASONAL else status(value)})
    for col in CATEGORICAL:
        ref = profile["categorical"][col]
        cur = window[col].value_counts(normalize=True)
        cats = sorted(set(ref) | set(cur.index))
        value = psi([ref.get(c, 0) for c in cats], [cur.get(c, 0) for c in cats])
        rows.append({"feature": col, "psi": round(value, 4), "status": status(value)})
    rows.sort(key=lambda r: -r["psi"])
    return {"n": len(window), "drifted": [r["feature"] for r in rows if r["status"] == "drift"],
            "features": rows}


def load_log(log_db):
    con = sqlite3.connect(log_db)
    df = pd.read_sql("SELECT * FROM prediction_log", con)
    con.close()
    feats = pd.json_normalize(df["features"].map(json.loads))
    return pd.concat([df.drop(columns="features"), feats], axis=1)


def performance_by_month(log, labels):
    """Once deliveries arrive, outcomes are known: track how many late orders
    the flagged set actually caught, month by month."""
    from sklearn.metrics import roc_auc_score
    m = log.merge(labels, on="order_id")
    out = {}
    for month, g in m.groupby("event_month"):
        if g["is_late"].nunique() < 2:
            continue
        out[month] = {
            "n": len(g), "late_rate": round(g["is_late"].mean(), 4),
            "roc_auc": round(roc_auc_score(g["is_late"], g["score"]), 4),
            "flag_recall": round(g.loc[g.flagged == 1, "is_late"].sum() / g["is_late"].sum(), 4),
            "flag_precision": round(g.loc[g.flagged == 1, "is_late"].mean(), 4),
        }
    return out
