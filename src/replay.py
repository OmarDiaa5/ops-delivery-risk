"""Simulate production: replay the held-out Apr-Aug 2018 orders through the
running API month by month, then build the monitoring report from the
prediction log (input drift per month, plus realized performance once the
delivery outcomes are known).

    uvicorn service.app:app --port 8000   # in one terminal
    python src/replay.py --url http://localhost:8000
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import load_log, performance_by_month  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--start", default="2018-04-01")
    args = ap.parse_args()

    con = sqlite3.connect(ROOT / "data" / "ops.db")
    ids = pd.read_sql("SELECT order_id, substr(purchase_ts, 1, 7) AS month, is_late "
                      "FROM order_features WHERE purchase_ts >= ? ORDER BY purchase_ts",
                      con, params=(args.start,))
    con.close()

    months = sorted(ids.month.unique())
    report = {"drift": {}, "performance": {}}
    with httpx.Client(base_url=args.url, timeout=30) as client:
        for month in months:
            rejected = 0
            for oid in ids.loc[ids.month == month, "order_id"]:
                r = client.get(f"/orders/{oid}/risk")
                if r.status_code == 422:
                    rejected += 1
                    continue
                r.raise_for_status()
            d = client.get("/monitoring/drift", params={"month": month}).json()
            report["drift"][month] = {"n": d["n"], "rejected": rejected, "drifted": d["drifted"],
                                      "top": d["features"][:5]}
            print(f"{month}: {d['n']} scored, {rejected} rejected, drifted={d['drifted']}")

    log = load_log(ROOT / "data" / "predictions.db")
    report["performance"] = performance_by_month(log, ids[["order_id", "is_late"]])
    for month, v in report["performance"].items():
        print(f"  {month}: {v}")
    (ROOT / "reports" / "monitoring.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
