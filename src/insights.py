"""Operational findings from the order data, answered in SQL: what a late
delivery costs in customer satisfaction, and where lateness concentrates."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUERIES = {
    "review_by_lateness": """
        SELECT f.is_late, COUNT(*) AS orders,
               ROUND(AVG(r.review_score), 2) AS avg_review,
               ROUND(AVG(r.review_score <= 2) * 100, 1) AS pct_1_2_star
        FROM order_features f
        JOIN (SELECT order_id, AVG(review_score) AS review_score
              FROM order_reviews GROUP BY order_id) r ON r.order_id = f.order_id
        GROUP BY f.is_late""",
    "late_rate_by_customer_state": """
        SELECT customer_state, COUNT(*) AS orders,
               ROUND(AVG(is_late) * 100, 1) AS late_pct
        FROM order_features GROUP BY customer_state
        HAVING orders >= 500 ORDER BY late_pct DESC LIMIT 5""",
    "late_rate_by_promised_window": """
        SELECT CASE WHEN promised_days < 15 THEN 'a: <15 days'
                    WHEN promised_days < 25 THEN 'b: 15-24 days'
                    WHEN promised_days < 35 THEN 'c: 25-34 days'
                    ELSE 'd: 35+ days' END AS promised_window,
               COUNT(*) AS orders, ROUND(AVG(is_late) * 100, 1) AS late_pct
        FROM order_features GROUP BY promised_window ORDER BY promised_window""",
    "late_rate_by_route": """
        SELECT CASE same_state WHEN 1 THEN 'same state' ELSE 'cross state' END AS route,
               COUNT(*) AS orders, ROUND(AVG(is_late) * 100, 1) AS late_pct
        FROM order_features GROUP BY same_state""",
}


def main():
    con = sqlite3.connect(ROOT / "data" / "ops.db")
    con.row_factory = sqlite3.Row
    out = {}
    for name, sql in QUERIES.items():
        rows = [dict(r) for r in con.execute(sql)]
        out[name] = rows
        print(f"\n{name}")
        for r in rows:
            print("  ", r)
    (ROOT / "reports" / "insights.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
