# Late-Delivery Risk: from order data to an ops review queue

Flags e-commerce orders likely to arrive after the date promised to the customer, at the moment the order is placed, so an operations team can act before the delay happens.

Built on the public [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 96,470 delivered orders (Sep 2016 - Aug 2018) across 9 related tables.

```
CSV tables ──► SQLite ──► SQL feature query ──► train.py (3 models + rule baseline, MLflow)
                                                     │
                                       late_risk.joblib + reference profile
                                                     │
   ASP.NET Core gateway  ──HTTP──►  FastAPI risk service ──► prediction log ──► drift + performance monitor
   (review queue, actions,          (validation, top drivers)                   (PSI per feature, monthly recall)
    dashboard)
```

## Why it matters

From the order data (`src/insights.py`, plain SQL):

- Late orders average **2.57 stars** vs **4.29** for on-time orders, and **54%** of late orders get a 1-2 star review vs 9.2% of on-time orders.
- Cross-state orders run late 9.3% of the time vs 6.1% within a state; the worst destination state (MA) runs 19.7% late.

## Results

Time-based split, no shuffling: train on 2017, choose the model on Jan-Mar 2018, score the untouched Apr-Aug 2018 months once.

| Test, Apr-Aug 2018 (32,150 orders, 6.0% late) | ROC AUC | Late orders caught when reviewing the riskiest 10% |
|---|---|---|
| Rule: flag sellers with the worst late history | 0.52 | 12.5% |
| **Logistic regression (chosen)** | **0.70** | **23.2%** (2.3x lift) |

At the same review budget the model catches **1.9x** as many late orders as the rule.

The gradient-boosted models (HistGB, XGBoost) scored lower on validation (ROC AUC 0.65-0.66 vs 0.705). The late rate swings from 1.4% to 21% month to month (Black Friday 2017, Feb-Mar 2018), and the trees fit 2017 too closely. Full comparison in [REPORT.md](REPORT.md) and MLflow.

Score is a risk ranking from a class-balanced model, not a calibrated probability.

## Leakage controls

- Only features known when the order is placed: promised window, seller-customer distance (zip-centroid haversine), route, basket size/weight/volume, freight, payment, purchase time, category.
- Seller late history counts **only deliveries that had already arrived before the purchase** (`merge_asof` on delivery time, smoothed toward the global rate). Unit-tested in `tests/test_features.py`.
- Test months are scored once, after model choice.

## Monitoring

`src/replay.py` replays the test months through the running API like production traffic. Every prediction is logged; `/monitoring/drift?month=YYYY-MM` reports PSI per input feature and for the score against the training reference, and once delivery outcomes arrive `monitor.performance_by_month` tracks how many late orders the flagged set actually caught. See section 3 of [REPORT.md](REPORT.md).

What the replay showed (32,150 orders, Apr-Aug 2018, none rejected after the data fix below):

- The score's PSI against training rose from 0.19 (Apr) to **0.93** (Aug); seller late history and freight cost drifted every month.
- With the threshold fixed at training time (top 10%), the share of orders flagged climbed from **18% to 36%**, and monthly ROC AUC fell from **0.77 to 0.58**. Both are retraining triggers an ops team can see before the review queue becomes unmanageable.
- The replay also surfaced a data bug: 42 of 1M geolocation points lie outside Brazil, making neighbouring-state orders look 8,678 km apart. The API's validation rejected those orders; the fix filters coordinates to Brazil's bounding box.

## Run it

```bash
pip install -r requirements-dev.txt
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
python src/build_features.py      # SQLite + SQL features (~15 s)
python src/train.py               # models, MLflow runs, saved model (~30 s)
python src/insights.py            # SQL business findings
uvicorn service.app:app --port 8000
python src/replay.py              # production replay + monitoring
python src/report.py              # REPORT.md

cd gateway/OpsGateway && RISK_API_URL=http://localhost:8000 dotnet run   # dashboard on :5000
```

Or `docker compose up` for both services. CI (GitHub Actions) runs the Python and .NET tests and builds and smoke-tests both containers.

## API

| Service | Endpoint | |
|---|---|---|
| FastAPI | `POST /predict` | score raw order fields (validated) |
| FastAPI | `GET /orders/{id}/risk` | score an order from the feature store |
| FastAPI | `GET /monitoring/drift?month=` | PSI drift report over logged traffic |
| ASP.NET Core | `POST /api/orders/review-queue` | rank up to 500 orders, attach the next action |
| ASP.NET Core | `GET /api/orders/{id}/risk` | one order + recommended action |
| ASP.NET Core | `GET /` | review-queue and drift dashboard |
