"""Train and compare late-delivery models on a time-based split, track every
run in MLflow, and save the chosen model plus the reference distributions the
drift monitor compares production traffic against."""
import json
import os
import sqlite3
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "8")

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CATEGORICAL, FEATURES, NUMERIC, TARGET  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "ops.db"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_END = "2018-01-01"   # train: 2017 purchases
VALID_END = "2018-04-01"   # valid: Q1 2018, test: Apr-Aug 2018
FLAG_SHARE = 0.10          # ops team can review the riskiest 10% of orders
SEED = 42


def load():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM order_features", con, parse_dates=["purchase_ts", "delivered_ts"])
    con.close()
    return df[df.purchase_ts >= "2017-01-01"]


def split(df):
    tr = df[df.purchase_ts < TRAIN_END]
    va = df[(df.purchase_ts >= TRAIN_END) & (df.purchase_ts < VALID_END)]
    te = df[df.purchase_ts >= VALID_END]
    return tr, va, te


def preprocessor(scale):
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), CATEGORICAL),
    ])


def candidates(pos_weight):
    return {
        "logreg": Pipeline([("prep", preprocessor(scale=True)),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "hist_gb": Pipeline([("prep", preprocessor(scale=False)),
                             ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                                    class_weight="balanced", random_state=SEED))]),
        "xgboost": Pipeline([("prep", preprocessor(scale=False)),
                             ("clf", XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6,
                                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                                   scale_pos_weight=pos_weight, eval_metric="aucpr",
                                                   n_jobs=8, random_state=SEED))]),
    }


def metrics(y, p):
    k = int(len(p) * FLAG_SHARE)
    top = np.argsort(-p)[:k]
    caught = y.iloc[top].sum()
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "recall_at_10pct": caught / y.sum(),
        "precision_at_10pct": caught / k,
        "lift_at_10pct": (caught / k) / y.mean(),
        "base_rate": y.mean(),
    }


def rule_baseline(df):
    """What an ops team without a model would do: flag the sellers with the
    worst track record. Score = seller's historical late rate."""
    return df["seller_hist_late_rate"].to_numpy()


def reference_profile(X, scores):
    """Decile bin edges + shares for each numeric feature and the score, and
    category shares, all from training data. The monitor computes PSI vs this."""
    prof = {"numeric": {}, "categorical": {}}
    for col in NUMERIC + ["score"]:
        v = np.asarray(scores if col == "score" else X[col], dtype=float)
        v = v[~np.isnan(v)]
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))[1:-1]
        counts, _ = np.histogram(v, bins=np.r_[-np.inf, edges, np.inf])
        prof["numeric"][col] = {"edges": edges.tolist(), "shares": (counts / counts.sum()).tolist()}
    for col in CATEGORICAL:
        prof["categorical"][col] = X[col].value_counts(normalize=True).to_dict()
    return prof


def main():
    df = load()
    tr, va, te = split(df)
    print(f"train {len(tr):,} ({tr[TARGET].mean():.3f} late) | valid {len(va):,} "
          f"({va[TARGET].mean():.3f}) | test {len(te):,} ({te[TARGET].mean():.3f})")

    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("late-delivery-risk")
    pos_weight = (1 - tr[TARGET].mean()) / tr[TARGET].mean()

    results = {}
    with mlflow.start_run(run_name="rule_seller_late_rate"):
        m = metrics(va[TARGET], rule_baseline(va))
        mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()})
        results["rule_seller_late_rate"] = {"valid": m}
        print(f"{'rule':8s} valid ROC AUC {m['roc_auc']:.3f}  PR AUC {m['pr_auc']:.3f}  "
              f"recall@10% {m['recall_at_10pct']:.3f}")

    fitted = {}
    for name, pipe in candidates(pos_weight).items():
        with mlflow.start_run(run_name=name):
            pipe.fit(tr[FEATURES], tr[TARGET])
            m = metrics(va[TARGET], pipe.predict_proba(va[FEATURES])[:, 1])
            mlflow.log_params({"model": name, "train_end": TRAIN_END, "valid_end": VALID_END,
                               "n_train": len(tr), "n_features": len(FEATURES)})
            mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()})
            results[name] = {"valid": m}
            fitted[name] = pipe
            print(f"{name:8s} valid ROC AUC {m['roc_auc']:.3f}  PR AUC {m['pr_auc']:.3f}  "
                  f"recall@10% {m['recall_at_10pct']:.3f}")

    # Pick on validation PR AUC (the label is rare), refit on train+valid and
    # score the untouched test months exactly once.
    best = max(fitted, key=lambda n: results[n]["valid"]["pr_auc"])
    final = candidates(pos_weight)[best]
    trva = pd.concat([tr, va])
    final.fit(trva[FEATURES], trva[TARGET])
    p_test = final.predict_proba(te[FEATURES])[:, 1]
    test_m = metrics(te[TARGET], p_test)
    rule_test = metrics(te[TARGET], rule_baseline(te))
    with mlflow.start_run(run_name=f"final_{best}"):
        mlflow.log_params({"model": best, "fit_on": "train+valid"})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_metrics({f"test_rule_{k}": v for k, v in rule_test.items()})
    results["final"] = {"model": best, "test": test_m, "test_rule_baseline": rule_test}

    # Monthly test breakdown: shows how ranking holds up as the late rate moves.
    te = te.assign(score=p_test)
    monthly = {}
    for month, g in te.groupby(te.purchase_ts.dt.to_period("M")):
        if g[TARGET].nunique() == 2:
            monthly[str(month)] = {"n": len(g), "late_rate": g[TARGET].mean(),
                                   "roc_auc": roc_auc_score(g[TARGET], g["score"])}
    results["final"]["test_monthly"] = monthly

    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "late_risk.joblib")
    ref_scores = final.predict_proba(trva[FEATURES])[:, 1]
    profile = reference_profile(trva[FEATURES], ref_scores)
    profile["flag_threshold"] = float(np.quantile(ref_scores, 1 - FLAG_SHARE))
    (MODELS / "reference_profile.json").write_text(json.dumps(profile, indent=1))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "experiments.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"best={best}  TEST ROC AUC {test_m['roc_auc']:.3f}  PR AUC {test_m['pr_auc']:.3f}  "
          f"recall@10% {test_m['recall_at_10pct']:.3f}  lift {test_m['lift_at_10pct']:.2f}x  "
          f"(rule recall@10% {rule_test['recall_at_10pct']:.3f})")
    for mth, v in monthly.items():
        print(f"  {mth}: n={v['n']} late={v['late_rate']:.3f} auc={v['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
