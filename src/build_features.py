"""Load the Olist CSVs into SQLite, build order-level features in SQL, then add
the two features SQL can't do cleanly: seller-customer distance and the
seller's late-delivery history as of each purchase (no look-ahead)."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "ops.db"

TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Smoothing prior for the seller late rate: a seller with few past deliveries
# is pulled toward the global rate instead of showing 0% or 100%.
PRIOR_N = 20


def load_raw(con):
    for table, fname in TABLES.items():
        pd.read_csv(RAW / fname).to_sql(table, con, if_exists="replace", index=False)
    for table, col in [("order_items", "order_id"), ("order_payments", "order_id"),
                       ("orders", "customer_id"), ("products", "product_id"),
                       ("sellers", "seller_id"), ("customers", "customer_id")]:
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_{col} ON {table}({col})")


def zip_centroids():
    geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")
    # 42 of the 1M points fall outside Brazil (bad coordinates in the source);
    # they made neighbouring-state orders look 8,000+ km apart.
    in_brazil = geo.geolocation_lat.between(-34, 5.3) & geo.geolocation_lng.between(-74, -34.7)
    geo = geo[in_brazil]
    return geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_distance(df):
    cent = zip_centroids()
    c = cent.reindex(df["customer_zip"]).to_numpy()
    s = cent.reindex(df["seller_zip"]).to_numpy()
    df["distance_km"] = haversine_km(c[:, 0], c[:, 1], s[:, 0], s[:, 1])
    return df


def add_seller_history(df):
    """For each order, the seller's deliveries that had already ARRIVED before
    this purchase: count and smoothed late rate. Uses merge_asof on delivery
    time so nothing from the future leaks in."""
    events = (df[["seller_id", "delivered_ts", "is_late"]]
              .sort_values("delivered_ts").copy())
    events["cum_n"] = events.groupby("seller_id").cumcount() + 1
    events["cum_late"] = events.groupby("seller_id")["is_late"].cumsum()

    left = df[["order_id", "seller_id", "purchase_ts"]].sort_values("purchase_ts")
    hist = pd.merge_asof(left, events[["seller_id", "delivered_ts", "cum_n", "cum_late"]],
                         left_on="purchase_ts", right_on="delivered_ts",
                         by="seller_id", allow_exact_matches=False)
    hist[["cum_n", "cum_late"]] = hist[["cum_n", "cum_late"]].fillna(0)

    # Global prior uses only deliveries before the training cutoff would be
    # stricter; the global rate barely moves (8.1%) so a fixed prior is used.
    prior = 0.08
    hist["seller_hist_orders"] = hist["cum_n"]
    hist["seller_hist_late_rate"] = (hist["cum_late"] + PRIOR_N * prior) / (hist["cum_n"] + PRIOR_N)
    return df.merge(hist[["order_id", "seller_hist_orders", "seller_hist_late_rate"]], on="order_id")


def main():
    con = sqlite3.connect(DB)
    load_raw(con)
    df = pd.read_sql((ROOT / "sql" / "order_features.sql").read_text(), con,
                     parse_dates=["purchase_ts", "delivered_ts"])
    df = add_distance(df)
    df = add_seller_history(df)
    df = df.sort_values("purchase_ts").reset_index(drop=True)
    df.to_sql("order_features", con, if_exists="replace", index=False)
    con.execute("CREATE UNIQUE INDEX ix_order_features_id ON order_features(order_id)")
    con.close()
    print(f"{len(df):,} orders, late rate {df.is_late.mean():.3f}, "
          f"distance missing {df.distance_km.isna().mean():.3%}")


if __name__ == "__main__":
    main()
