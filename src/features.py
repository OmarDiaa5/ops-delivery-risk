"""Feature list shared by training, the API and the monitor."""
NUMERIC = [
    "promised_days", "distance_km", "same_state", "n_items", "n_sellers",
    "total_price", "total_freight", "freight_ratio", "total_weight_g",
    "total_volume_cm3", "installments", "purchase_dow", "purchase_hour",
    "purchase_month", "seller_hist_orders", "seller_hist_late_rate",
]
CATEGORICAL = ["customer_state", "seller_state", "main_category", "payment_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "is_late"
