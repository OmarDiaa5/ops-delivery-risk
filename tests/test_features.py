import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_features import add_seller_history  # noqa: E402
from monitor import psi, status  # noqa: E402


def test_seller_history_only_counts_deliveries_before_purchase():
    # Seller S: order a delivered late on Jan 10; order b bought Jan 5 (before
    # a arrived) must not see it; order c bought Jan 15 must.
    df = pd.DataFrame({
        "order_id": ["a", "b", "c"], "seller_id": ["S"] * 3,
        "purchase_ts": pd.to_datetime(["2018-01-01", "2018-01-05", "2018-01-15"]),
        "delivered_ts": pd.to_datetime(["2018-01-10", "2018-01-20", "2018-01-25"]),
        "is_late": [1, 0, 0],
    })
    out = add_seller_history(df).set_index("order_id")
    assert out.loc["b", "seller_hist_orders"] == 0
    assert out.loc["c", "seller_hist_orders"] == 1
    assert out.loc["c", "seller_hist_late_rate"] > out.loc["b", "seller_hist_late_rate"]


def test_psi_zero_for_identical_and_large_for_shift():
    ref = np.full(10, 0.1)
    assert psi(ref, ref) < 1e-9
    shifted = np.r_[np.full(5, 0.0), np.full(5, 0.2)]
    assert status(psi(ref, shifted)) == "drift"
