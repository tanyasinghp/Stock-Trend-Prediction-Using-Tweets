"""Market feature engineering.

Every feature here is *causal*: computed from information available at or
before session D's close. Every rolling window is trailing (pandas .rolling
is trailing by default; .shift() is applied where a same-bar value would
otherwise sneak in). The target is the only forward-looking column and is
explicitly named so it can never be mistaken for a feature.

Target
------
    ret_fwd_1 = close(D+1) / close(D) - 1        (the thing being predicted)

Feature naming contract
-----------------------
    ret_fwd_*   -> TARGET ONLY. Guarded against in build_panel().
    everything else -> safe to use as a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_PREFIX = "ret_fwd_"
LAGS = (1, 2, 3, 5, 10)
WINDOWS = (5, 10, 21)


def build_ticker_features(px: pd.DataFrame, horizons=(1, 3)) -> pd.DataFrame:
    """Build causal market features + forward targets for one ticker.

    Parameters
    ----------
    px : columns [Date, Open, High, Low, Close, Adj Close, Volume], one ticker.
    """
    df = px.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Adj Close"].astype(float)
    vol = df["Volume"].astype(float)

    out = pd.DataFrame({"session": df["Date"]})

    # --- realised return up to and including today's close (known at D) ----
    ret = close.pct_change()
    out["ret_1"] = ret

    # --- lagged returns: strictly past observations -----------------------
    for k in LAGS:
        out[f"ret_lag_{k}"] = ret.shift(k)

    # --- trailing momentum / volatility -----------------------------------
    for w in WINDOWS:
        out[f"ret_mean_{w}"] = ret.rolling(w).mean()
        out[f"vol_{w}"] = ret.rolling(w).std()
        out[f"mom_{w}"] = close / close.shift(w) - 1.0

    # --- price position within its own trailing range (ticker-relative) ---
    for w in (10, 21):
        lo = close.rolling(w).min()
        hi = close.rolling(w).max()
        out[f"px_pos_{w}"] = (close - lo) / (hi - lo).replace(0.0, np.nan)

    # --- volume, expressed relative to its own trailing average -----------
    out["vol_chg"] = vol.pct_change()
    for w in (5, 21):
        out[f"vol_rel_{w}"] = vol / vol.rolling(w).mean()

    # --- intraday range, a cheap volatility proxy -------------------------
    out["hl_range"] = (df["High"] - df["Low"]) / close.replace(0.0, np.nan)
    out["hl_range_ma5"] = out["hl_range"].rolling(5).mean()

    # --- FORWARD TARGETS (never features) ---------------------------------
    for h in horizons:
        out[f"{TARGET_PREFIX}{h}"] = close.shift(-h) / close - 1.0

    out["close"] = close  # kept for price-level models (ARIMA/Prophet)

    # Zero-volume sessions and flat trailing ranges produce +/-inf from the
    # ratio features above. Convert to NaN rather than clipping: XGBoost
    # treats NaN as "unknown" and learns a default split direction, whereas a
    # clipped sentinel would be a fabricated numeric value.
    feat_cols = [c for c in out.columns if c not in ("session",)]
    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(df: pd.DataFrame, include_sentiment: bool) -> list[str]:
    """Return the modelling feature list, with targets structurally excluded."""
    sentiment_cols = {
        "sent_mean", "sent_median", "sent_std", "pos_ratio",
        "tweet_count", "log_tweet_count", "sent_mean_ma5", "sent_chg", "tweet_count_rel",
    }
    drop = {"session", "ticker", "close"}
    cols = []
    for c in df.columns:
        if c in drop or c.startswith(TARGET_PREFIX):
            continue
        if c in sentiment_cols and not include_sentiment:
            continue
        if df[c].dtype.kind not in "fiu":
            continue
        cols.append(c)
    return cols


def add_sentiment_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing sentiment transforms, computed per ticker (causal)."""
    df = df.sort_values(["ticker", "session"]).copy()
    g = df.groupby("ticker", observed=True)
    df["sent_mean_ma5"] = g["sent_mean"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    df["sent_chg"] = g["sent_mean"].transform(lambda s: s.diff())
    df["tweet_count_rel"] = df["tweet_count"] / g["tweet_count"].transform(
        lambda s: s.rolling(21, min_periods=5).mean()
    )
    return df
