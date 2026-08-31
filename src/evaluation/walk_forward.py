"""Walk-forward evaluation and pre-registered universe selection.

All selection rules in this module are fixed BEFORE any test-set metric is
computed, and every rule depends only on training-window data. This is the
main safeguard against the two failure modes the original project had:
random splits on temporal data, and choosing what to report after seeing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def expanding_window_folds(
    sessions: pd.DatetimeIndex, n_folds: int = 5, min_train_frac: float = 0.5
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Expanding-window folds over the calendar, shared by every model.

    Returns (train_end, test_start, test_end) per fold. Train is always
    everything strictly before test_start, so training data grows and test
    blocks are contiguous, disjoint and strictly forward in time.

        |--train--|test|
        |----train----|test|
        |-------train-------|test|
    """
    s = pd.DatetimeIndex(sorted(pd.unique(sessions)))
    start = int(len(s) * min_train_frac)
    edges = np.linspace(start, len(s), n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 5:
            continue
        folds.append((s[lo - 1], s[lo], s[hi - 1]))
    return folds


def purged_train_mask(
    sessions: pd.Series, train_end: pd.Timestamp, horizon: int = 1
) -> pd.Series:
    """Training rows, purged of observations whose target overlaps the test block.

    A row at session D has a target that resolves at D+horizon. If D is within
    `horizon` sessions of train_end, its label is realised inside the test
    window, so it is dropped. Without this, the last row of every training
    fold leaks the first test outcome.
    """
    uniq = pd.DatetimeIndex(sorted(pd.unique(sessions)))
    cutoff_idx = uniq.searchsorted(train_end) - horizon
    if cutoff_idx < 0:
        return pd.Series(False, index=sessions.index)
    return sessions <= uniq[cutoff_idx]


# --------------------------------------------------------------------------
# Universe selection (pre-registered, train-window only)
# --------------------------------------------------------------------------
def select_high_coverage(
    panel: pd.DataFrame, train_end: pd.Timestamp, min_rate: float = 0.75, min_tweets: int = 1000
) -> list[str]:
    """Tickers with enough social coverage for sentiment to be testable.

    Rationale, fixed in advance: the median ticker in this corpus sees about
    one tweet per session. Asking whether sentiment adds signal there is not a
    fair test of the hypothesis -- it is a test of whether one tweet predicts a
    return. Coverage is measured on TRAINING sessions only.
    """
    tr = panel[panel["session"] <= train_end]
    g = tr.groupby("ticker").agg(
        rate=("tweet_count", lambda s: s.notna().mean()),
        total=("tweet_count", "sum"),
    )
    keep = g[(g["rate"] >= min_rate) & (g["total"] >= min_tweets)]
    return sorted(keep.index)


def select_sector_stratified(
    panel: pd.DataFrame, sectors: pd.DataFrame, train_end: pd.Timestamp, n: int = 10
) -> pd.DataFrame:
    """Deterministic, sector-stratified ticker sample for the per-ticker models.

    Rule (fixed before results are seen):
      1. keep tickers with a full session history in the overlap window
      2. within each sector, rank by total training-window tweet volume
      3. take the top-ranked ticker from each sector, round-robin, until n

    Ranking uses tweet volume -- a property of the social data, never of
    returns or of model error -- so it cannot select for good performance.
    """
    tr = panel[panel["session"] <= train_end]
    full_len = tr.groupby("ticker")["session"].size().max()
    eligible = tr.groupby("ticker").agg(
        n_sessions=("session", "size"), total_tweets=("tweet_count", "sum")
    )
    eligible = eligible[eligible["n_sessions"] >= full_len]
    df = eligible.join(sectors.set_index("ticker")["sector"], how="inner").reset_index()
    df = df.sort_values(["sector", "total_tweets", "ticker"], ascending=[True, False, True])
    df["rank_in_sector"] = df.groupby("sector").cumcount()
    df = df.sort_values(["rank_in_sector", "total_tweets", "ticker"], ascending=[True, False, True])
    return df.head(n)[["ticker", "sector", "total_tweets"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    err = y_pred - y_true
    # Directional accuracy is scored only where the realised move is non-zero;
    # a flat day has no direction to get right.
    nz = y_true != 0
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "dir_acc": float(np.mean(np.sign(y_pred[nz]) == np.sign(y_true[nz]))) if nz.any() else np.nan,
    }


def paired_bootstrap(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    metric: str = "rmse", n_boot: int = 2000, seed: int = SEED,
) -> dict:
    """Paired bootstrap CI for metric(B) - metric(A) on identical observations.

    Paired because both arms predict the same rows from the same folds; the
    comparison is within-observation, which removes market-wide variance and
    is far more sensitive than comparing two independent score distributions.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    pred_a = np.asarray(pred_a, float)
    pred_b = np.asarray(pred_b, float)
    m = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    y_true, pred_a, pred_b = y_true[m], pred_a[m], pred_b[m]

    def score(y, p):
        if metric == "rmse":
            return np.sqrt(np.mean((p - y) ** 2))
        if metric == "mae":
            return np.mean(np.abs(p - y))
        if metric == "dir_acc":
            nz = y != 0
            return np.mean(np.sign(p[nz]) == np.sign(y[nz]))
        raise ValueError(metric)

    observed = score(y_true, pred_b) - score(y_true, pred_a)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = score(y_true[idx], pred_b[idx]) - score(y_true[idx], pred_a[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided bootstrap p-value for H0: no difference.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "delta": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "significant_at_5pct": bool(lo > 0 or hi < 0),
    }
