"""Leakage tests for feature construction, targets and splits.

The original project fed `Percent_Change_Bin` -- a binning of the target --
into the feature vector and reported R2 = 0.3555. These tests exist so that
class of error cannot recur silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.market import (  # noqa: E402
    TARGET_PREFIX,
    build_ticker_features,
    feature_columns,
)
from src.evaluation.walk_forward import (  # noqa: E402
    expanding_window_folds,
    purged_train_mask,
)


@pytest.fixture
def px() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    dates = pd.bdate_range("2014-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1e6, 5e6, n),
        }
    )


# --- targets --------------------------------------------------------------
def test_target_is_strictly_forward(px):
    f = build_ticker_features(px)
    close = px["Adj Close"].to_numpy()
    expected = close[1:] / close[:-1] - 1
    assert np.allclose(f[f"{TARGET_PREFIX}1"].to_numpy()[:-1], expected)


def test_last_row_target_is_nan(px):
    """No target can exist for the final session -- the future isn't observed."""
    f = build_ticker_features(px)
    assert np.isnan(f[f"{TARGET_PREFIX}1"].iloc[-1])
    assert f[f"{TARGET_PREFIX}3"].iloc[-3:].isna().all()


def test_target_excluded_from_features(px):
    f = build_ticker_features(px)
    f["ticker"] = "TEST"
    for include_sent in (True, False):
        cols = feature_columns(f, include_sentiment=include_sent)
        assert not any(c.startswith(TARGET_PREFIX) for c in cols)
        assert "close" not in cols


def test_binned_target_cannot_sneak_in(px):
    """A future-derived column, however named, must not become a feature."""
    f = build_ticker_features(px)
    f["ticker"] = "TEST"
    f[f"{TARGET_PREFIX}1_bin"] = (f[f"{TARGET_PREFIX}1"] > 0).astype(float)
    cols = feature_columns(f, include_sentiment=False)
    assert f"{TARGET_PREFIX}1_bin" not in cols


# --- causality of features ------------------------------------------------
def test_no_feature_correlates_perfectly_with_target(px):
    """Blanket guard: a feature that IS the target shows up as |r| ~ 1."""
    f = build_ticker_features(px)
    f["ticker"] = "TEST"
    y = f[f"{TARGET_PREFIX}1"]
    for c in feature_columns(f, include_sentiment=False):
        r = f[c].corr(y)
        assert not (np.isfinite(r) and abs(r) > 0.99), f"{c} is effectively the target"


def test_features_depend_only_on_past(px):
    """Perturbing the future must not change any feature at time t.

    Rewrites every price strictly after index t, rebuilds, and asserts the
    feature row at t is byte-identical. This catches centred rolling windows,
    negative shifts and lookahead joins in one shot.
    """
    t = 80
    f_full = build_ticker_features(px)
    px2 = px.copy()
    px2.loc[t + 1 :, ["Open", "High", "Low", "Close", "Adj Close"]] *= 1.5
    px2.loc[t + 1 :, "Volume"] *= 3
    f_pert = build_ticker_features(px2)

    cols = feature_columns(f_full.assign(ticker="T"), include_sentiment=False)
    a = f_full.loc[t, cols].astype(float).to_numpy()
    b = f_pert.loc[t, cols].astype(float).to_numpy()
    assert np.allclose(a, b, equal_nan=True), "a feature at t moved when the future changed"


def test_rolling_windows_are_trailing(px):
    f = build_ticker_features(px)
    ret = px["Adj Close"].pct_change()
    for w in (5, 10, 21):
        manual = ret.rolling(w).mean()
        assert np.allclose(
            f[f"ret_mean_{w}"].to_numpy(), manual.to_numpy(), equal_nan=True
        )


def test_lags_use_past_rows_only(px):
    f = build_ticker_features(px)
    ret = px["Adj Close"].pct_change()
    for k in (1, 2, 3, 5, 10):
        assert np.allclose(
            f[f"ret_lag_{k}"].to_numpy(), ret.shift(k).to_numpy(), equal_nan=True
        )


def test_no_infinities_in_features():
    """Zero-volume / flat-range sessions must yield NaN, never inf."""
    rng = np.random.default_rng(1)
    n = 60
    close = np.full(n, 100.0)          # perfectly flat -> zero trailing range
    vol = np.concatenate([np.zeros(10), rng.integers(1e6, 2e6, n - 10)])
    df = pd.DataFrame({
        "Date": pd.bdate_range("2014-01-02", periods=n),
        "Open": close, "High": close, "Low": close,
        "Close": close, "Adj Close": close, "Volume": vol,
    })
    f = build_ticker_features(df)
    cols = feature_columns(f.assign(ticker="T"), include_sentiment=False)
    assert not np.isinf(f[cols].to_numpy(dtype=float)).any()


# --- splits ---------------------------------------------------------------
def test_folds_are_forward_and_disjoint():
    sessions = pd.bdate_range("2014-01-02", periods=500)
    folds = expanding_window_folds(sessions, n_folds=5)
    assert len(folds) == 5
    prev_end = None
    for train_end, test_start, test_end in folds:
        assert train_end < test_start <= test_end
        if prev_end is not None:
            assert test_start > prev_end
        prev_end = test_end


def test_purging_removes_boundary_rows():
    sessions = pd.bdate_range("2014-01-02", periods=100)
    s = pd.Series(sessions)
    train_end = sessions[59]
    mask = purged_train_mask(s, train_end, horizon=1)
    # The row at train_end has a target realised at train_end+1, which is the
    # first test session -- it must be purged.
    assert not mask.iloc[59]
    assert mask.iloc[58]


def test_test_rows_never_in_train():
    sessions = pd.bdate_range("2014-01-02", periods=300)
    s = pd.Series(sessions)
    for train_end, test_start, test_end in expanding_window_folds(sessions, n_folds=4):
        train = s[purged_train_mask(s, train_end, horizon=1)]
        test = s[(s >= test_start) & (s <= test_end)]
        assert set(train).isdisjoint(set(test))
        assert train.max() < test.min()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
