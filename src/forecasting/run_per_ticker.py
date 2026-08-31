"""Per-ticker classical forecasters: ARIMA and Prophet.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
ARIMA and Prophet are univariate, per-series models. XGBoost here is a pooled
panel model that shares one parameter set across 87 tickers. They are not the
same kind of estimator and are not fitted on the same amount of data, so the
comparison is reported side by side with that caveat stated, not merged into
a single leaderboard that implies a controlled contest.

TICKER SELECTION
----------------
Sector-stratified and deterministic, via select_sector_stratified(). Ranking
is by training-window tweet volume -- a property of the social data, never of
returns or model error -- so the sample cannot be selected for good results.

TARGETS
-------
ARIMA is fitted on returns (stationary; ADF reported). Prophet is fitted on
log price, its natural domain, and its forecast is converted to an implied
return so both land in return space alongside XGBoost.

MAPE is computed ONLY on the price-level forecast, where the denominator is
a positive price. It is never computed on returns, whose denominator is a
near-zero quantity that makes the statistic explode.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.walk_forward import (  # noqa: E402
    expanding_window_folds,
    regression_metrics,
    select_sector_stratified,
)

STOCKNET = ROOT / "data/external/stocknet-dataset"
REFIT_EVERY = 21          # sessions; monthly refit, forecasts are still 1-step
ARIMA_GRID = [(p, 0, q) for p in range(3) for q in range(3)]
N_TICKERS = 10


def load_sectors() -> pd.DataFrame:
    t = pd.read_csv(STOCKNET / "StockTable", sep="\t")
    t["ticker"] = t["Symbol"].str.lstrip("$").str.upper()
    t["sector"] = t["Sector"].str.strip()
    return t[["ticker", "sector"]]


def pick_arima_order(y: np.ndarray) -> tuple[int, int, int]:
    """Lowest-AIC order from a fixed grid. Selected on training data only."""
    best, best_aic = (1, 0, 0), np.inf
    for order in ARIMA_GRID:
        try:
            aic = ARIMA(y, order=order).fit().aic
            if np.isfinite(aic) and aic < best_aic:
                best, best_aic = order, aic
        except Exception:
            continue
    return best


def run_arima(rets: pd.Series, test_idx: np.ndarray) -> np.ndarray:
    preds = np.full(len(test_idx), np.nan)
    order, model = None, None
    for j, i in enumerate(test_idx):
        if j % REFIT_EVERY == 0:
            hist = rets.iloc[:i].dropna().to_numpy()
            if len(hist) < 60:
                continue
            if order is None:
                order = pick_arima_order(hist[-500:])
            try:
                model = ARIMA(hist, order=order).fit()
            except Exception:
                model = None
        if model is None:
            continue
        try:
            # Append observations seen since the last refit, without refitting,
            # so each forecast uses all information up to that session.
            since = rets.iloc[i - (j % REFIT_EVERY) : i].dropna().to_numpy() if j % REFIT_EVERY else np.array([])
            m = model.append(since, refit=False) if len(since) else model
            preds[j] = float(m.forecast(1)[0])
        except Exception:
            preds[j] = float(model.forecast(1)[0])
    return preds


def run_prophet(dates: pd.Series, close: pd.Series, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from prophet import Prophet

    ret_pred = np.full(len(test_idx), np.nan)
    px_pred = np.full(len(test_idx), np.nan)
    fitted = None
    for j, i in enumerate(test_idx):
        if j % REFIT_EVERY == 0:
            hist = pd.DataFrame({"ds": dates.iloc[:i], "y": np.log(close.iloc[:i])}).dropna()
            if len(hist) < 60:
                continue
            try:
                fitted = Prophet(
                    daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, changepoint_prior_scale=0.05,
                ).fit(hist)
            except Exception:
                fitted = None
        if fitted is None:
            continue
        try:
            fc = fitted.predict(pd.DataFrame({"ds": [dates.iloc[i + 1]]}))
            log_hat = float(fc["yhat"].iloc[0])
            px_pred[j] = np.exp(log_hat)
            ret_pred[j] = np.exp(log_hat - np.log(close.iloc[i])) - 1.0
        except Exception:
            continue
    return ret_pred, px_pred


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    panel["session"] = pd.to_datetime(panel["session"])
    sessions = pd.DatetimeIndex(sorted(panel["session"].unique()))
    folds = expanding_window_folds(sessions, 5)
    first_train_end, test_start = folds[0][0], folds[0][1]
    test_end = folds[-1][2]

    picks = select_sector_stratified(panel, load_sectors(), first_train_end, N_TICKERS)
    picks.to_csv(ROOT / "results/forecasting/per_ticker_universe.csv", index=False)
    print("sector-stratified sample (rule fixed pre-hoc):")
    print(picks.to_string(index=False), "\n")

    rows, adf_rows = [], []
    for tk in picks["ticker"]:
        d = panel[panel.ticker == tk].sort_values("session").reset_index(drop=True)
        test_idx = d.index[(d.session >= test_start) & (d.session <= test_end)].to_numpy()
        test_idx = test_idx[test_idx < len(d) - 1]
        if len(test_idx) < 50:
            continue

        rets, close, dates = d["ret_1"], d["close"], d["session"]
        y_true = d.loc[test_idx, "ret_fwd_1"].to_numpy()
        px_true = close.iloc[test_idx + 1].to_numpy()

        train_ret = rets[d.session <= first_train_end].dropna()
        adf_p = float(adfuller(train_ret.to_numpy(), autolag="AIC")[1])
        adf_px = float(adfuller(np.log(close[d.session <= first_train_end]).to_numpy(), autolag="AIC")[1])
        adf_rows.append({"ticker": tk, "adf_p_returns": adf_p, "adf_p_log_price": adf_px})

        p_arima = run_arima(rets, test_idx)
        p_prophet_ret, p_prophet_px = run_prophet(dates, close, test_idx)
        p_naive = rets.iloc[test_idx].fillna(0.0).to_numpy()
        p_rw_px = close.iloc[test_idx].to_numpy()   # random walk on price

        for name, pred in [
            ("Naive persistence", p_naive),
            ("Zero baseline", np.zeros(len(test_idx))),
            ("ARIMA (per-ticker)", p_arima),
            ("Prophet (per-ticker)", p_prophet_ret),
        ]:
            m = regression_metrics(y_true, pred)
            m.update(ticker=tk, model=name, space="return")
            rows.append(m)

        # Price-level task: MAPE is meaningful here (positive denominator).
        for name, pred in [
            ("Random walk (price)", p_rw_px),
            ("Prophet (price)", p_prophet_px),
        ]:
            mask = np.isfinite(pred) & np.isfinite(px_true)
            if mask.sum() < 20:
                continue
            rows.append({
                "ticker": tk, "model": name, "space": "price", "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((pred[mask] - px_true[mask]) ** 2))),
                "mae": float(np.mean(np.abs(pred[mask] - px_true[mask]))),
                "mape": float(np.mean(np.abs((pred[mask] - px_true[mask]) / px_true[mask]))),
                "dir_acc": np.nan,
            })
        print(f"  {tk:6s} done  (arima order search + {len(test_idx)} rolling forecasts)")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/forecasting/per_ticker_results.csv", index=False)
    pd.DataFrame(adf_rows).to_csv(ROOT / "results/forecasting/stationarity_tests.csv", index=False)

    print("\n=== return space, averaged across tickers ===")
    agg = (res[res.space == "return"]
           .groupby("model")[["rmse", "mae", "dir_acc"]].mean()
           .sort_values("rmse"))
    print(agg.to_string())
    print("\n=== price space (MAPE meaningful here) ===")
    print(res[res.space == "price"].groupby("model")[["rmse", "mae", "mape"]].mean().to_string())
    print("\n=== ADF p-values (train window) ===")
    a = pd.DataFrame(adf_rows)
    print(f"returns   : median p={a.adf_p_returns.median():.2e}  "
          f"stationary at 5% for {(a.adf_p_returns < 0.05).sum()}/{len(a)} tickers")
    print(f"log price : median p={a.adf_p_log_price.median():.3f}  "
          f"stationary at 5% for {(a.adf_p_log_price < 0.05).sum()}/{len(a)} tickers")

    (ROOT / "results/forecasting/per_ticker_config.json").write_text(json.dumps({
        "n_tickers": int(picks.shape[0]),
        "selection_rule": "sector-stratified; ranked by TRAIN-window tweet volume; deterministic",
        "arima_grid": [list(o) for o in ARIMA_GRID],
        "arima_order_criterion": "lowest AIC on training returns",
        "refit_every_n_sessions": REFIT_EVERY,
        "forecast_horizon": 1,
        "test_window": [str(test_start.date()), str(test_end.date())],
    }, indent=2))


if __name__ == "__main__":
    main()
