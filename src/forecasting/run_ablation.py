"""Core experiment: does sentiment add signal beyond market history?

Two arms, identical in every respect except the feature set:

    A. market-only        lagged returns, trailing vol/momentum, volume
    B. market + sentiment  the same features, plus daily sentiment aggregates

Both arms share the same walk-forward folds, the same purging, the same
hyperparameter grid and the same inner-validation selection procedure. The
only difference is the columns. That is what makes the delta interpretable.

A naive persistence baseline (predict tomorrow's return = today's return)
and a zero baseline (predict 0, the unconditional mean of returns) are
included as reference points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.walk_forward import (  # noqa: E402
    expanding_window_folds,
    paired_bootstrap,
    purged_train_mask,
    regression_metrics,
    select_high_coverage,
)
from src.features.market import TARGET_PREFIX, feature_columns  # noqa: E402

SEED = 42
N_FOLDS = 5

# Fixed grid, identical for both arms. Deliberately small: the point of the
# experiment is the feature-set contrast, not squeezing out the best model.
GRID = [
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.8,
     "colsample_bytree": 0.8, "min_child_weight": 20, "reg_lambda": 2.0},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 400, "subsample": 0.8,
     "colsample_bytree": 0.6, "min_child_weight": 50, "reg_lambda": 5.0},
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.9,
     "colsample_bytree": 0.8, "min_child_weight": 10, "reg_lambda": 1.0},
]


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: str
) -> np.ndarray:
    """Fit XGBoost with inner-validation hyperparameter selection, then predict.

    The inner validation block is the last 20% of TRAINING sessions. The test
    block is never touched during selection.
    """
    sess = pd.DatetimeIndex(sorted(train["session"].unique()))
    cut = sess[int(len(sess) * 0.8)]
    inner_tr = train[train["session"] < cut]
    inner_va = train[train["session"] >= cut]

    best, best_rmse = None, np.inf
    for params in GRID:
        m = xgb.XGBRegressor(
            **params, random_state=SEED, n_jobs=4, tree_method="hist",
            objective="reg:squarederror",
        )
        m.fit(inner_tr[cols], inner_tr[target], verbose=False)
        rmse = float(np.sqrt(np.mean((m.predict(inner_va[cols]) - inner_va[target]) ** 2)))
        if rmse < best_rmse:
            best, best_rmse = params, rmse

    model = xgb.XGBRegressor(
        **best, random_state=SEED, n_jobs=4, tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(train[cols], train[target], verbose=False)
    return model.predict(test[cols])


def run(panel: pd.DataFrame, target: str, label: str, horizon: int = 1) -> tuple[pd.DataFrame, dict]:
    panel = panel.dropna(subset=[target]).sort_values(["session", "ticker"]).reset_index(drop=True)
    folds = expanding_window_folds(pd.DatetimeIndex(panel["session"].unique()), N_FOLDS)

    cols_mkt = feature_columns(panel, include_sentiment=False)
    cols_sent = feature_columns(panel, include_sentiment=True)
    assert set(cols_mkt).issubset(set(cols_sent))
    assert not any(c.startswith(TARGET_PREFIX) for c in cols_sent)

    records = []
    for i, (train_end, test_start, test_end) in enumerate(folds, 1):
        tr = panel[purged_train_mask(panel["session"], train_end, horizon)]
        te = panel[(panel["session"] >= test_start) & (panel["session"] <= test_end)]
        if len(tr) < 500 or len(te) < 50:
            continue
        rec = te[["ticker", "session", target]].copy()
        rec["fold"] = i
        rec["pred_zero"] = 0.0
        rec["pred_naive"] = te["ret_1"].fillna(0.0).to_numpy()
        rec["pred_market"] = fit_predict(tr, te, cols_mkt, target)
        rec["pred_sentiment"] = fit_predict(tr, te, cols_sent, target)
        records.append(rec)
        print(f"    fold {i}: train={len(tr):>6,} test={len(te):>5,} "
              f"[{test_start.date()} -> {test_end.date()}]")

    preds = pd.concat(records, ignore_index=True)
    y = preds[target].to_numpy()

    rows = []
    for name, col in [
        ("Zero baseline", "pred_zero"),
        ("Naive persistence", "pred_naive"),
        ("XGBoost (market-only)", "pred_market"),
        ("XGBoost (market + sentiment)", "pred_sentiment"),
    ]:
        m = regression_metrics(y, preds[col].to_numpy())
        m["model"] = name
        m["universe"] = label
        m["target"] = target
        rows.append(m)
    table = pd.DataFrame(rows)[["universe", "target", "model", "n", "rmse", "mae", "dir_acc"]]

    tests = {}
    for metric in ("rmse", "mae", "dir_acc"):
        tests[metric] = paired_bootstrap(
            y, preds["pred_market"].to_numpy(), preds["pred_sentiment"].to_numpy(), metric=metric
        )
    return table, {"preds": preds, "tests": tests}


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    panel["session"] = pd.to_datetime(panel["session"])

    sessions = pd.DatetimeIndex(sorted(panel["session"].unique()))
    first_train_end = expanding_window_folds(sessions, N_FOLDS)[0][0]
    high_cov = select_high_coverage(panel, first_train_end)
    print(f"high-coverage universe ({len(high_cov)} tickers, rule fixed pre-hoc): "
          f"{', '.join(high_cov)}\n")

    all_tables, all_tests, all_preds = [], {}, {}
    for label, sub in [
        ("all tickers (n=%d)" % panel.ticker.nunique(), panel),
        ("high-coverage (n=%d)" % len(high_cov), panel[panel.ticker.isin(high_cov)]),
    ]:
        for h in (1, 3):
            target = f"{TARGET_PREFIX}{h}"
            print(f"  {label} | horizon={h}d")
            tbl, extra = run(sub, target, label, horizon=h)
            all_tables.append(tbl)
            all_tests[f"{label}|h{h}"] = extra["tests"]
            all_preds[f"{label}|h{h}"] = extra["preds"]
            print(tbl.to_string(index=False), "\n")

    comparison = pd.concat(all_tables, ignore_index=True)
    comparison.to_csv(ROOT / "results/forecasting/model_comparison.csv", index=False)

    abl_rows = []
    for key, t in all_tests.items():
        universe, h = key.split("|")
        for metric, r in t.items():
            abl_rows.append({"universe": universe, "horizon": h, "metric": metric, **r})
    pd.DataFrame(abl_rows).to_csv(ROOT / "results/forecasting/ablation.csv", index=False)

    for key, p in all_preds.items():
        safe = key.replace(" ", "_").replace("|", "_").replace("(", "").replace(")", "").replace("=", "")
        p.to_parquet(ROOT / f"data/processed/preds_{safe}.parquet", index=False)

    (ROOT / "results/forecasting/high_coverage_universe.json").write_text(
        json.dumps(
            {
                "rule": "train-window coverage_rate >= 0.75 AND total training tweets >= 1000",
                "measured_on": f"sessions <= {first_train_end.date()}",
                "n_tickers": len(high_cov),
                "tickers": high_cov,
            },
            indent=2,
        )
    )
    print("=== ablation: market+sentiment MINUS market-only (negative = sentiment helps) ===")
    print(pd.DataFrame(abl_rows).to_string(index=False))


if __name__ == "__main__":
    main()
