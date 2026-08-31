"""Robustness: does the sentiment effect depend on regime or coverage?

Secondary to the main ablation. Conditions the market-only vs market+sentiment
comparison on:

  * trailing realised volatility (terciles, computed from vol_21 which is a
    trailing feature -- so the conditioning variable is itself causal)
  * per-ticker tweet volume (terciles over the training window)
  * forecast horizon (1d vs 3d, already produced by the main ablation)

Subgroups are cut on variables known BEFORE the prediction, never on outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.walk_forward import paired_bootstrap, regression_metrics  # noqa: E402


def load(key: str) -> pd.DataFrame:
    safe = key.replace(" ", "_").replace("|", "_").replace("(", "").replace(")", "").replace("=", "")
    return pd.read_parquet(ROOT / f"data/processed/preds_{safe}.parquet")


def subgroup_table(preds: pd.DataFrame, target: str, group_col: str, label: str) -> pd.DataFrame:
    rows = []
    for name, g in preds.groupby(group_col, observed=True):
        y = g[target].to_numpy()
        mkt = regression_metrics(y, g["pred_market"].to_numpy())
        sen = regression_metrics(y, g["pred_sentiment"].to_numpy())
        bs = paired_bootstrap(y, g["pred_market"].to_numpy(), g["pred_sentiment"].to_numpy(), "rmse")
        bs_dir = paired_bootstrap(
            y, g["pred_market"].to_numpy(), g["pred_sentiment"].to_numpy(), "dir_acc"
        )
        rows.append({
            "dimension": label,
            "subgroup": str(name),
            "n": mkt["n"],
            "rmse_market": mkt["rmse"],
            "rmse_sentiment": sen["rmse"],
            "rmse_delta": bs["delta"],
            "rmse_rel_pct": 100.0 * bs["delta"] / mkt["rmse"],
            "rmse_p": bs["p_value"],
            "rmse_sig": bs["significant_at_5pct"],
            "diracc_market": mkt["dir_acc"],
            "diracc_sentiment": sen["dir_acc"],
            "diracc_delta": bs_dir["delta"],
            "diracc_p": bs_dir["p_value"],
            "diracc_sig": bs_dir["significant_at_5pct"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    panel["session"] = pd.to_datetime(panel["session"])

    # Ticker tweet-volume terciles, computed on the training window only.
    train_end = panel["session"].quantile(0.5)
    tv = (panel[panel.session <= train_end].groupby("ticker")["tweet_count"]
          .sum().rename("train_tweets").reset_index())
    tv["tweet_tier"] = pd.qcut(
        tv["train_tweets"].rank(method="first"), 3, labels=["low", "mid", "high"]
    )

    out = []
    for universe in ("all tickers (n=87)", "high-coverage (n=11)"):
        for h in (1, 3):
            key = f"{universe}|h{h}"
            target = f"ret_fwd_{h}"
            try:
                preds = load(key)
            except FileNotFoundError:
                continue
            preds["session"] = pd.to_datetime(preds["session"])
            preds = preds.merge(
                panel[["ticker", "session", "vol_21"]], on=["ticker", "session"], how="left"
            ).merge(tv[["ticker", "tweet_tier"]], on="ticker", how="left")

            preds = preds.dropna(subset=["vol_21"])
            preds["vol_regime"] = pd.qcut(preds["vol_21"], 3, labels=["low", "mid", "high"])

            for col, lbl in [("vol_regime", "trailing volatility"), ("tweet_tier", "ticker tweet volume")]:
                t = subgroup_table(preds, target, col, lbl)
                t.insert(0, "horizon", f"{h}d")
                t.insert(0, "universe", universe)
                out.append(t)

    res = pd.concat(out, ignore_index=True)
    res.to_csv(ROOT / "results/forecasting/robustness.csv", index=False)

    show = ["universe", "horizon", "dimension", "subgroup", "n",
            "rmse_market", "rmse_sentiment", "rmse_rel_pct", "rmse_p", "rmse_sig"]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    n_sig = int(res["rmse_sig"].sum())
    n_help = int(((res["rmse_delta"] < 0) & res["rmse_sig"]).sum())
    print(f"\nsubgroups tested                    : {len(res)}")
    print(f"significant RMSE differences at 5%  : {n_sig}")
    print(f"  of which sentiment HELPS          : {n_help}")
    print(f"  of which sentiment HURTS          : {n_sig - n_help}")
    print(f"expected false positives at alpha=5%: {0.05 * len(res):.1f}")


if __name__ == "__main__":
    main()
