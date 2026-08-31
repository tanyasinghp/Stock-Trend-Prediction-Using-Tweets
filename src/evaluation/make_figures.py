"""Generate all figures from result files.

Every figure reads from results/ or data/processed/ -- nothing is recomputed
here, so figures cannot drift away from the reported numbers.

Writes: results/figures/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_MKT, C_SENT, C_BASE = "#2b6cb0", "#c05621", "#4a5568"


def fig_coverage() -> None:
    """Social coverage is extremely skewed -- this motivates the subset rule."""
    c = pd.read_csv(ROOT / "results/forecasting/coverage_report.csv")
    c = c.sort_values("total_tweets", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(range(len(c)), c["total_tweets"], color=C_MKT, width=0.9)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("ticker (ranked)")
    ax[0].set_ylabel("total tweets (log)")
    ax[0].set_title(f"Tweet volume by ticker (n={len(c)})")
    for i, tk in enumerate(c["ticker"].head(3)):
        ax[0].annotate(tk, (i, c["total_tweets"].iloc[i]), fontsize=7,
                       ha="left", va="bottom")
    ax[0].axhline(1000, ls="--", lw=1, color=C_SENT)
    ax[0].annotate("1,000-tweet threshold", (len(c) * 0.45, 1200),
                   color=C_SENT, fontsize=7)

    ax[1].hist(c["coverage_rate"], bins=20, color=C_MKT, edgecolor="white")
    ax[1].axvline(0.75, ls="--", lw=1, color=C_SENT)
    ax[1].annotate("0.75 threshold", (0.76, ax[1].get_ylim()[1] * 0.8),
                   color=C_SENT, fontsize=7)
    ax[1].set_xlabel("fraction of sessions with any tweet")
    ax[1].set_ylabel("tickers")
    ax[1].set_title("Session coverage rate")
    fig.suptitle("Social coverage is heavily skewed", y=1.03, fontsize=11)
    fig.savefig(FIG / "01_coverage.png")
    plt.close(fig)


def fig_sentiment_models() -> None:
    s = pd.read_csv(ROOT / "results/sentiment/model_comparison.csv")
    s = s.sort_values("test_roc_auc")
    sel = "TF-IDF + LogisticRegression"
    colors = [C_SENT if m == sel else C_MKT for m in s["model"]]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(s["model"], s["test_roc_auc"], color=colors)
    for y, v in enumerate(s["test_roc_auc"]):
        ax.annotate(f"{v:.3f}", (v, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_xlim(0.5, 0.95)
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set_xlabel("test ROC-AUC")
    ax.set_title("Sentiment classifier selection (orange = selected by 1-SE rule)")
    fig.savefig(FIG / "02_sentiment_models.png")
    plt.close(fig)


def fig_model_comparison() -> None:
    m = pd.read_csv(ROOT / "results/forecasting/model_comparison.csv")
    m = m[(m.universe.str.startswith("all")) & (m.target == "ret_fwd_1")]
    order = ["Zero baseline", "Naive persistence",
             "XGBoost (market-only)", "XGBoost (market + sentiment)"]
    m = m.set_index("model").loc[order].reset_index()
    colors = [C_BASE, C_BASE, C_MKT, C_SENT]
    short = ["Zero", "Naive", "XGB\nmarket", "XGB\nmarket+sent"]

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].bar(short, m["rmse"], color=colors)
    ax[0].axhline(m["rmse"].iloc[0], ls="--", lw=1, color=C_BASE)
    ax[0].annotate("zero baseline", (0.5, m["rmse"].iloc[0] * 0.90),
                   fontsize=7, color="white", ha="center")
    ax[0].set_ylabel("RMSE (returns)")
    ax[0].set_title("Nothing beats predicting zero")
    for i, v in enumerate(m["rmse"]):
        ax[0].annotate(f"{v:.5f}", (i, v), ha="center",
                       xytext=(0, 3), textcoords="offset points", fontsize=7)

    d = m[m.dir_acc > 0]
    ax[1].bar(short[1:], d["dir_acc"], color=colors[1:])
    ax[1].axhline(0.5, ls="--", lw=1.2, color="k")
    ax[1].annotate("coin flip", (1.6, 0.502), fontsize=7)
    ax[1].set_ylim(0.45, 0.55)
    ax[1].set_ylabel("directional accuracy")
    ax[1].set_title("Direction is a coin flip")
    for i, v in enumerate(d["dir_acc"]):
        ax[1].annotate(f"{v:.3f}", (i, v), ha="center",
                       xytext=(0, 3), textcoords="offset points", fontsize=7)
    fig.suptitle("1-day forward returns, 87 tickers, n=24,621", y=1.04, fontsize=11)
    fig.savefig(FIG / "03_model_comparison.png")
    plt.close(fig)


def fig_ablation() -> None:
    """Effect sizes with CIs, shown against a plausible-effect reference."""
    a = pd.read_csv(ROOT / "results/forecasting/ablation.csv")
    a = a[a.metric == "rmse"].reset_index(drop=True)
    base = pd.read_csv(ROOT / "results/forecasting/model_comparison.csv")
    base = base[base.model == "XGBoost (market-only)"]

    rel, labels = [], []
    for _, r in a.iterrows():
        h = int(r["horizon"].lstrip("h"))
        b = base[(base.universe == r["universe"]) &
                 (base.target == f"ret_fwd_{h}")]["rmse"].iloc[0]
        rel.append((100 * r["delta"] / b, 100 * r["ci_low"] / b, 100 * r["ci_high"] / b))
        labels.append(f"{r['universe'].split(' (')[0]}\n{r['horizon'][1:]}d")  # plain str.split

    rel = np.array(rel)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(rel))
    ax.errorbar(rel[:, 0], y, xerr=[rel[:, 0] - rel[:, 1], rel[:, 2] - rel[:, 0]],
                fmt="o", color=C_SENT, capsize=3, lw=1.4, markersize=5)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("relative RMSE change from adding sentiment (%)  |  negative = helps")
    ax.set_xlim(-1.2, 1.2)
    ax.axvspan(-0.25, 0.25, color="grey", alpha=0.12)
    ax.annotate("effects inside this band are\nfar below anything tradeable",
                (0.27, 0.35), fontsize=7, color=C_BASE)
    ax.set_title("Sentiment ablation: significant, but negligible")
    fig.savefig(FIG / "04_ablation.png")
    plt.close(fig)


def fig_robustness() -> None:
    r = pd.read_csv(ROOT / "results/forecasting/robustness.csv")
    v = r[r.dimension == "trailing volatility"].copy()
    v["grp"] = v["universe"].str.split(" (", regex=False).str[0] + " " + v["horizon"]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    order = ["low", "mid", "high"]
    width = 0.2
    for i, (name, g) in enumerate(v.groupby("grp")):
        g = g.set_index("subgroup").reindex(order)
        pos = np.arange(3) + (i - 1.5) * width
        bars = ax.bar(pos, g["rmse_rel_pct"], width, label=name)
        for b, sig in zip(bars, g["rmse_sig"]):
            if sig:
                ax.annotate("*", (b.get_x() + b.get_width() / 2,
                                  b.get_height() - 0.06),
                            ha="center", fontsize=12, color="k")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low vol", "mid vol", "high vol"])
    ax.set_ylabel("relative RMSE change (%)")
    ax.set_title("Sentiment's effect concentrates in high volatility  (* p<0.05)")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "05_robustness.png")
    plt.close(fig)


def fig_per_ticker() -> None:
    p = pd.read_csv(ROOT / "results/forecasting/per_ticker_results.csv")
    ret = p[p.space == "return"]
    piv = ret.pivot(index="ticker", columns="model", values="rmse")
    piv["delta"] = 100 * (piv["ARIMA (per-ticker)"] / piv["Zero baseline"] - 1)
    piv = piv.sort_values("delta")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = [C_MKT if d < 0 else C_SENT for d in piv["delta"]]
    ax[0].barh(piv.index, piv["delta"], color=colors)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("ARIMA RMSE vs zero baseline (%)")
    ax[0].set_title(f"ARIMA wins on {int((piv['delta'] < 0).sum())}/{len(piv)} tickers")

    agg = ret.groupby("model")["rmse"].mean().sort_values()
    ax[1].bar(range(len(agg)), agg.to_numpy(),
              color=[C_BASE, C_MKT, C_BASE, C_SENT])
    ax[1].set_xticks(range(len(agg)))
    ax[1].set_xticklabels([m.replace(" (per-ticker)", "").replace(" ", "\n")
                           for m in agg.index], fontsize=7)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("mean RMSE (log)")
    ax[1].set_title("Prophet fails badly on returns")
    for i, val in enumerate(agg.to_numpy()):
        ax[1].annotate(f"{val:.4f}", (i, val), ha="center",
                       xytext=(0, 3), textcoords="offset points", fontsize=7)
    fig.savefig(FIG / "06_per_ticker.png")
    plt.close(fig)


def fig_series() -> None:
    """One ticker: price, tweet volume, sentiment, over the modelled window."""
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    panel["session"] = pd.to_datetime(panel["session"])
    d = panel[panel.ticker == "AAPL"].sort_values("session")

    fig, ax = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True)
    ax[0].plot(d["session"], d["close"], color=C_MKT, lw=1.1)
    ax[0].set_ylabel("adj close")
    ax[0].set_title("AAPL: price, social volume and sentiment")

    ax[1].fill_between(d["session"], d["tweet_count"].fillna(0),
                       color=C_BASE, alpha=0.65, lw=0)
    ax[1].set_ylabel("tweets/session")

    ax[2].plot(d["session"], d["sent_mean"], color=C_SENT, lw=0.7, alpha=0.5)
    ax[2].plot(d["session"], d["sent_mean_ma5"], color=C_SENT, lw=1.5)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("sentiment")
    ax[2].set_xlabel("session")
    ax[2].annotate("thin line = daily, thick = 5-session trailing mean",
                   (0.02, 0.06), xycoords="axes fraction", fontsize=7)
    fig.savefig(FIG / "07_series_aapl.png")
    plt.close(fig)


def fig_actual_vs_pred() -> None:
    p = pd.read_parquet(ROOT / "data/processed/preds_all_tickers_n87_h1.parquet")
    y = p["ret_fwd_1"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].scatter(p["pred_market"], y, s=2, alpha=0.12, color=C_MKT, edgecolors="none")
    lim = 0.08
    ax[0].plot([-lim, lim], [-lim, lim], color="k", lw=0.9, ls="--")
    ax[0].set_xlim(-0.02, 0.02)
    ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel("predicted return")
    ax[0].set_ylabel("actual return")
    ax[0].set_title("Predictions barely vary; actuals do")

    ax[1].hist(y, bins=120, range=(-0.08, 0.08), color=C_BASE,
               alpha=0.55, label="actual", density=True)
    ax[1].hist(p["pred_market"], bins=120, range=(-0.08, 0.08), color=C_MKT,
               alpha=0.75, label="predicted (market)", density=True)
    ax[1].set_xlabel("return")
    ax[1].set_ylabel("density")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("The model shrinks toward zero")
    fig.savefig(FIG / "08_actual_vs_pred.png")
    plt.close(fig)


def main() -> None:
    for fn in (fig_coverage, fig_sentiment_models, fig_model_comparison,
               fig_ablation, fig_robustness, fig_per_ticker, fig_series,
               fig_actual_vs_pred):
        fn()
        print(f"  {fn.__name__}")
    print(f"\n{len(list(FIG.glob('*.png')))} figures -> {FIG}")


if __name__ == "__main__":
    sys.exit(main())
