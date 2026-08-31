"""Assemble the modelling panel.

The market/sentiment join, calendar alignment and coverage filtering are done
in SQL (SQLite), not in pandas. Feature *arithmetic* stays in pandas because
trailing windows are clearer and safer there; the SQL layer owns the relational
work: loading, joining on (ticker, session), left-joining so that zero-tweet
sessions survive, and applying the coverage thresholds.

Writes:
    data/processed/panel.db        (queryable SQLite database)
    data/processed/panel.parquet   (modelling table)
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.market import (  # noqa: E402
    TARGET_PREFIX,
    add_sentiment_dynamics,
    build_ticker_features,
)

STOCKNET = ROOT / "data/external/stocknet-dataset"
DB_PATH = ROOT / "data/processed/panel.db"
MIN_SESSIONS_PER_TICKER = 200


def load_market() -> pd.DataFrame:
    frames = []
    for f in sorted((STOCKNET / "price/raw").glob("*.csv")):
        ticker = f.stem.upper()
        feats = build_ticker_features(pd.read_csv(f))
        feats.insert(0, "ticker", ticker)
        frames.append(feats)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    market = load_market()
    sentiment = pd.read_parquet(ROOT / "data/processed/daily_sentiment.parquet")
    sentiment["session"] = pd.to_datetime(sentiment["session"]).dt.normalize()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)

    market.assign(session=market.session.dt.strftime("%Y-%m-%d")).to_sql(
        "market", con, index=False
    )
    sentiment.assign(session=sentiment.session.dt.strftime("%Y-%m-%d")).to_sql(
        "sentiment", con, index=False
    )
    con.executescript((ROOT / "sql/transformations.sql").read_text())

    panel = pd.read_sql(
        "SELECT * FROM panel ORDER BY ticker, session", con, parse_dates=["session"]
    )
    coverage = pd.read_sql("SELECT * FROM coverage_report", con)
    con.commit()
    con.close()

    panel = add_sentiment_dynamics(panel)

    # has_tweets marks sessions with zero social coverage. Sentiment columns
    # stay NaN there rather than being imputed to a neutral value -- XGBoost
    # handles NaN natively, and imputing would invent information.
    panel["has_tweets"] = panel["tweet_count"].notna().astype(int)

    out = ROOT / "data/processed/panel.parquet"
    panel.to_parquet(out, index=False)
    coverage.to_csv(ROOT / "results/forecasting/coverage_report.csv", index=False)

    tgt = f"{TARGET_PREFIX}1"
    modelable = panel.dropna(subset=[tgt])
    stats = {
        "panel_rows": int(len(panel)),
        "rows_with_target": int(len(modelable)),
        "n_tickers": int(panel.ticker.nunique()),
        "session_min": str(panel.session.min().date()),
        "session_max": str(panel.session.max().date()),
        "pct_sessions_with_tweets": float(panel.has_tweets.mean()),
        "median_tweets_per_covered_session": float(panel.tweet_count.median()),
    }
    (ROOT / "results/forecasting/panel_stats.json").write_text(json.dumps(stats, indent=2))
    for k, v in stats.items():
        print(f"{k:36s}: {v}")


if __name__ == "__main__":
    main()
