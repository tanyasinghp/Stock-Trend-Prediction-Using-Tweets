"""Ingest raw StockNet tweets, deduplicate, score sentiment, assign sessions.

Reads  : data/external/stocknet-dataset/tweet/raw/{TICKER}/{YYYY-MM-DD}
         (one JSON object per line, Twitter API format)
Writes : data/processed/daily_sentiment.parquet

The sentiment model is the classifier selected by src/sentiment/train_sentiment.py,
trained on the labeled corpus carried over from the original project. This is the
step that fixes the original architectural disconnect: the classifier's own
predictions now feed the forecasting dataset, rather than sentiment arriving
pre-computed from an unspecified source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.temporal import (  # noqa: E402
    aggregate_daily_sentiment,
    assign_session,
    parse_twitter_timestamp,
    sessions_from_prices,
)
from src.sentiment.finance_text import Preprocess_Tweets  # noqa: E402

STOCKNET = ROOT / "data/external/stocknet-dataset"


def read_raw_tweets() -> pd.DataFrame:
    rows = []
    tweet_root = STOCKNET / "tweet/raw"
    for ticker_dir in sorted(tweet_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        for day_file in ticker_dir.iterdir():
            with day_file.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows.append(
                        (ticker, obj.get("id_str"), obj.get("created_at"), obj.get("text"))
                    )
    df = pd.DataFrame(rows, columns=["ticker", "tweet_id", "created_at", "text"])
    print(f"raw tweet lines parsed        : {len(df):,}")
    return df


def main() -> None:
    df = read_raw_tweets()

    # --- deduplication -----------------------------------------------------
    # The same tweet can appear under several tickers (it cashtags both).
    # That is legitimate: it is genuine information for each. We therefore
    # dedupe on (ticker, tweet_id), not on tweet_id alone.
    before = len(df)
    df = df.dropna(subset=["tweet_id", "created_at", "text"])
    df = df.drop_duplicates(subset=["ticker", "tweet_id"])
    print(f"after dedup on (ticker,id)    : {len(df):,}  (removed {before - len(df):,})")
    print(f"distinct tweets overall       : {df.tweet_id.nunique():,}")

    # --- timestamps --------------------------------------------------------
    df["ts_utc"] = parse_twitter_timestamp(df["created_at"])
    df = df.dropna(subset=["ts_utc"])
    print(f"timestamp range (UTC)         : {df.ts_utc.min()} -> {df.ts_utc.max()}")

    # --- session assignment, per ticker's own trading calendar -------------
    parts = []
    for ticker, grp in df.groupby("ticker", sort=True):
        price_path = STOCKNET / f"price/raw/{ticker}.csv"
        if not price_path.exists():
            continue
        sessions = sessions_from_prices(pd.read_csv(price_path))
        grp = grp.copy()
        grp["session"] = assign_session(grp["ts_utc"], sessions)
        parts.append(grp)
    df = pd.concat(parts, ignore_index=True)
    n_unassigned = df["session"].isna().sum()
    df = df.dropna(subset=["session"])
    print(f"tweets with no later session  : {n_unassigned:,} (dropped)")
    print(f"tweets assigned to a session  : {len(df):,}")

    # --- how often does the cutoff actually bite? --------------------------
    same_day = (df["ts_utc"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
                == df["session"])
    print(f"rolled to a LATER session     : {(~same_day).sum():,} "
          f"({(~same_day).mean():.1%} of assigned tweets)")

    # --- sentiment scoring -------------------------------------------------
    model = joblib.load(ROOT / "models/sentiment_model.joblib")
    clean = Preprocess_Tweets(df[["text"]].rename(columns={"text": "Text"}))
    df["text_clean"] = clean["Text_Cleaned"].to_numpy()
    df = df[df["text_clean"].astype(str).str.strip().str.len() > 0].copy()
    df["sentiment_proba"] = model.predict_proba(df["text_clean"])[:, 1]
    print(f"scored tweets                 : {len(df):,}  "
          f"mean P(pos)={df.sentiment_proba.mean():.3f}")

    daily = aggregate_daily_sentiment(df)
    # Microsecond precision: pandas defaults to ns, which Spark's Parquet
    # reader rejects (PARQUET_TYPE_ILLEGAL on TIMESTAMP(NANOS)).
    daily["session"] = daily["session"].astype("datetime64[us]")
    out = ROOT / "data/processed/daily_sentiment.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out, index=False)
    print(f"\nticker-sessions with tweets   : {len(daily):,}")
    print(f"tickers                       : {daily.ticker.nunique()}")
    print(f"session range                 : {daily.session.min().date()} -> {daily.session.max().date()}")

    stats = {
        "raw_lines_parsed": int(before),
        "distinct_tweets": int(df.tweet_id.nunique()),
        "tweets_assigned": int(len(df)),
        "pct_rolled_to_later_session": float((~same_day).mean()),
        "ticker_sessions": int(len(daily)),
        "n_tickers": int(daily.ticker.nunique()),
        "session_min": str(daily.session.min().date()),
        "session_max": str(daily.session.max().date()),
    }
    (ROOT / "results/sentiment/corpus_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
