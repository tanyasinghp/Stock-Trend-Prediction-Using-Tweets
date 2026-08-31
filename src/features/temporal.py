"""Causal temporal alignment of tweets to trading sessions.

CONVENTION (this is the thing to be able to explain in an interview)
--------------------------------------------------------------------
Tweet timestamps in StockNet are Twitter `created_at` strings in UTC.
US equity markets close at 16:00 America/New_York, which is UTC-5 (EST)
or UTC-4 (EDT) depending on the date. We therefore convert every tweet
to America/New_York rather than assuming a fixed offset.

For a trading session D (a date on which the exchange actually traded):

    information_set(D) = { tweets t : prev_session(D) 16:00 ET  <  t  <=  D 16:00 ET }

    features(D)  are built only from information_set(D) and from prices
                 observed up to and including D's close
    target(D)    = return from D's close to next_session(D)'s close

Consequences that fall out of this rule, all handled explicitly:

* A tweet posted at 18:30 ET on Monday is NOT in Monday's information set.
  It lands in Tuesday's, because at Monday's close nobody had seen it.
* Weekend tweets (Fri 16:00 ET -> Mon 16:00 ET) accumulate into Monday's
  information set. That makes Monday's tweet counts structurally larger,
  which is why tweet_count is a modelled feature and not a nuisance.
* Market holidays are handled the same way as weekends. We never hard-code
  a holiday list: the set of valid sessions is taken from the price file
  itself, which by construction only contains days the exchange traded.
* Sessions with no tweets are retained with count 0 and NaN sentiment,
  rather than dropped, so the price series stays contiguous.

The trading calendar is derived from observed price dates, so it cannot
drift out of sync with the data being modelled.
"""
from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)
TWITTER_TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_twitter_timestamp(series: pd.Series) -> pd.Series:
    """Parse Twitter `created_at` strings to tz-aware UTC timestamps."""
    return pd.to_datetime(series, format=TWITTER_TS_FORMAT, utc=True, errors="coerce")


def to_market_time(ts_utc: pd.Series) -> pd.Series:
    """Convert tz-aware UTC timestamps to US market local time."""
    return ts_utc.dt.tz_convert(MARKET_TZ)


def assign_session(ts_utc: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Map each tweet to the trading session whose information set contains it.

    A tweet at or before 16:00 ET on a trading day belongs to that day's
    session. A tweet after 16:00 ET -- or on a weekend/holiday -- belongs to
    the next trading session. Tweets after the final session are dropped
    (NaT), since no target exists for them.

    Parameters
    ----------
    ts_utc : tz-aware UTC timestamps
    sessions : sorted DatetimeIndex of valid trading dates (tz-naive dates)

    Returns
    -------
    Series of session dates (tz-naive, normalized), NaT where unassignable.
    """
    local = to_market_time(ts_utc)

    # "Effective date": the calendar date whose close first sees this tweet.
    # After the close, roll to the following calendar day.
    eff = local.dt.normalize().dt.tz_localize(None)
    after_close = local.dt.time > MARKET_CLOSE
    eff = eff + pd.to_timedelta(after_close.astype(int), unit="D")

    # Roll forward to the next real trading session (covers weekends/holidays).
    sess = pd.DatetimeIndex(sessions).normalize().sort_values()
    idx = sess.searchsorted(eff.to_numpy(), side="left")

    out = np.full(len(eff), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = (idx < len(sess)) & eff.notna().to_numpy()
    out[valid] = sess.to_numpy()[idx[valid]]
    return pd.Series(out, index=ts_utc.index)


def sessions_from_prices(price_df: pd.DataFrame, date_col: str = "Date") -> pd.DatetimeIndex:
    """Derive the trading calendar from observed price dates."""
    return pd.DatetimeIndex(pd.to_datetime(price_df[date_col]).dt.normalize().unique()).sort_values()


def aggregate_daily_sentiment(tweets: pd.DataFrame) -> pd.DataFrame:
    """Collapse session-assigned, scored tweets into per-(ticker, session) features.

    Expects columns: ticker, session, sentiment_proba (P(positive) in [0,1]).
    All outputs are functions of that session's information set only.
    """
    t = tweets.dropna(subset=["session"]).copy()
    t["polarity"] = 2.0 * t["sentiment_proba"] - 1.0  # rescale to [-1, 1]
    t["is_pos"] = (t["sentiment_proba"] >= 0.5).astype(int)

    g = t.groupby(["ticker", "session"], observed=True)
    out = g.agg(
        sent_mean=("polarity", "mean"),
        sent_median=("polarity", "median"),
        sent_std=("polarity", "std"),
        pos_ratio=("is_pos", "mean"),
        tweet_count=("polarity", "size"),
    ).reset_index()
    out["sent_std"] = out["sent_std"].fillna(0.0)  # single-tweet sessions
    out["log_tweet_count"] = np.log1p(out["tweet_count"])
    return out
