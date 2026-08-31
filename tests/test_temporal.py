"""Leakage tests for tweet -> trading session assignment."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.temporal import assign_session, parse_twitter_timestamp  # noqa: E402

# Thu 2014-01-02 .. Wed 2014-01-08, skipping the weekend (4th, 5th).
SESSIONS = pd.DatetimeIndex(
    ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
)


def sess(ts: str) -> pd.Timestamp:
    s = assign_session(parse_twitter_timestamp(pd.Series([ts])), SESSIONS)
    return s.iloc[0]


def test_before_close_stays_same_session():
    # 15:30 ET == 20:30 UTC in January (EST, UTC-5)
    assert sess("Thu Jan 02 20:30:00 +0000 2014") == pd.Timestamp("2014-01-02")


def test_after_close_rolls_to_next_session():
    # 18:30 ET == 23:30 UTC -> not knowable at Thursday's close
    assert sess("Thu Jan 02 23:30:00 +0000 2014") == pd.Timestamp("2014-01-03")


def test_exactly_at_close_is_included():
    # 16:00:00 ET == 21:00:00 UTC, boundary is inclusive
    assert sess("Thu Jan 02 21:00:00 +0000 2014") == pd.Timestamp("2014-01-02")


def test_one_second_after_close_rolls():
    assert sess("Thu Jan 02 21:00:01 +0000 2014") == pd.Timestamp("2014-01-03")


def test_weekend_tweets_land_on_monday():
    for ts in ("Sat Jan 04 15:00:00 +0000 2014", "Sun Jan 05 15:00:00 +0000 2014"):
        assert sess(ts) == pd.Timestamp("2014-01-06")


def test_friday_after_close_lands_on_monday():
    # Fri 2014-01-03 22:00 UTC == 17:00 ET, after close, weekend intervenes
    assert sess("Fri Jan 03 22:00:00 +0000 2014") == pd.Timestamp("2014-01-06")


def test_holiday_gap_is_respected():
    # Calendar with 2014-01-06 removed, mimicking a market holiday.
    holiday_cal = SESSIONS.drop(pd.Timestamp("2014-01-06"))
    s = assign_session(
        parse_twitter_timestamp(pd.Series(["Sat Jan 04 15:00:00 +0000 2014"])), holiday_cal
    )
    assert s.iloc[0] == pd.Timestamp("2014-01-07")


def test_dst_boundary_uses_edt_not_fixed_offset():
    # July -> EDT (UTC-4). 20:30 UTC == 16:30 ET, which is AFTER the close.
    # A naive fixed UTC-5 assumption would wrongly call this 15:30 ET.
    july = pd.DatetimeIndex(["2014-07-10", "2014-07-11"])
    s = assign_session(
        parse_twitter_timestamp(pd.Series(["Thu Jul 10 20:30:00 +0000 2014"])), july
    )
    assert s.iloc[0] == pd.Timestamp("2014-07-11")


def test_tweet_after_last_session_is_dropped():
    assert pd.isna(sess("Fri Jan 31 15:00:00 +0000 2014"))


def test_assignment_never_moves_backwards():
    """No tweet may be assigned to a session earlier than its own timestamp."""
    ts = pd.Series(
        [
            "Thu Jan 02 20:30:00 +0000 2014",
            "Thu Jan 02 23:30:00 +0000 2014",
            "Sat Jan 04 15:00:00 +0000 2014",
            "Mon Jan 06 14:00:00 +0000 2014",
        ]
    )
    parsed = parse_twitter_timestamp(ts)
    assigned = assign_session(parsed, SESSIONS)
    # Session close (16:00 ET) must be >= tweet time for every assignment.
    closes = (
        assigned.dt.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    ).dt.tz_convert("UTC")
    assert (closes >= parsed).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
