-- Panel assembly: join market features to session-aligned social sentiment.
--
-- Design notes
-- ------------
-- 1. LEFT JOIN, not INNER. A trading session with zero tweets is a real
--    observation and must survive; dropping it would silently bias the panel
--    toward heavily-discussed stocks and break the price series' contiguity.
--
-- 2. The join key is (ticker, session), where `session` was already assigned
--    by the 16:00 ET cutoff rule in src/features/temporal.py. No date
--    arithmetic happens here -- doing it in SQL would duplicate the
--    timezone/holiday logic in a second place and let the two drift apart.
--
-- 3. Restricted to the window where BOTH modalities exist. Outside it, the
--    market-only and market+sentiment arms would not be comparable.

DROP TABLE IF EXISTS panel;
DROP VIEW  IF EXISTS coverage_report;

CREATE TEMP TABLE overlap AS
SELECT
    MAX((SELECT MIN(session) FROM sentiment),
        (SELECT MIN(session) FROM market))   AS lo,
    MIN((SELECT MAX(session) FROM sentiment),
        (SELECT MAX(session) FROM market))   AS hi;

-- Tickers with enough sessions in the overlap window to support
-- trailing-window features and a walk-forward split.
CREATE TEMP TABLE eligible_tickers AS
SELECT m.ticker
FROM market m, overlap o
WHERE m.session BETWEEN o.lo AND o.hi
GROUP BY m.ticker
HAVING COUNT(*) >= 200;

CREATE TABLE panel AS
SELECT
    m.*,
    s.sent_mean,
    s.sent_median,
    s.sent_std,
    s.pos_ratio,
    s.tweet_count,
    s.log_tweet_count
FROM market m
JOIN overlap o
JOIN eligible_tickers e ON e.ticker = m.ticker
LEFT JOIN sentiment s
       ON s.ticker  = m.ticker
      AND s.session = m.session
WHERE m.session BETWEEN o.lo AND o.hi;

CREATE INDEX idx_panel_ticker_session ON panel (ticker, session);

-- Per-ticker social coverage, used to split stocks into high/low tweet-volume
-- groups for the robustness analysis.
CREATE VIEW coverage_report AS
SELECT
    ticker,
    COUNT(*)                                            AS n_sessions,
    SUM(CASE WHEN tweet_count IS NOT NULL THEN 1 ELSE 0 END) AS n_sessions_with_tweets,
    ROUND(AVG(CASE WHEN tweet_count IS NOT NULL THEN 1.0 ELSE 0.0 END), 4) AS coverage_rate,
    ROUND(COALESCE(SUM(tweet_count), 0), 0)             AS total_tweets,
    ROUND(AVG(tweet_count), 2)                          AS mean_tweets_per_covered_session
FROM panel
GROUP BY ticker;
