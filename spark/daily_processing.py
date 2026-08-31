"""Spark job: distributed market aggregation and session-level tweet rollup.

The original project used Spark to compute a daily percentage change. That
idea is kept and extended: this job does the per-ticker windowed market
transformations and the tweet-to-session aggregation as partitioned Spark
work, writing Parquet partitioned by ticker.

HONEST SCOPE NOTE
-----------------
This dataset is ~49K market rows and ~119K tweets. That fits comfortably in
memory, and Spark is NOT faster here than pandas -- the JVM startup alone
costs more than the whole pandas path. Spark is used because the windowed
per-ticker logic is the kind that stops fitting in memory as the universe
grows from 87 tickers to thousands, and because partitioned Parquet output
is the sane way to hand this to a downstream feature store.

The README says exactly this rather than claiming big-data scale. The
correct claim is "implemented Spark-based distributed preprocessing", not
"processed data at scale".

Run:
    python3 spark/daily_processing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/spark"


def build_session() -> SparkSession:
    return (
        SparkSession.builder.appName("stock-sentiment-preprocessing")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def market_features(spark: SparkSession):
    """Per-ticker windowed returns, volatility and volume ratios."""
    # Explicit schema rather than inferSchema: at least one price file carries
    # a non-numeric sentinel, which makes inference fall back to STRING and
    # silently breaks every downstream arithmetic window.
    schema = StructType([
        StructField("Date", StringType(), True),
        StructField("Open", StringType(), True),
        StructField("High", StringType(), True),
        StructField("Low", StringType(), True),
        StructField("Close", StringType(), True),
        StructField("Adj Close", StringType(), True),
        StructField("Volume", StringType(), True),
    ])
    df = (
        spark.read.csv(str(ROOT / "data/external/stocknet-dataset/price/raw"),
                       header=True, schema=schema)
        .withColumn("ticker", F.upper(F.regexp_extract(F.input_file_name(), r"([^/]+)\.csv$", 1)))
        .withColumn("session", F.to_date("Date"))
        # try_cast, not cast: the raw files contain the literal string "null",
        # which ANSI-mode cast rejects outright. try_cast maps it to NULL so
        # the offending rows can be filtered rather than killing the job.
        .withColumn("adj_close", F.expr("try_cast(`Adj Close` as double)"))
        .withColumn("Volume", F.expr("try_cast(Volume as double)"))
        .filter(F.col("adj_close").isNotNull() & F.col("session").isNotNull())
    )

    w = Window.partitionBy("ticker").orderBy("session")
    prev_close = F.lag("adj_close", 1).over(w)
    # try_divide throughout: zero-volume sessions and zero prior closes exist
    # in the raw files. This mirrors the pandas path, which maps the same
    # cases to NaN rather than fabricating a sentinel value.
    df = df.withColumn("ret_1", F.try_divide(F.col("adj_close"), prev_close) - 1.0)

    # Trailing windows: rowsBetween(-(n-1), 0) is inclusive of the current
    # row and n-1 rows of history. Never a forward offset.
    for n in (5, 21):
        wn = w.rowsBetween(-(n - 1), 0)
        df = (
            df.withColumn(f"ret_mean_{n}", F.avg("ret_1").over(wn))
              .withColumn(f"vol_{n}", F.stddev("ret_1").over(wn))
              .withColumn(f"vol_rel_{n}", F.try_divide(F.col("Volume"), F.avg("Volume").over(wn)))
        )

    # Forward target, explicitly named so it can never be mistaken for input.
    df = df.withColumn(
        "ret_fwd_1", F.try_divide(F.lead("adj_close", 1).over(w), F.col("adj_close")) - 1.0
    )

    return df.select(
        "ticker", "session", "adj_close", "Volume", "ret_1",
        "ret_mean_5", "vol_5", "vol_rel_5", "ret_mean_21", "vol_21", "vol_rel_21",
        "ret_fwd_1",
    )


def tweet_rollup(spark: SparkSession):
    """Aggregate scored tweets to (ticker, session) in Spark.

    Consumes the session assignment produced by src/features/temporal.py --
    the 16:00 ET cutoff logic lives in exactly one place and is not
    reimplemented here, so the two paths cannot drift apart.
    """
    path = ROOT / "data/processed/daily_sentiment.parquet"
    if not path.exists():
        return None
    df = spark.read.parquet(str(path)).withColumn("session", F.to_date("session"))
    return df.groupBy("ticker", "session").agg(
        F.avg("sent_mean").alias("sent_mean"),
        F.avg("pos_ratio").alias("pos_ratio"),
        F.sum("tweet_count").alias("tweet_count"),
    )


def main() -> None:
    spark = build_session()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        mkt = market_features(spark)
        n_mkt, n_tickers = mkt.count(), mkt.select("ticker").distinct().count()
        print(f"market rows      : {n_mkt:,} across {n_tickers} tickers")

        tw = tweet_rollup(spark)
        if tw is not None:
            print(f"tweet sessions   : {tw.count():,}")
            joined = mkt.join(tw, on=["ticker", "session"], how="left")
        else:
            joined = mkt

        joined = joined.withColumn(
            "has_tweets", F.when(F.col("tweet_count").isNotNull(), 1).otherwise(0)
        )
        OUT.mkdir(parents=True, exist_ok=True)
        (joined.repartition("ticker")
               .write.mode("overwrite")
               .partitionBy("ticker")
               .parquet(str(OUT / "panel")))
        print(f"wrote partitioned parquet -> {OUT / 'panel'}")

        print("\nsanity: 5 rows")
        joined.filter(F.col("ticker") == "AAPL").orderBy("session").select(
            "ticker", "session", "ret_1", "vol_21", "tweet_count", "ret_fwd_1"
        ).show(5, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
