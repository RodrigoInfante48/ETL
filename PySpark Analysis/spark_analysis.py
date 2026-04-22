"""
spark_analysis.py — Advanced analytics with PySpark SQL and window functions.

Covers:
  - Country vs global benchmark comparisons
  - Outlier detection (Z-score & IQR methods)
  - Decade-over-decade comparisons
  - LATAM vs non-LATAM economic contrasts
  - Ranking stability analysis
"""

import os
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from config import OUTPUT_DIR, CSV_DIR


# ── Benchmark comparison ──────────────────────────────────────────────────────

def country_vs_global_benchmark(df: DataFrame) -> DataFrame:
    """
    For each (country, year) row, compute how far above/below the country
    sits relative to the global average that year.
    """
    print("📊 Computing country vs. global benchmark…")

    w_year = Window.partitionBy("year")

    benchmarked = (
        df
        .withColumn("world_avg_gdp_pc",    F.round(F.avg("gdp_per_capita").over(w_year), 2))
        .withColumn("world_avg_inflation",  F.round(F.avg("inflation").over(w_year), 2))
        .withColumn("world_avg_unemployment", F.round(F.avg("unemployment_rate").over(w_year), 2))
        .withColumn(
            "gdp_pc_vs_world_pct",
            F.round(
                (F.col("gdp_per_capita") - F.col("world_avg_gdp_pc")) / F.col("world_avg_gdp_pc") * 100,
                1
            )
        )
        .withColumn(
            "inflation_delta_vs_world",
            F.round(F.col("inflation") - F.col("world_avg_inflation"), 2)
        )
        .select(
            "country", "year",
            "gdp_per_capita", "world_avg_gdp_pc", "gdp_pc_vs_world_pct",
            "inflation", "world_avg_inflation", "inflation_delta_vs_world",
            "unemployment_rate", "world_avg_unemployment",
        )
        .orderBy("year", "country")
    )

    print("✅ Benchmark comparison ready")
    return benchmarked


# ── Outlier detection ─────────────────────────────────────────────────────────

def detect_outliers_zscore(df: DataFrame, column: str = "inflation", threshold: float = 2.0) -> DataFrame:
    """
    Flag rows where the Z-score of `column` (within year) exceeds `threshold`.
    Z-score = (value - mean) / std_dev
    """
    print(f"📊 Z-score outlier detection on '{column}' (threshold={threshold})…")

    w_year = Window.partitionBy("year")

    result = (
        df
        .withColumn("_mean", F.avg(column).over(w_year))
        .withColumn("_std",  F.stddev(column).over(w_year))
        .withColumn(
            f"{column}_zscore",
            F.when(
                F.col("_std") > 0,
                F.round((F.col(column) - F.col("_mean")) / F.col("_std"), 2)
            ).otherwise(F.lit(0.0))
        )
        .withColumn(
            f"is_{column}_outlier",
            F.abs(F.col(f"{column}_zscore")) > threshold
        )
        .drop("_mean", "_std")
        .filter(F.col(f"is_{column}_outlier"))
        .select("country", "year", column, f"{column}_zscore", f"is_{column}_outlier")
        .orderBy(F.desc(F.abs(F.col(f"{column}_zscore"))))
    )

    count = result.count()
    print(f"✅ Found {count} outlier rows for '{column}'")
    return result


def detect_outliers_iqr(df: DataFrame, column: str = "unemployment_rate") -> DataFrame:
    """
    Flag rows outside IQR bounds: [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Applied globally across all years.
    """
    print(f"📊 IQR outlier detection on '{column}'…")

    # Spark's approxQuantile is fast on large datasets; exactness not needed here
    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    print(f"   Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  bounds=[{lower:.2f}, {upper:.2f}]")

    result = (
        df
        .filter((F.col(column) < lower) | (F.col(column) > upper))
        .select("country", "year", column)
        .withColumn("iqr_lower_bound", F.lit(round(lower, 2)))
        .withColumn("iqr_upper_bound", F.lit(round(upper, 2)))
        .orderBy(F.desc(column))
    )

    print(f"✅ Found {result.count()} IQR outliers for '{column}'")
    return result


# ── Decade comparison ─────────────────────────────────────────────────────────

def decade_comparison(df: DataFrame) -> DataFrame:
    """
    Summarise economic performance by decade (2000s, 2010s, 2020s) per country.
    """
    print("📊 Building decade-over-decade comparison…")

    result = (
        df
        .withColumn(
            "decade",
            F.when(F.col("year").between(2000, 2009), "2000s")
             .when(F.col("year").between(2010, 2019), "2010s")
             .when(F.col("year").between(2020, 2029), "2020s")
             .otherwise("other")
        )
        .groupBy("country", "is_latam", "decade")
        .agg(
            F.round(F.avg("gdp"), 2).alias("avg_gdp"),
            F.round(F.avg("gdp_per_capita"), 2).alias("avg_gdp_per_capita"),
            F.round(F.avg("inflation"), 2).alias("avg_inflation"),
            F.round(F.avg("unemployment_rate"), 2).alias("avg_unemployment"),
            F.round(F.avg("gdp_yoy_growth_pct"), 2).alias("avg_yoy_growth"),
            F.round(F.avg("economic_health_score"), 1).alias("avg_health_score"),
        )
        .orderBy("country", "decade")
    )

    print(f"✅ Decade comparison: {result.count()} rows")
    return result


# ── LATAM vs non-LATAM ────────────────────────────────────────────────────────

def latam_vs_world(df: DataFrame) -> DataFrame:
    """Compare LATAM aggregate vs. non-LATAM aggregate by year."""
    print("📊 LATAM vs. non-LATAM yearly comparison…")

    result = (
        df
        .groupBy("year", "is_latam")
        .agg(
            F.round(F.avg("gdp_per_capita"), 2).alias("avg_gdp_per_capita"),
            F.round(F.avg("inflation"), 2).alias("avg_inflation"),
            F.round(F.avg("unemployment_rate"), 2).alias("avg_unemployment"),
            F.round(F.avg("economic_health_score"), 1).alias("avg_health_score"),
            F.count("country").alias("country_count"),
        )
        .withColumn("group", F.when(F.col("is_latam"), "LATAM").otherwise("Non-LATAM"))
        .drop("is_latam")
        .orderBy("year", "group")
    )

    print(f"✅ LATAM vs. world: {result.count()} rows")
    return result


# ── Ranking stability ─────────────────────────────────────────────────────────

def ranking_stability(df: DataFrame) -> DataFrame:
    """
    Measure how stable each country's GDP rank is across years.
    Low std-dev → consistent position in the world economy.
    """
    print("📊 Computing ranking stability…")

    result = (
        df
        .groupBy("country")
        .agg(
            F.round(F.avg("gdp_rank_in_year"), 1).alias("avg_rank"),
            F.round(F.stddev("gdp_rank_in_year"), 2).alias("rank_std_dev"),
            F.min("gdp_rank_in_year").alias("best_rank"),
            F.max("gdp_rank_in_year").alias("worst_rank"),
        )
        .withColumn("rank_range", F.col("worst_rank") - F.col("best_rank"))
        .orderBy("avg_rank")
    )

    print("✅ Ranking stability computed")
    return result


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_analysis(intermediate_df: DataFrame) -> dict[str, DataFrame]:
    """Run all analysis steps and return results dict."""
    os.makedirs(CSV_DIR, exist_ok=True)

    results = {
        "benchmark":         country_vs_global_benchmark(intermediate_df),
        "inflation_outliers": detect_outliers_zscore(intermediate_df, "inflation"),
        "unemp_outliers":    detect_outliers_iqr(intermediate_df, "unemployment_rate"),
        "decade_comparison": decade_comparison(intermediate_df),
        "latam_vs_world":    latam_vs_world(intermediate_df),
        "ranking_stability": ranking_stability(intermediate_df),
    }

    # Persist key outputs as CSV for downstream use (e.g. Power BI)
    for name, result_df in results.items():
        path = f"{CSV_DIR}/{name}"
        result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
        print(f"💾 Saved {name} → {path}/")

    return results
