"""
spark_transform.py — Distributed transformations on economic data.

Mirrors what dbt does in SQL but executed on the Spark engine:
  - Staging   : type casting, null filtering, normalisation
  - Intermediate: derived metrics (YoY growth, economic health score, LATAM flag)
  - Marts     : country summaries, global yearly trends, performance tiers
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from config import LATAM_COUNTRIES


# ── Staging layer ─────────────────────────────────────────────────────────────

def transform_staging(df: DataFrame) -> DataFrame:
    """
    Clean and type-enforce the raw DataFrame.
    Equivalent to dbt model: stg_economic_indicators
    """
    print("🔧 [Staging] Cleaning raw data…")

    staged = (
        df
        # Drop rows where GDP is null — without GDP the row is useless for analysis
        .filter(F.col("gdp").isNotNull())
        .filter(F.col("year").between(2000, 2023))
        # Normalise country name casing
        .withColumn("country", F.initcap(F.col("country")))
        # Ensure numeric precision
        .withColumn("gdp",               F.round("gdp", 2))
        .withColumn("gdp_per_capita",    F.round("gdp_per_capita", 2))
        .withColumn("inflation",         F.round("inflation", 2))
        .withColumn("unemployment_rate", F.round("unemployment_rate", 2))
        .withColumn("population",        F.round("population", 0).cast("long"))
    )

    count = staged.count()
    print(f"✅ [Staging] {count:,} rows after cleaning")
    return staged


# ── Intermediate layer ────────────────────────────────────────────────────────

def transform_intermediate(df: DataFrame) -> DataFrame:
    """
    Add derived metrics.
    Equivalent to dbt model: int_country_metrics
    """
    print("🔧 [Intermediate] Computing derived metrics…")

    # Window ordered by year within each country — for lag-based calculations
    w_country_year = Window.partitionBy("country").orderBy("year")

    # Window for ranking countries within a given year
    w_year_rank = Window.partitionBy("year").orderBy(F.desc("gdp"))

    enriched = (
        df
        # YoY GDP growth rate
        .withColumn("gdp_prev_year",  F.lag("gdp", 1).over(w_country_year))
        .withColumn(
            "gdp_yoy_growth_pct",
            F.when(
                F.col("gdp_prev_year").isNotNull() & (F.col("gdp_prev_year") != 0),
                F.round(
                    (F.col("gdp") - F.col("gdp_prev_year")) / F.col("gdp_prev_year") * 100,
                    2
                )
            )
        )
        .drop("gdp_prev_year")

        # 3-year rolling average GDP (smooths out single-year shocks)
        .withColumn(
            "gdp_3yr_avg",
            F.round(
                F.avg("gdp").over(w_country_year.rowsBetween(-2, 0)),
                2
            )
        )

        # Economic health score 0–100
        # Higher GDP per capita → good; lower inflation → good; lower unemployment → good
        # Each component capped at its respective "good" threshold
        .withColumn(
            "gdp_pc_score",
            F.least(F.col("gdp_per_capita") / 500, F.lit(40.0))
        )
        .withColumn(
            "inflation_score",
            F.greatest(F.lit(0.0), F.lit(30.0) - F.col("inflation") * 2)
        )
        .withColumn(
            "unemployment_score",
            F.greatest(F.lit(0.0), F.lit(30.0) - F.col("unemployment_rate") * 2)
        )
        .withColumn(
            "economic_health_score",
            F.round(
                F.col("gdp_pc_score") + F.col("inflation_score") + F.col("unemployment_score"),
                1
            )
        )
        .drop("gdp_pc_score", "inflation_score", "unemployment_score")

        # LATAM flag
        .withColumn(
            "is_latam",
            F.col("country").isin(LATAM_COUNTRIES)
        )

        # GDP rank within year (1 = largest economy that year)
        .withColumn("gdp_rank_in_year", F.rank().over(w_year_rank))
    )

    print("✅ [Intermediate] Derived columns added: gdp_yoy_growth_pct, gdp_3yr_avg, economic_health_score, is_latam, gdp_rank_in_year")
    return enriched


# ── Marts layer ───────────────────────────────────────────────────────────────

def build_mart_country_summary(df: DataFrame) -> DataFrame:
    """
    Country-level aggregations across all years.
    Equivalent to dbt model: mart_country_summary
    """
    print("📊 [Marts] Building country summary…")

    summary = (
        df.groupBy("country", "country_code", "is_latam")
        .agg(
            F.count("year").alias("years_of_data"),
            F.round(F.avg("gdp"), 2).alias("avg_gdp"),
            F.round(F.max("gdp"), 2).alias("peak_gdp"),
            F.round(F.min("gdp"), 2).alias("trough_gdp"),
            F.round(F.avg("gdp_per_capita"), 2).alias("avg_gdp_per_capita"),
            F.round(F.avg("inflation"), 2).alias("avg_inflation"),
            F.round(F.avg("unemployment_rate"), 2).alias("avg_unemployment"),
            F.round(F.avg("economic_health_score"), 1).alias("avg_health_score"),
            F.round(F.avg("gdp_yoy_growth_pct"), 2).alias("avg_yoy_growth_pct"),
        )
        # Performance tier — mirrors dbt CASE logic
        .withColumn(
            "performance_tier",
            F.when(F.col("avg_health_score") >= 60, "High Performer")
             .when(F.col("avg_health_score") >= 40, "Mid Performer")
             .when(F.col("avg_health_score") >= 20, "Developing")
             .otherwise("Needs Attention")
        )
        .orderBy(F.desc("avg_gdp"))
    )

    print(f"✅ [Marts] Country summary: {summary.count()} countries")
    return summary


def build_mart_global_trend(df: DataFrame) -> DataFrame:
    """
    Yearly global aggregations.
    Equivalent to dbt model: mart_global_trend
    """
    print("📊 [Marts] Building global trend…")

    w_year = Window.orderBy("year")

    trend = (
        df.groupBy("year")
        .agg(
            F.round(F.sum("gdp"), 2).alias("world_gdp"),
            F.round(F.avg("gdp_per_capita"), 2).alias("avg_gdp_per_capita"),
            F.round(F.avg("inflation"), 2).alias("avg_inflation"),
            F.round(F.avg("unemployment_rate"), 2).alias("avg_unemployment"),
            F.count("country").alias("country_count"),
        )
        .withColumn("world_gdp_prev", F.lag("world_gdp", 1).over(w_year))
        .withColumn(
            "world_gdp_yoy_pct",
            F.when(
                F.col("world_gdp_prev").isNotNull(),
                F.round((F.col("world_gdp") - F.col("world_gdp_prev")) / F.col("world_gdp_prev") * 100, 2)
            )
        )
        .drop("world_gdp_prev")
        .orderBy("year")
    )

    print(f"✅ [Marts] Global trend: {trend.count()} years")
    return trend


# ── Convenience orchestrator ──────────────────────────────────────────────────

def run_all_transforms(raw_df: DataFrame) -> dict[str, DataFrame]:
    """
    Run the full staging → intermediate → marts pipeline.
    Returns a dict with all output DataFrames.
    """
    staged = transform_staging(raw_df)
    enriched = transform_intermediate(staged)
    enriched = enriched.cache()
    enriched.count()  # materialise

    country_summary = build_mart_country_summary(enriched)
    global_trend = build_mart_global_trend(enriched)

    return {
        "staged": staged,
        "intermediate": enriched,
        "mart_country_summary": country_summary,
        "mart_global_trend": global_trend,
    }
