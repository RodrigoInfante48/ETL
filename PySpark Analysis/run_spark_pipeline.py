"""
run_spark_pipeline.py — Orchestrates the full PySpark ETL + analysis + ML pipeline.

Execution order:
  1. SparkSession init
  2. Extract (PostgreSQL or synthetic fallback)
  3. Transform (staging → intermediate → marts)
  4. Analysis (benchmarks, outliers, LATAM comparison, ranking)
  5. ML (K-Means clustering, Linear Regression, feature importance)
  6. Export marts to Parquet (production-grade storage format)
  7. SparkSession stop

Usage:
  python run_spark_pipeline.py
  python run_spark_pipeline.py --skip-ml      (skip ML for faster runs)
  python run_spark_pipeline.py --skip-analysis
"""

import argparse
import os
import time

from spark_session import get_spark_session, stop_spark
from spark_extract import extract
from spark_transform import run_all_transforms
from spark_analysis import run_all_analysis
from spark_ml import run_all_ml
from config import PARQUET_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Bank PySpark Pipeline")
    parser.add_argument("--skip-ml",       action="store_true", help="Skip ML experiments")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis step")
    return parser.parse_args()


def export_to_parquet(transforms: dict, base_dir: str) -> None:
    """Persist transformed DataFrames as partitioned Parquet files."""
    os.makedirs(base_dir, exist_ok=True)
    targets = {
        "intermediate":        ("country", "year"),
        "mart_country_summary": None,
        "mart_global_trend":   None,
    }
    for name, partition_cols in targets.items():
        df = transforms[name]
        path = os.path.join(base_dir, name)
        writer = df.write.mode("overwrite")
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.parquet(path)
        print(f"💾 Parquet → {path}/")


def main() -> None:
    args = parse_args()
    pipeline_start = time.time()

    print("=" * 65)
    print("🚀 World Bank ETL — PySpark Pipeline")
    print("=" * 65)

    # ── 1. SparkSession ───────────────────────────────────────────────────────
    spark = get_spark_session()
    print(f"✅ SparkSession ready | Spark {spark.version} | master: {spark.sparkContext.master}")

    try:
        # ── 2. Extract ────────────────────────────────────────────────────────
        print("\n[Step 1/5] EXTRACT")
        raw_df = extract(spark)
        raw_df.printSchema()

        # ── 3. Transform ──────────────────────────────────────────────────────
        print("\n[Step 2/5] TRANSFORM")
        transforms = run_all_transforms(raw_df)

        print("\n📊 Sample — intermediate layer (5 rows):")
        transforms["intermediate"].select(
            "country", "year", "gdp", "gdp_yoy_growth_pct", "economic_health_score", "is_latam"
        ).orderBy("country", "year").show(5)

        print("\n📊 Mart — country summary:")
        transforms["mart_country_summary"].show(truncate=False)

        print("\n📊 Mart — global trend (last 5 years):")
        transforms["mart_global_trend"].orderBy("year", ascending=False).show(5)

        # ── 4. Export Parquet ────────────────────────────────────────────────
        print("\n[Step 3/5] EXPORT TO PARQUET")
        export_to_parquet(transforms, PARQUET_DIR)

        # ── 5. Analysis ───────────────────────────────────────────────────────
        if not args.skip_analysis:
            print("\n[Step 4/5] ANALYSIS")
            analysis_results = run_all_analysis(transforms["intermediate"])

            print("\n📊 Inflation outliers (Z-score > 2):")
            analysis_results["inflation_outliers"].show(10, truncate=False)

            print("\n📊 LATAM vs. non-LATAM (latest 3 years):")
            analysis_results["latam_vs_world"].orderBy("year", ascending=False).show(6)

            print("\n📊 Ranking stability:")
            analysis_results["ranking_stability"].show(truncate=False)
        else:
            print("\n[Step 4/5] ANALYSIS — skipped")

        # ── 6. ML ─────────────────────────────────────────────────────────────
        if not args.skip_ml:
            print("\n[Step 5/5] MACHINE LEARNING")
            ml_results = run_all_ml(transforms["intermediate"])

            print("\n📊 Country clusters:")
            ml_results["clusters"].show(truncate=False)

            print(f"\n📊 LR metrics: {ml_results['lr_metrics']}")
        else:
            print("\n[Step 5/5] ML — skipped")

    finally:
        # Always stop Spark — even if an exception occurs
        stop_spark(spark)

    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 65}")
    print(f"✅ Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
