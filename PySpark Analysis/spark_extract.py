"""
spark_extract.py — Read economic data into a Spark DataFrame.

Priority:
  1. PostgreSQL via JDBC (production)
  2. Synthetic data (development / CI — no DB required)
"""

import random
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType,
)
from config import JDBC_URL, JDBC_PROPERTIES, SOURCE_TABLE, COUNTRIES, YEAR_RANGE


# ── Schema ───────────────────────────────────────────────────────────────────

ECONOMIC_SCHEMA = StructType([
    StructField("country",           StringType(),  True),
    StructField("country_code",      StringType(),  True),
    StructField("year",              IntegerType(), True),
    StructField("gdp",               DoubleType(),  True),
    StructField("gdp_per_capita",    DoubleType(),  True),
    StructField("inflation",         DoubleType(),  True),
    StructField("unemployment_rate", DoubleType(),  True),
    StructField("population",        DoubleType(),  True),
])

COUNTRY_CODES = {
    "Colombia": "COL", "Brazil": "BRA", "Mexico": "MEX",
    "Argentina": "ARG", "Chile": "CHL", "Peru": "PER",
    "Ecuador": "ECU", "United States": "USA", "China": "CHN", "Germany": "DEU",
}

# Base GDP values (USD billions) — used to generate realistic synthetic data
BASE_GDP = {
    "Colombia": 100, "Brazil": 1500, "Mexico": 900, "Argentina": 300,
    "Chile": 150, "Peru": 120, "Ecuador": 70, "United States": 10000,
    "China": 1500, "Germany": 2000,
}


# ── Public API ────────────────────────────────────────────────────────────────

def extract_from_postgres(spark: SparkSession) -> DataFrame:
    """Read economic_indicators from PostgreSQL via JDBC."""
    print("📡 Connecting to PostgreSQL via JDBC…")
    df = (
        spark.read
        .format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", SOURCE_TABLE)
        .option("driver", JDBC_PROPERTIES["driver"])
        .option("user", JDBC_PROPERTIES["user"])
        .option("password", JDBC_PROPERTIES["password"])
        # Partition by year so Spark can read rows in parallel
        .option("partitionColumn", "year")
        .option("lowerBound", str(YEAR_RANGE[0]))
        .option("upperBound", str(YEAR_RANGE[1]))
        .option("numPartitions", "8")
        .load()
    )
    count = df.count()
    print(f"✅ Loaded {count:,} rows from PostgreSQL ({df.rdd.getNumPartitions()} partitions)")
    return df


def extract_synthetic(spark: SparkSession) -> DataFrame:
    """Generate realistic synthetic economic data — no DB required."""
    print("🔧 Generating synthetic World Bank data (no DB connection)…")

    random.seed(42)
    rows = []
    for country in COUNTRIES:
        base_gdp = BASE_GDP[country] * 1e9
        base_pop = 50_000_000 if country not in ("China", "United States", "Brazil") else 200_000_000
        code = COUNTRY_CODES[country]

        for year in range(YEAR_RANGE[0], YEAR_RANGE[1] + 1):
            growth = 1 + random.uniform(-0.03, 0.10)
            base_gdp *= growth
            pop = base_pop * (1 + 0.01 * (year - YEAR_RANGE[0])) + random.uniform(-500_000, 500_000)
            gdp_pc = base_gdp / pop
            inflation = round(random.uniform(1.0, 8.0) + (15 if country == "Argentina" else 0), 2)
            unemployment = round(random.uniform(3.0, 12.0), 2)

            rows.append((
                country, code, year,
                round(base_gdp, 2),
                round(gdp_pc, 2),
                inflation,
                unemployment,
                round(pop, 0),
            ))

    df = spark.createDataFrame(rows, schema=ECONOMIC_SCHEMA)
    print(f"✅ Synthetic dataset: {df.count():,} rows × {len(df.columns)} columns")
    return df


def extract(spark: SparkSession) -> DataFrame:
    """
    Try PostgreSQL first; fall back to synthetic data on any error.
    Returns a cached DataFrame ready for downstream transformations.
    """
    try:
        df = extract_from_postgres(spark)
    except Exception as exc:
        print(f"⚠️  PostgreSQL unavailable ({exc.__class__.__name__}). Using synthetic data.")
        df = extract_synthetic(spark)

    # Cache early — every downstream step re-uses this DF
    df = df.cache()
    df.count()  # materialise the cache
    print(f"📊 Schema: {df.columns}")
    return df
