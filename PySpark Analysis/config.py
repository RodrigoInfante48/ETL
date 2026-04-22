"""
PySpark Analysis — centralized configuration.
Mirrors the credentials from 'World bank/config.py' and adds Spark-specific settings.
"""

# ── PostgreSQL (JDBC) ────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "world_bank_db",
    "user": "postgres",
    "password": "password",
}

JDBC_URL = (
    f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

JDBC_PROPERTIES = {
    "user": DB_CONFIG["user"],
    "password": DB_CONFIG["password"],
    "driver": "org.postgresql.Driver",
}

# Path to the PostgreSQL JDBC driver JAR (download from https://jdbc.postgresql.org/)
JDBC_JAR_PATH = "postgresql-42.7.3.jar"

# ── Source table ─────────────────────────────────────────────────────────────
SOURCE_TABLE = "public.economic_indicators"

# ── SparkSession settings ────────────────────────────────────────────────────
SPARK_CONFIG = {
    "app_name": "WorldBankETL",
    "master": "local[*]",          # all available cores; change to yarn/k8s for clusters
    "executor_memory": "2g",
    "driver_memory": "2g",
    "log_level": "WARN",           # reduces console noise; use INFO for debugging
}

# ── Domain constants ─────────────────────────────────────────────────────────
COUNTRIES = [
    "Colombia", "Brazil", "Mexico", "Argentina", "Chile",
    "Peru", "Ecuador", "United States", "China", "Germany",
]

INDICATORS = ["gdp", "gdp_per_capita", "inflation", "unemployment_rate", "population"]

LATAM_COUNTRIES = ["Colombia", "Brazil", "Mexico", "Argentina", "Chile", "Peru", "Ecuador"]

YEAR_RANGE = (2000, 2023)

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
PARQUET_DIR = f"{OUTPUT_DIR}/parquet"
CSV_DIR = f"{OUTPUT_DIR}/csv"
CHARTS_DIR = f"{OUTPUT_DIR}/charts"
ML_MODELS_DIR = f"{OUTPUT_DIR}/models"
