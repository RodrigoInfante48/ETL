# PySpark Analysis Module

Distributed processing layer for the World Bank ETL pipeline. Replicates and extends the pandas + dbt transformations using Apache Spark, demonstrating how the same pipeline scales horizontally to large datasets.

## What it does

| Script | Role | Equivalent |
|--------|------|-----------|
| `spark_session.py` | SparkSession factory | — |
| `spark_extract.py` | Read from PostgreSQL (JDBC) or synthetic fallback | `World bank/extract.py` |
| `spark_transform.py` | Staging → Intermediate → Marts transformations | `world_bank_dbt/models/` |
| `spark_analysis.py` | Benchmarks, outlier detection, LATAM vs world | `6Sigma Seaborn Graphs/` |
| `spark_ml.py` | K-Means clustering, Linear Regression, feature importance | new |
| `run_spark_pipeline.py` | End-to-end orchestrator | `World bank/run_pipeline.py` |

## Why PySpark vs pandas + dbt?

| Aspect | pandas + dbt | PySpark |
|--------|-------------|---------|
| Scale | Single machine | Multi-node cluster |
| Data size | Millions of rows | Billions of rows |
| SQL transforms | ✅ dbt native | ✅ Spark SQL / DataFrame API |
| ML | scikit-learn | MLlib (distributed) |
| Storage | PostgreSQL | Parquet (columnar) |
| Best for | BI / reporting | Big data / streaming |

## Setup

```bash
pip install -r requirements.txt
```

For PostgreSQL reads via JDBC, download the driver:

```bash
curl -O https://jdbc.postgresql.org/download/postgresql-42.7.3.jar
```

Place the `.jar` in `PySpark Analysis/` — the session factory picks it up automatically.

## Run

```bash
# Full pipeline
python run_spark_pipeline.py

# Skip ML for quick testing
python run_spark_pipeline.py --skip-ml

# Skip both analysis and ML
python run_spark_pipeline.py --skip-analysis --skip-ml
```

No live database? No problem — the extract step falls back to realistic synthetic data automatically.

## Outputs

```
PySpark Analysis/output/
├── parquet/
│   ├── intermediate/          ← partitioned by country/year
│   ├── mart_country_summary/
│   └── mart_global_trend/
├── csv/
│   ├── benchmark/
│   ├── inflation_outliers/
│   ├── decade_comparison/
│   ├── latam_vs_world/
│   ├── ranking_stability/
│   ├── ml_clusters/
│   ├── ml_predictions/
│   └── ml_feature_importance/
└── models/                    ← reserved for saved ML models
```

## Key PySpark concepts demonstrated

- **Partitioned JDBC reads** — parallel ingestion using `partitionColumn` / `numPartitions`
- **Window functions** — `lag()`, `rank()`, `avg() over rows between` for time-series metrics
- **DataFrame caching** — `.cache()` before DataFrames reused across multiple operations
- **MLlib Pipelines** — `VectorAssembler → StandardScaler → KMeans/LinearRegression`
- **Parquet output** — columnar storage partitioned by country + year for fast BI queries
- **Synthetic fallback** — scripts run without a DB, making CI and demos easy
