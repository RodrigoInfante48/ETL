# CLAUDE.md — ETL Project Context

## Project Overview

Full Modern Data Stack ETL pipeline using **World Bank open economic data** (GDP, inflation, unemployment, population) for 10 countries from 2000–2023. The pipeline flows from raw API extraction through distributed processing, SQL transformation, statistical analysis, and automated email reporting.

## Architecture

```
World Bank API
    ↓ Python (requests, pandas, SQLAlchemy)
PostgreSQL — economic_indicators table
    ↓ dbt (staging → intermediate → marts)
marts.mart_country_summary / mart_global_trend
    ↓ PostgreSQL Materialized Views
mv_01_country_profile … mv_04_outliers_detection
    ↓ Python (seaborn, matplotlib)
6-Sigma statistical charts
    ↓ Python (smtplib, email)
HTML report → stakeholder inbox
```

PySpark sits as an **alternative/parallel distributed processing layer** that can replace or complement the pandas + dbt path for large-scale workloads.

## Module Map

| Folder | Stage | Key Files | Libraries |
|--------|-------|-----------|-----------|
| `World bank/` | Extract & Load | `extract.py`, `transform.py`, `load.py`, `run_pipeline.py` | requests, pandas, SQLAlchemy, psycopg2 |
| `world_bank_dbt/` | Transform (ELT) | `models/staging/`, `models/intermediate/`, `models/marts/` | dbt-core, dbt-postgres |
| `PostgreSQL Materialized Views/` | Optimization | `mv_01_*.sql` … `mv_04_*.sql` | Native SQL |
| `6Sigma Seaborn Graphs/` | Analysis | `sigma_01_descriptive.py` … `sigma_05_capability_dashboard.py` | pandas, matplotlib, seaborn, scipy, numpy |
| `Email Automation/` | Reporting | `generate_report.py`, `send_email.py`, `run_automation.py` | smtplib, email, SQLAlchemy |
| `PySpark Analysis/` | Distributed Processing | `spark_session.py`, `spark_extract.py`, `spark_transform.py`, `spark_analysis.py`, `spark_ml.py`, `run_spark_pipeline.py` | pyspark, pandas |

## Data Domain

**Countries:** Colombia, Brazil, Mexico, Argentina, Chile, Peru, Ecuador, USA, China, Germany  
**Indicators:** GDP (total), GDP per capita, Inflation, Unemployment rate, Population  
**Years:** 2000–2023  
**Source table:** `public.economic_indicators` (PostgreSQL)

**dbt schemas:**
- `staging` — cleaned, typed raw data
- `intermediate` — derived metrics (YoY growth, economic health score, LATAM flag)
- `marts` — country summaries, global trends, performance tiers

## Coding Conventions

- **Console output:** emoji-based status markers — `✅` success, `❌` error, `📡` API/network, `🔧` config, `🚀` pipeline start, `📊` data/stats, `💾` DB write
- **Docstrings:** short one-liner, then Parameters / Returns when non-obvious
- **Config:** centralised `config.py` dict, never hardcode credentials in logic files
- **Error handling:** try/except with graceful fallback (e.g. synthetic data when DB is unavailable); never swallow exceptions silently
- **SQL style:** explicit column list (no `SELECT *`), CTEs for readability, snake_case names, `ROUND(..., 2)` for metrics
- **dbt style:** staging → intermediate → marts separation; YAML tests for unique/not_null/accepted_values
- **PySpark style:** always stop SparkSession at the end; prefer DataFrame API over RDD; use `.cache()` before reused DataFrames; always call `.show()` and `.printSchema()` in exploration scripts

## Running the Pipeline

```bash
# 1. Extract & Load (requires DB)
cd "World bank" && python run_pipeline.py

# 2. dbt transform
cd world_bank_dbt && dbt run && dbt test

# 3. Refresh materialized views
psql -U <user> -d <db> -f "PostgreSQL Materialized Views/mv_01_country_profile.sql"

# 4. Statistical analysis
cd "6Sigma Seaborn Graphs" && python sigma_01_descriptive.py

# 5. Email report
cd "Email Automation" && python run_automation.py

# 6. PySpark distributed pipeline
cd "PySpark Analysis" && python run_spark_pipeline.py
```

## Database

PostgreSQL. Connection configured via `World bank/config.py` and mirrored in each module's `config.py`.  
Key table: `public.economic_indicators` — columns: `country`, `country_code`, `year`, `gdp`, `gdp_per_capita`, `inflation`, `unemployment_rate`, `population`.

The PySpark module connects via **JDBC** (requires `postgresql-42.x.x.jar` on the classpath) or falls back to synthetic data for development without a live DB.

## Branch Strategy

- `main` — stable production code  
- `claude/add-pyspark-module-hTZ4X` — PySpark module + this CLAUDE.md (in review)

## Key Design Decisions

- **Synthetic data fallback** in every analysis module so scripts run without a live DB — critical for CI and demos.
- **Materialized views** exist because dbt marts are views by default; MVs give BI tools sub-millisecond response.
- **6-Sigma methodology** applied to economic indicators to treat countries as "processes" — control charts flag anomalies (e.g. hyperinflation outliers).
- **PySpark** added as the distributed layer to demonstrate how the same transformations scale beyond single-node pandas.
