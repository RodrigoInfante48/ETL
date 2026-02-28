🌍 PROJECT: DATA PIPELINE (ETL + dbt + PostgreSQL + BI)
│
├── 1. EXTRACTION & LOADING (Basic ETL - Python)
│   📂 Folder: `World bank/`
│   ├── extract.py    --> Extracts raw data (e.g., World Bank API or CSVs).
│   ├── transform.py  --> Basic cleaning with Pandas (handling nulls, date formatting).
│   ├── load.py       --> Connects to PostgreSQL (using SQLAlchemy/psycopg2) and inserts data.
│   ├── run_pipeline.py-> Orchestrator: Runs Extract -> Transform -> Load sequentially.
│   └── config.py     --> Credentials and environment variables.
│
├── 2. DATA TRANSFORMATION & MODELING (dbt)
│   📂 Folder: `world_bank_dbt/`
│   │   (Applies analytics engineering within the PostgreSQL database)
│   ├── models/staging/      --> stg_economic_indicators.sql (Base views, naming standardization).
│   ├── models/intermediate/ --> int_country_metrics.sql (Business logic, joins, advanced cleaning).
│   └── models/marts/        --> mart_country_summary.sql / mart_global_trend.sql (Final tables ready for consumption).
│
├── 3. BI OPTIMIZATION (PostgreSQL)
│   📂 Folder: `PostgreSQL Materialized Views/`
│   │   (Materialized views for ultra-fast querying in Power BI/Tableau)
│   ├── mv_01_country_profile.sql
│   ├── mv_02_global_yearly_trend.sql
│   ├── mv_03_latest_year_ranking.sql
│   └── mv_04_outliers_detection.sql
│
└── 4. CONSUMPTION & USE CASES (Pipeline Outputs)
    ├── BI Tools (Power BI, Looker, Tableau) --> Connect to dbt Marts or Materialized Views.
    ├── 📂 `6Sigma Seaborn Graphs/`          --> Statistical analysis in Python (Control charts, variability).
    ├── 📂 `Email Automation/`               --> Automated HTML reports sent via email (generate_report.py -> send_email.py).
    └── 📄 `world_bank_analysis.ipynb`       --> Exploratory Jupyter notebook for Data Science / ad-hoc analysis.
