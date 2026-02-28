## 🧠 Project ETL Architecture & Data Flow

| Stage | Technology | Core Files / Directories | Description | Libraries Used |
| :--- | :--- | :--- | :--- | :--- |
| **1. Extraction & Load** | Python | `World bank/run_pipeline.py` | Extracts raw data via API/CSV, applies basic Pandas cleaning, and loads it into PostgreSQL. | `requests`, `pandas`, `SQLAlchemy`, `psycopg2-binary`, `schedule` |
| **2. Transformation** | dbt | `world_bank_dbt/models/` | Applies ELT principles using the Modern Data Stack. Transforms raw data through *Staging*, *Intermediate*, and *Marts* layers. | `dbt-core`, `dbt-postgres` |
| **3. Optimization** | PostgreSQL | `PostgreSQL Materialized Views/` | Creates physical snapshots of complex dbt queries to ensure millisecond load times for BI tools. | *N/A (Native SQL)* |
| **4. Analysis** | Python (Seaborn) | `6Sigma Seaborn Graphs/` | Consumes clean data for advanced statistical analysis (control charts, variability, capability). | `pandas`, `matplotlib`, `seaborn`, `numpy`, `scipy` |
| **5. Reporting** | Python (HTML) | `Email Automation/` | Queries the DB to generate automated HTML reports and distributes them to stakeholders. | `smtplib`, `email`, `base64`, `io`, `SQLAlchemy`, `datetime` |
