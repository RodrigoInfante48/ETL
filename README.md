## 🧠 Project ETL Architecture & Data Flow

| Stage | Technology | Core Files / Directories | Description |
| :--- | :--- | :--- | :--- |
| **1. Extraction & Load** | Python | `World bank/run_pipeline.py` | Extracts raw data via API/CSV, applies basic Pandas cleaning, and loads it into PostgreSQL. |
| **2. Transformation** | dbt | `world_bank_dbt/models/` | Applies ELT principles using the Modern Data Stack. Transforms raw data through *Staging*, *Intermediate*, and *Marts* layers. |
| **3. Optimization** | PostgreSQL | `PostgreSQL Materialized Views/` | Creates physical snapshots of complex dbt queries to ensure millisecond load times for BI tools. |
| **4. Analysis** | Python (Seaborn) | `6Sigma Seaborn Graphs/` | Consumes clean data for advanced statistical analysis (control charts, variability, capability). |
| **5. Reporting** | Python (HTML) | `Email Automation/` | Queries the DB to generate automated HTML reports and distributes them to stakeholders. |


**External libraries:**

* pandas
* requests
* SQLAlchemy
* psycopg2-binary
* matplotlib
* seaborn
* numpy
* scipy
* schedule
