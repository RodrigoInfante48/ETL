# ============================================
# run_pipeline.py - RUN THE FULL ETL PIPELINE
# ============================================
# This is the main file. Just run:
#   python run_pipeline.py
# And it will Extract, Transform, and Load everything.
#
# Optional dbt trigger
# ─────────────────────
# Set RUN_DBT=1 to kick off dbt build after a successful load.
# Set DBT_TARGET to override the target (default: prod).
#
#   RUN_DBT=1 python run_pipeline.py                      # local dbt build
#   RUN_DBT=1 DBT_CLOUD_JOB_ID=42 python run_pipeline.py # dbt Cloud API

import os
from datetime import datetime
from extract import extract_all
from transform import transform
from load import load


def run_pipeline():
    """
    Runs the complete ETL pipeline:
    1. EXTRACT → Pull data from World Bank API
    2. TRANSFORM → Clean and reshape the data
    3. LOAD → Store in PostgreSQL
    4. DBT    → Trigger dbt build (optional, RUN_DBT=1)
    """
    start_time = datetime.now()

    print("\n" + "🔥" * 25)
    print("   WORLD BANK ETL PIPELINE")
    print(f"   Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔥" * 25 + "\n")

    # ---- STEP 1: EXTRACT ----
    raw_data = extract_all()
    if raw_data.empty:
        print("❌ Pipeline failed at EXTRACT step")
        return

    # ---- STEP 2: TRANSFORM ----
    clean_data = transform(raw_data)
    if clean_data.empty:
        print("❌ Pipeline failed at TRANSFORM step")
        return

    # ---- STEP 3: LOAD ----
    success = load(clean_data)

    # ---- STEP 4: DBT (optional) ----
    dbt_success = True
    if success and os.environ.get("RUN_DBT") == "1":
        from trigger_dbt import trigger_dbt_cloud, trigger_dbt_local
        cloud_job_id = os.environ.get("DBT_CLOUD_JOB_ID")
        if cloud_job_id:
            dbt_success = trigger_dbt_cloud(job_id=int(cloud_job_id))
        else:
            dbt_success = trigger_dbt_local(target=os.environ.get("DBT_TARGET", "prod"))

    # ---- SUMMARY ----
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "🏁" * 25)
    if success and dbt_success:
        print("   ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("   ⚠️ PIPELINE COMPLETED WITH ERRORS")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Records processed: {len(clean_data)}")
    print(f"   Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🏁" * 25 + "\n")


if __name__ == "__main__":
    run_pipeline()
