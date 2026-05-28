# ============================================================
# trigger_dbt.py — invoke dbt after the ETL load completes
#
# Two modes:
#   trigger_dbt_local()  — subprocess dbt build (local / CI / self-hosted)
#   trigger_dbt_cloud()  — dbt Cloud REST API (managed SaaS scheduler)
#
# Integration with run_pipeline.py
# ──────────────────────────────────
#   from trigger_dbt import trigger_dbt_local  # or trigger_dbt_cloud
#   ...
#   success = load(clean_data)
#   if success:
#       trigger_dbt_local(target="prod")
#
# dbt Cloud Scheduler vs dbt Cloud Orchestration
# ────────────────────────────────────────────────
# dbt Cloud Job (Scheduler)
#   A single dbt invocation (dbt build / dbt run / dbt test) with a cron
#   schedule OR triggered via API. Simplest option for "run after ETL".
#   Lives in dbt Cloud UI → Deploy → Jobs.
#   API endpoint: POST /api/v2/accounts/{id}/jobs/{job_id}/run/
#
# dbt Cloud Orchestration (dbt Cloud + dbt Mesh / dbt Explore)
#   Cross-project dependencies, DAG-aware scheduling across multiple dbt
#   projects. Requires dbt Cloud Enterprise. Use when you have multiple
#   dbt projects that share models or when you need fan-out/fan-in between
#   teams. NOT needed for a single-project pipeline like this one.
#
# manifest.json and Slim CI
# ──────────────────────────
# dbt writes target/manifest.json after every run. It contains:
#   {
#     "metadata": { "dbt_schema_version": "...", "generated_at": "..." },
#     "nodes": {
#       "model.world_bank_dbt.stg_economic_indicators": {
#         "unique_id": "model.world_bank_dbt.stg_economic_indicators",
#         "fqn": ["world_bank_dbt", "staging", "stg_economic_indicators"],
#         "original_file_path": "models/staging/stg_economic_indicators.sql",
#         "checksum": { "name": "sha256", "checksum": "<hash-of-sql-file>" },
#         "depends_on": { "nodes": ["source.world_bank_dbt.world_bank.economic_indicators"] },
#         "config": { "materialized": "view", "schema": "staging" },
#         ...
#       },
#       ...
#     },
#     "sources": { ... },
#     "exposures": { ... },
#     "metrics": { ... }
#   }
#
# Slim CI uses the prod manifest as a baseline:
#   dbt build --select state:modified+ --state path/to/prod-manifest/
#
# dbt compares each node's checksum against the baseline. A node is
# "modified" if its SQL, config, or any upstream dependency changed.
# The `+` suffix means "and all descendants of modified nodes".
# This cuts CI time from O(all models) to O(changed models).
# ============================================================

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


DBT_PROJECT_DIR = Path(__file__).parent.parent / "world_bank_dbt"


def trigger_dbt_local(target: str = "dev", select: str = None) -> bool:
    """
    Run `dbt build` as a subprocess (local dev, CI runners, self-hosted).

    Parameters
    ----------
    target : dbt target name — dev | ci | prod
    select : dbt node selector; None runs the full project graph
    """
    cmd = [
        "dbt", "build",
        f"--target={target}",
        "--profiles-dir=.",
    ]
    if select:
        cmd += [f"--select={select}"]

    print(f"\n🔧 Triggering dbt build")
    print(f"   target : {target}")
    print(f"   select : {select or '(all models)'}")
    print(f"   dir    : {DBT_PROJECT_DIR}")

    result = subprocess.run(cmd, cwd=DBT_PROJECT_DIR)

    if result.returncode == 0:
        print("✅ dbt build completed successfully")
        return True

    print(f"❌ dbt build failed (exit code {result.returncode})")
    return False


def trigger_dbt_cloud(
    job_id: int,
    cause: str = "Triggered by World Bank ETL pipeline",
    wait: bool = True,
    poll_interval: int = 15,
) -> bool:
    """
    Trigger a dbt Cloud job via the REST API and optionally poll until done.

    Setup (one-time)
    ─────────────────
    1. dbt Cloud UI → Account settings → Service tokens → New token (Job Admin)
    2. Set env vars:
         DBT_CLOUD_API_TOKEN  = <your service token>
         DBT_CLOUD_ACCOUNT_ID = <numeric account ID from the URL>
    3. Find your job_id:
         dbt Cloud UI → Deploy → Jobs → click job → copy ID from URL

    dbt Cloud API — dependency ordering with the ETL
    ─────────────────────────────────────────────────
    dbt Cloud jobs have no native "wait for external step" trigger, so the
    cleanest pattern is to call this function at the end of run_pipeline.py:

        if success:
            trigger_dbt_cloud(job_id=int(os.environ["DBT_CLOUD_JOB_ID"]))

    For a daily schedule: set the dbt Cloud job's cron to fire 15–30 min
    after the ETL's scheduled start. Use the API trigger when you need
    guaranteed ordering (ETL success → dbt run, not time-based).

    Parameters
    ----------
    job_id        : dbt Cloud job ID (integer)
    cause         : human-readable label shown in the dbt Cloud run history
    wait          : if True, poll until the run reaches a terminal state
    poll_interval : seconds between status polls (default 15)
    """
    api_token = os.environ.get("DBT_CLOUD_API_TOKEN")
    account_id = os.environ.get("DBT_CLOUD_ACCOUNT_ID")

    if not api_token or not account_id:
        print("❌ DBT_CLOUD_API_TOKEN or DBT_CLOUD_ACCOUNT_ID env var not set")
        return False

    base_url = f"https://cloud.getdbt.com/api/v2/accounts/{account_id}"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }

    # ── 1. Trigger the run ────────────────────────────────────────────────────
    payload = json.dumps({"cause": cause}).encode()
    req = urllib.request.Request(
        f"{base_url}/jobs/{job_id}/run/",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            run_data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"❌ HTTP {exc.code} triggering dbt Cloud job {job_id}: {exc.reason}")
        return False
    except Exception as exc:
        print(f"❌ Failed to trigger dbt Cloud job {job_id}: {exc}")
        return False

    run_id = run_data["data"]["id"]
    run_url = f"https://cloud.getdbt.com/runs/{run_id}"
    print(f"🚀 dbt Cloud job triggered — run #{run_id}")
    print(f"   {run_url}")

    if not wait:
        return True

    # ── 2. Poll until terminal state ──────────────────────────────────────────
    # Status codes: 1=Queued, 2=Starting, 3=Running, 10=Success, 20=Error, 30=Cancelled
    terminal = {10: "✅ Success", 20: "❌ Error", 30: "⚠️ Cancelled"}

    print("📡 Polling for completion", end="", flush=True)
    while True:
        time.sleep(poll_interval)
        print(".", end="", flush=True)

        status_req = urllib.request.Request(
            f"{base_url}/runs/{run_id}/", headers=headers
        )
        try:
            with urllib.request.urlopen(status_req, timeout=30) as resp:
                status_data = json.loads(resp.read())
        except Exception as exc:
            print(f"\n⚠️  Polling error (will retry): {exc}")
            continue

        status_code = status_data["data"]["status"]
        if status_code in terminal:
            print(f"\n{terminal[status_code]} — dbt Cloud run #{run_id}")
            return status_code == 10
