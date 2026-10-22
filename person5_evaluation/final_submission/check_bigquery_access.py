"""Read-only authentication/table/query-permission preflight; never print secrets."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

from pandas_gbq.auth import get_credentials
from google.cloud import bigquery

HERE = Path(__file__).resolve().parent
report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
          "authentication": False, "tables": [], "query_dry_run": False}
try:
    print("Checking authorized cached/browser authentication...", flush=True)
    credentials, project = get_credentials(project_id=report["project"], reauth=False, auth_local_webserver=True)
    report["authentication"] = True
    client = bigquery.Client(project=project, credentials=credentials)
    for table_id in ["physionet-data.mimiciv_ed.edstays", "physionet-data.mimiciv_2_2_hosp.patients",
                     "physionet-data.mimiciv_v2_2_hosp.patients"]:
        try:
            table = client.get_table(table_id, timeout=20, retry=None)
            report["tables"].append({"table": table_id, "readable": True, "rows": table.num_rows})
            print(table_id, "READABLE", table.num_rows, flush=True)
        except Exception as error:
            report["tables"].append({"table": table_id, "readable": False, "error_type": type(error).__name__, "message": str(error)[:500]})
            print(table_id, type(error).__name__, str(error)[:500], flush=True)
        try:
            dry = client.query(f"SELECT subject_id FROM `{table_id}` LIMIT 1",
                job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False), timeout=20, retry=None)
            report["tables"][-1]["query_dry_run"] = True
            report["tables"][-1]["estimated_query_bytes"] = dry.total_bytes_processed
        except Exception as error:
            report["tables"][-1]["query_dry_run"] = False
            report["tables"][-1]["query_error_type"] = type(error).__name__
            report["tables"][-1]["query_error"] = str(error)[:500]
            print("Dry run:", type(error).__name__, str(error)[:500], flush=True)
    job = client.query("SELECT 1 AS connection_check", job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False),
                       timeout=20, retry=None)
    report["query_dry_run"] = True
    report["dry_run_bytes"] = job.total_bytes_processed
except Exception as error:
    report["error_type"], report["message"] = type(error).__name__, str(error)[:800]
    print(type(error).__name__, str(error)[:800], flush=True)
(HERE / "verification").mkdir(exist_ok=True)
(HERE / "verification/bigquery_access.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Authentication:", report["authentication"], "query permission:", report["query_dry_run"], flush=True)
