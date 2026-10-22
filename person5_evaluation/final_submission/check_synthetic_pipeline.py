"""Exercise every notebook code cell on invented raw table fixtures, without network.

This is NOT a clinical result or a successful real-data Run All. No existing data
files are read. BigQuery is replaced at its API boundary; all phenotype, split,
training, threshold, bootstrap and plotting code runs unchanged. Only Bootstrap
replicates and historical-count enforcement are changed for this engineering check.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import os
import sys
import time
import traceback

import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def make_raw_tables():
    rng = np.random.default_rng(271828)
    ed, patients, admissions, triage, labs, diagnoses = [], [], [], [], [], []
    event_id = 0
    for i in range(800):
        subject = 100000 + i
        age = int(rng.integers(20, 90))
        gender = rng.choice(["F", "M"])
        patients.append({"subject_id": subject, "gender": gender, "anchor_age": age,
            "anchor_year": 2150, "anchor_year_group": "2014 - 2016"})
        for visit in range(2):
            stay = 1000000 + 2 * i + visit
            hadm = 2000000 + 2 * i + visit
            arrival = pd.Timestamp("2150-01-01") + pd.Timedelta(days=40 * visit + i % 20)
            category = i % 10  # two severe, two stage 1, six negative patient strata
            stage = 2 if category < 2 else 1 if category < 4 else 0
            ed.append({"subject_id": subject, "hadm_id": hadm, "stay_id": stay,
                "intime": arrival, "outtime": arrival + pd.Timedelta(hours=6),
                "gender": gender, "race": "WHITE" if i % 3 else "BLACK/AFRICAN AMERICAN",
                "arrival_transport": "AMBULANCE" if rng.random() < 0.4 else "WALK IN"})
            admissions.append({"subject_id": subject, "hadm_id": hadm,
                "admittime": arrival + pd.Timedelta(hours=2), "dischtime": arrival + pd.Timedelta(hours=72)})
            vitals = [98.6 + rng.normal(), 85 + rng.normal(0, 15), 18 + rng.normal(0, 2),
                      min(100, 96 + rng.normal(0, 2)), 125 + rng.normal(0, 20), 75 + rng.normal(0, 12)]
            if rng.random() < 0.04:
                vitals = [np.nan] * 6
            elif rng.random() < 0.07:
                vitals[int(rng.integers(6))] = np.nan
            triage.append({"subject_id": subject, "stay_id": stay,
                **dict(zip(["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"], vitals)),
                "acuity": float(rng.integers(1, 6)),
                "pain": rng.choice(["0", "2", "5.0", "8", "N/A", None]),
                "chiefcomplaint": rng.choice(["abdominal pain", "weakness fatigue", "chest pain", "abnormal lab", "fever nausea", None])})
            # Some positives are first-sample positive; others rise on a later sample.
            first_positive = stage > 0 and i % 3 == 0
            times_values = [(-12, 1.0), (1, 2.2 if stage == 2 else 1.4), (8, 1.1)] if first_positive else [
                (1, 1.0), (8, 2.2 if stage == 2 else 1.4 if stage else 1.05)]
            # Fresh unresolved encounters for P2's selection-bias tables.
            if i % 47 == 0 and visit == 1:
                times_values = [(1, 1.0)]
            elif i % 53 == 0 and visit == 1:
                times_values = [(-12, 1.0)]
            for hours, value in times_values:
                event_id += 1
                labs.append({"labevent_id": event_id, "subject_id": subject, "hadm_id": hadm,
                    "specimen_id": event_id, "itemid": 50912, "charttime": arrival + pd.Timedelta(hours=hours),
                    "storetime": arrival + pd.Timedelta(hours=hours, minutes=10), "value": str(value),
                    "valuenum": value, "valueuom": "mg/dL"})
            diagnoses.append({"subject_id": subject, "hadm_id": hadm, "seq_num": 1,
                              "icd_code": "R509", "icd_version": 10})
    return {"edstays": pd.DataFrame(ed), "patients": pd.DataFrame(patients),
        "admissions": pd.DataFrame(admissions), "triage": pd.DataFrame(triage),
        "diagnoses_icd": pd.DataFrame(diagnoses), "labevents": pd.DataFrame(labs),
        "d_labitems": pd.DataFrame([{"itemid": item, "label": "Creatinine", "fluid": "Blood", "category": "Chemistry"}
                                   for item in [50912, 52546, 52024]])}


class FixtureClient:
    tables = None

    def __init__(self, *args, **kwargs):
        self.jobs = 0

    def get_table(self, table_id):
        return SimpleNamespace(modified="synthetic fixture", etag="synthetic", schema=[],
                               num_rows=len(self.tables[table_id.rsplit(".", 1)[-1]]))

    def query(self, query, job_config=None):
        if job_config.dry_run:
            return SimpleNamespace(total_bytes_processed=1024)
        name = next(name for name in self.tables if f".{name}`" in query)
        frame = self.tables[name].copy()
        if job_config.query_parameters:
            ids = job_config.query_parameters[0].values
            frame = frame.loc[frame.subject_id.isin(ids)].copy()
        self.jobs += 1
        result = SimpleNamespace(to_dataframe=lambda **kwargs: frame)
        return SimpleNamespace(result=lambda: result, job_id=f"synthetic_{self.jobs}",
                               total_bytes_processed=1024, total_bytes_billed=0, cache_hit=False)


def no_file_input(*args, **kwargs):
    raise AssertionError("Existing data/model input was attempted during the fresh pipeline check.")


def main():
    if Path(sys.prefix).name.lower() != "sml":
        raise RuntimeError("Use sml.")
    verification = HERE / "verification"
    verification.mkdir(exist_ok=True)
    fixture_root = verification / "synthetic_only"
    fixture_root.mkdir(exist_ok=True)
    os.chdir(fixture_root)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    source_path = HERE / "Final_AKI_End_to_End.ipynb"
    notebook = nbformat.read(source_path, as_version=4)
    namespace = {"__name__": "__main__"}
    FixtureClient.tables = make_raw_tables()
    timings, error_details = [], None
    import joblib
    # Network is unreachable by design: the only source API is replaced here.
    with patch("google.auth.default", return_value=(object(), "synthetic-project")), \
         patch("pandas_gbq.auth.get_credentials", return_value=(object(), "synthetic-project")), \
         patch("google.cloud.bigquery.Client", FixtureClient), \
         patch("pandas.read_csv", no_file_input), patch("pandas.read_parquet", no_file_input), \
         patch("numpy.load", no_file_input), patch.object(joblib, "load", no_file_input):
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            tag = cell.metadata.tags[0]
            start = time.monotonic()
            print("SYNTHETIC ONLY:", tag, flush=True)
            try:
                exec(compile(cell.source, f"notebook:{tag}", "exec"), namespace)
                if tag == "configuration":
                    namespace.update(BILLING_PROJECT="synthetic-project", REQUIRE_HISTORICAL_COUNTS=False,
                                     BOOTSTRAP_REPLICATES=20, N_JOBS=2, display=lambda *a, **k: None)
                timings.append({"tag": tag, "seconds": time.monotonic() - start, "passed": True})
            except Exception as error:
                traceback.print_exc()
                timings.append({"tag": tag, "seconds": time.monotonic() - start, "passed": False})
                error_details = {"tag": tag, "type": type(error).__name__, "message": str(error)}
                break
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "synthetic engineering check, NOT clinical results or a real-data Run All",
        "source_notebook_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "all_code_cells_passed": error_details is None, "cells_executed": len(timings),
        "bootstrap_replicates": 20, "all_original_lr_and_xgb_grid_candidates_used": True,
        "existing_csv_matrix_and_model_reads_blocked": True, "raw_source_api": "in-memory invented fixtures",
        "fixtures": {k: len(v) for k, v in FixtureClient.tables.items()}, "error": error_details, "timings": timings}
    if "RUN_DIR" in namespace:
        report["synthetic_output_directory"] = str((fixture_root / namespace["RUN_DIR"]).relative_to(HERE))
        namespace["write_json"](namespace["CHECKS"] / "run_status.json", {
            "status": "synthetic_check_passed" if not error_details else "synthetic_check_failed",
            "source": "invented fixtures; no clinical data", "real_data_run": False})
    (verification / "synthetic_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SYNTHETIC CHECK PASSED" if not error_details else "SYNTHETIC CHECK FAILED", flush=True)
    return int(error_details is not None)


if __name__ == "__main__":
    raise SystemExit(main())
