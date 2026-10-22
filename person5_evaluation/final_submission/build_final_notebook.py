"""Build the self-contained submission; teammate files are source references only.

Run with sml from any working directory. The resulting notebook never imports
these source files and never reads an existing cohort, matrix or saved model.
"""
from pathlib import Path
import ast
import hashlib
import json
import textwrap

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "p1": ROOT / "person1_handoff/ML_for_health_Group_Project.ipynb",
    "p2qa": ROOT / "person2_handoff/person2_phenotype_review/P2_phenotype_review.ipynb",
    "p2pre": ROOT / "person2_handoff/person2_split_preprocessing/aki_preprocessing.py",
    "p2split": ROOT / "person2_handoff/person2_split_preprocessing/P2_split_preprocessing.ipynb",
    "p3": ROOT / "person3_handoff/primary/person3_baselines.py",
    "p3secondary": ROOT / "person3_handoff/secondary/stage2_baselines.py",
    "p4": ROOT / "person4_handoff/legacy_primary_training/p4_xgboost.py",
    "p4secondary": ROOT / "person4_handoff/person4_handoff/outputs/stage2_model_settings.json",
    "p5": ROOT / "person5_evaluation/current/metrics.py",
}
notebooks = {k: nbf.read(p, as_version=4) for k, p in SOURCES.items() if p.suffix == ".ipynb"}
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source, tag):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(), metadata={"tags": [tag]}))


def src(key, index):
    return notebooks[key].cells[index].source.strip()


def definitions(source):
    tree = ast.parse(source)
    return "\n\n".join(ast.get_source_segment(source, n) for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef)))


def quiet_source(source):
    """Remove display-only expressions, retaining P1's original computations."""
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, ast.Expr)]
    return ast.unparse(tree)


md("""
# AKI at ED admission — integrated final submission

**Primary:** observed creatinine AKI Stage ≥1. **Secondary:** Stage ≥2 in the same labelled population.

This notebook extracts raw MIMIC data from Google BigQuery on each run, constructs the cohort and labels,
fits preprocessing and 14 models from scratch, and evaluates both outcomes. It has **no local input-data,
pretrained-model, or teammate-script dependency**. Existing CSV/NPY/NPZ/joblib files are never read.
All analysis code is embedded below. New outputs use a unique directory relative to the working directory.

| Contributor | Integrated work |
|---|---|
| Person 1 | BigQuery extraction, eligible cohort, Decimal-based creatinine phenotype, predictor cleaning |
| Person 2 | Independent phenotype QA, descriptive tables, patient split (seed 90089), train-only preprocessing |
| Person 3 | Dummy, age/gender LR, structured LR, structured + chief complaint LR for both outcomes |
| Person 4 | XGBoost search and explicit sparse structural-zero handling; P5 extends the documented training recipe to Stage ≥2 |
| Person 5 | Integration, validation-only thresholds, common test metrics, paired patient Bootstrap, sensitivity analyses and plots |

**Run:** select the existing `sml` Python kernel, set `BILLING_PROJECT` below (or `GOOGLE_CLOUD_PROJECT`),
and use Run All. The Google account supplying cached pandas-gbq credentials or ADC must have permission to
access PhysioNet MIMIC and run BigQuery jobs in that project. Credentials stay outside the notebook.
The notebook reuses pandas-gbq's cached/browser authentication, as used in the team's local environment.
ADC is also supported. No Colab-only authentication is used. Internet access and authorized BigQuery billing are necessary.

The historical test results were already known when this integration was written. This is a reproducibility
rerun of the team's protocol, **not a new external or previously unseen validation study**. Within this run,
test predictions are generated only after the models and validation thresholds are frozen.
""")

code('''
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.metrics import precision_recall_curve, roc_curve
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from google.cloud import bigquery
import google.auth
import pandas_gbq.auth

BILLING_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()  # set your own authorized billing project
ED = "physionet-data.mimiciv_ed"
HOSP = "physionet-data.mimiciv_2_2_hosp"
SPLIT_SEED, MODEL_SEED, BOOTSTRAP_SEED = 90089, 42, 9008905
SENSITIVITY_TARGET = 0.80
BOOTSTRAP_REPLICATES = 1000
N_JOBS = min(8, os.cpu_count() or 1)
# Bytes are a scan limit, not a currency estimate. Each query is dry-run before execution.
MAX_QUERY_BYTES = 100 * 1024**3
MAX_TOTAL_QUERY_BYTES = 200 * 1024**3
REQUIRE_HISTORICAL_COUNTS = True
OUTPUT_BASE = Path("final_submission_runs")
assert not OUTPUT_BASE.is_absolute(), "Use a relative output directory."
assert Path(sys.prefix).name.lower() == "sml", "Select the sml Python environment."

def utc_now():
    return datetime.now(timezone.utc).isoformat()

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
RUN_DIR = OUTPUT_BASE / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=False)
TABLES, FIGURES, CHECKS, PRIVATE = [RUN_DIR / p for p in ("tables", "figures", "checks", "private")]
for directory in (TABLES, FIGURES, CHECKS, PRIVATE):
    directory.mkdir()
OUT_DIR = TABLES  # P2's newly generated tables

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def save_csv(frame, filename, directory=TABLES):
    frame.to_csv(directory / filename, index=False, encoding="utf-8-sig")

ENVIRONMENT = {name: importlib.metadata.version(name) for name in (
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "google-cloud-bigquery", "pandas-gbq")}
ENVIRONMENT.update(python=sys.version.split()[0], environment=Path(sys.prefix).name)
write_json(CHECKS / "environment.json", ENVIRONMENT)
write_json(CHECKS / "run_status.json", {"status": "started", "started_at_utc": utc_now(),
    "raw_source": "Google BigQuery", "reads_existing_data_or_models": False})
np.random.seed(SPLIT_SEED)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10})
print("Run directory:", RUN_DIR)
print("Environment:", ENVIRONMENT)
''', "configuration")

md("""
## Shared definitions and checks before data access

P2 preprocessing is fitted on training patients only. Numeric missingness indicators are used for LR;
XGBoost retains numeric NaN. Clinical text uses training-only TF-IDF unigrams/bigrams (`min_df=5`).
Feature dimensions are derived from the fresh training data rather than hard-coded to the previous handoff.
""")
code(SOURCES["p2pre"].read_text(encoding="utf-8"), "preprocessing_definitions")
code(SOURCES["p5"].read_text(encoding="utf-8"), "metric_definitions")
code(definitions(src("p2qa", 11)) + '\n\npredictor_columns = PREDICTORS.copy()', "cleaning_definition")

outcome_indices = [27, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 53, 55, 57, 59, 62, 64, 66]
outcome_source = "\n\n".join(quiet_source(src("p1", i)) for i in outcome_indices)
code('def derive_phenotype(cohort, creatinine_raw):\n'
     '    """P1 cells 27–66: unchanged computation, in-memory inputs; displays removed."""\n'
     '    cohort = cohort.copy()\n' + textwrap.indent(outcome_source, "    ")
     + '\n    return cohort, sample_results\n', "phenotype_definition")
code(src("p2qa", 13), "independent_phenotype_definition")
synthetic = src("p2qa", 14)
synthetic += '''
assert tests["pass"].all(), "P2 synthetic boundary tests failed."
p1_synthetic, _ = derive_phenotype(enc_df.drop(columns="case"), lab_df)
p1_check = p1_synthetic.set_index("stay_id").sort_index()
p2_check = result.set_index("stay_id").sort_index()
for column in ("label_state", "current_sample_count", "first_sample_status"):
    assert p1_check[column].equals(p2_check[column]), column
assert p1_check["highest_creatinine_stage"].equals(p2_check["highest_stage"])
assert np.array_equal(p1_check["observed_stage2_positive"].astype(int), p2_check["stage2_label"])
write_json(CHECKS / "phenotype_boundary_tests.json", {"cases": 24, "encounter_assertions": len(tests),
    "p2_reference_passed": True, "production_p1_matches_independent_p2": True})
print("P1 and independent P2 agree on all 26 synthetic encounters (24 boundary cases).")
'''
code(synthetic, "phenotype_boundary_tests")

md("""
## Part 1 — fresh extraction and cohort construction (Person 1)

Only the following raw BigQuery tables are read: ED `edstays`, `triage`; HOSP `patients`, `admissions`,
`diagnoses_icd`, `d_labitems`, `labevents`. A successful run logs table metadata, SQL hashes, scan sizes
and query job IDs. Query-result caching is disabled. An inaccessible source stops the run; there is no
fallback to existing files. `triage` is the source-table name; the study prediction anchor is **at ED admission**.
""")

code('''
if not BILLING_PROJECT:
    write_json(CHECKS / "run_status.json", {"status": "blocked_before_extraction",
        "reason": "BILLING_PROJECT is empty; authorized BigQuery project and ADC are required.",
        "updated_at_utc": utc_now()})
    raise RuntimeError("Set BILLING_PROJECT or GOOGLE_CLOUD_PROJECT and configure authorized Google ADC; no existing CSV will be used.")
# Same cached/browser OAuth mechanism as pandas_gbq.read_gbq. No credential contents are displayed.
credentials, _ = pandas_gbq.auth.get_credentials(
    project_id=BILLING_PROJECT, reauth=True, auth_local_webserver=True
)
client = bigquery.Client(project=BILLING_PROJECT, credentials=credentials)
query_audit, table_audit = [], []
raw_tables = [f"{ED}.{name}" for name in ("edstays", "triage")] + [
    f"{HOSP}.{name}" for name in ("patients", "admissions", "diagnoses_icd", "d_labitems", "labevents")]
for table_id in raw_tables:
    table = client.get_table(table_id)
    table_audit.append({"table": table_id, "modified": str(table.modified), "etag": table.etag,
        "num_rows": table.num_rows, "schema": [(f.name, f.field_type, f.mode) for f in table.schema]})
write_json(CHECKS / "source_tables.json", table_audit)

def query_dataframe(query, query_parameters=None):
    parameters = query_parameters or []
    dry_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=parameters)
    dry_job = client.query(query, job_config=dry_config)
    estimated = dry_job.total_bytes_processed
    if estimated is None:
        raise RuntimeError("BigQuery returned no scan estimate; query not executed.")
    estimated = int(estimated)
    if estimated > MAX_QUERY_BYTES or sum(q["estimated_bytes"] for q in query_audit) + estimated > MAX_TOTAL_QUERY_BYTES:
        raise RuntimeError(f"Scan limit exceeded ({estimated / 1024**3:.2f} GiB for next query); review limits before rerunning.")
    config = bigquery.QueryJobConfig(use_query_cache=False, query_parameters=parameters,
                                    maximum_bytes_billed=MAX_QUERY_BYTES)
    job = client.query(query, job_config=config)
    frame = job.result().to_dataframe(create_bqstorage_client=False, int_dtype=None, bool_dtype=None)
    query_audit.append({"query_number": len(query_audit) + 1, "sql": query,
        "sql_sha256": hashlib.sha256(query.encode()).hexdigest(), "job_id": job.job_id,
        "estimated_bytes": estimated, "processed_bytes": job.total_bytes_processed,
        "billed_bytes": job.total_bytes_billed, "cache_hit": job.cache_hit,
        "rows_downloaded": len(frame), "completed_at_utc": utc_now(),
        "array_parameter_lengths": {p.name: len(p.values) for p in parameters}})
    write_json(CHECKS / "query_audit.json", query_audit)
    print(f"Raw query {len(query_audit)}: {len(frame):,} rows; scan {estimated / 1024**3:.3f} GiB")
    return frame

def patient_parameter(ids):
    return [bigquery.ArrayQueryParameter("patient_ids", "INT64", [int(i) for i in ids])]
''', "bigquery_access")

code('''
edstays_raw = query_dataframe(f"SELECT * FROM `{ED}.edstays`")
ed_patient_ids = edstays_raw["subject_id"].dropna().unique().tolist()
patients_raw = query_dataframe(f"SELECT * FROM `{HOSP}.patients` WHERE subject_id IN UNNEST(@patient_ids)", patient_parameter(ed_patient_ids))
admissions_raw = query_dataframe(f"SELECT * FROM `{HOSP}.admissions` WHERE subject_id IN UNNEST(@patient_ids)", patient_parameter(ed_patient_ids))
triage_raw = query_dataframe(f"SELECT * FROM `{ED}.triage`")
assert edstays_raw.stay_id.is_unique and patients_raw.subject_id.is_unique
assert admissions_raw.hadm_id.is_unique and triage_raw.stay_id.is_unique
flow = [{"step": "All ED encounters", "encounters": len(edstays_raw), "patients": edstays_raw.subject_id.nunique()}]
''', "raw_core_extraction")
code(src("p1", 9), "numeric_source_fields")
for index, tag, label in [(13, "admitted_cohort", "Hospital-admitted ED encounters"),
                          (15, "adult_cohort", "Adults"), (17, "valid_time_cohort", "Valid encounter times")]:
    code(src("p1", index) + f'\nflow.append({{"step": {label!r}, "encounters": len(cohort), "patients": cohort.subject_id.nunique()}})', tag)
code('''
diagnoses_raw = query_dataframe(f"""SELECT subject_id, hadm_id, seq_num, icd_code, icd_version
FROM `{HOSP}.diagnoses_icd` WHERE subject_id IN UNNEST(@patient_ids)""", patient_parameter(cohort.subject_id.unique()))
''', "raw_history_extraction")
code("\n\n".join(src("p1", i) for i in [21, 23, 25]) + '''
flow.append({"step": "After prior ESKD/dialysis exclusions", "encounters": len(cohort), "patients": cohort.subject_id.nunique()})
''', "renal_exclusions")
code('''
lab_items = query_dataframe(f"SELECT * FROM `{HOSP}.d_labitems`")
display(lab_items.loc[lab_items.itemid.isin([50912, 52546, 52024]), ["itemid", "label", "fluid", "category"]])
creatinine_raw = query_dataframe(f"""SELECT labevent_id, subject_id, hadm_id, specimen_id,
itemid, charttime, storetime, value, valuenum, valueuom
FROM `{HOSP}.labevents` WHERE subject_id IN UNNEST(@patient_ids)
AND itemid IN (50912, 52546, 52024)""", patient_parameter(cohort.subject_id.unique()))
cohort, sample_results = derive_phenotype(cohort, creatinine_raw)
assert cohort.stay_id.is_unique
labelled = cohort.loc[cohort.aki_label.notna()].copy()
flow.append({"step": "Labelled modelling population", "encounters": len(labelled), "patients": labelled.subject_id.nunique()})
save_csv(pd.DataFrame(flow), "cohort_flow.csv")
save_csv(cohort.label_state.value_counts().rename_axis("label_state").reset_index(name="encounters"), "outcome_counts.csv")
display(pd.DataFrame(flow))
''', "raw_labs_and_phenotype")
code('''
model_data = labelled.copy()
vitals = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
''' + "\n\n".join(src("p1", i) for i in [73, 75]) + '''
modelling = model_data[["subject_id", "stay_id", "hadm_id"] + PREDICTORS + ["aki_label"]].copy()
modelling["aki_label"] = modelling["aki_label"].astype(int)
modelling["aki_stage2_label"] = model_data["observed_stage2_positive"].astype(int)
# Normalize pandas extension dtypes for sklearn; preserve literal text such as "N/A".
for column in NUMERIC:
    modelling[column] = pd.to_numeric(modelling[column], errors="coerce").astype(float)
for column in CATEGORICAL + [TEXT]:
    modelling[column] = modelling[column].astype(object).where(modelling[column].notna(), np.nan)
audit = cohort.copy()
raw_labelled = labelled.copy()
observed_counts = {"eligible_encounters": len(cohort), "modelling_encounters": len(modelling),
    "modelling_patients": modelling.subject_id.nunique(), "primary_positive": int(modelling.aki_label.sum()),
    "secondary_positive": int(modelling.aki_stage2_label.sum())}
expected_counts = {"eligible_encounters": 195340, "modelling_encounters": 113097,
    "modelling_patients": 66090, "primary_positive": 7866, "secondary_positive": 2254}
write_json(CHECKS / "historical_count_comparison.json", {"observed": observed_counts,
    "historical_reference": expected_counts, "matches": observed_counts == expected_counts})
if REQUIRE_HISTORICAL_COUNTS:
    assert observed_counts == expected_counts, "Fresh source cohort differs from historical counts; inspect source versions and query audit before proceeding."
assert (modelling.aki_stage2_label <= modelling.aki_label).all()
print(observed_counts)
''', "predictor_cleaning")

md("""
## Part 2 — phenotype review, descriptive statistics, patient split and preprocessing (Person 2)

P1's production phenotype is checked against P2's independent boundary-case implementation. The fresh
eligible cohort also undergoes the original row-level timing, label and severity checks. Literal text is
preserved directly from BigQuery; no CSV parsing can silently convert a recorded `N/A` pain entry to missing.
The historical cohort counts are a drift check, not fabricated output. A mismatch stops the default run.
""")
code(src("p2qa", 7) + '\nassert rule_checks["violating_encounters"].eq(0).all()', "phenotype_row_qa")
cleancheck = src("p2qa", 11)
cleancheck = cleancheck[cleancheck.index("recleaned ="):cleancheck.index("# Raw values in the audit")]
code(cleancheck + '\nassert cleaning_check["rows_disagreeing"].eq(0).all()', "independent_cleaning_qa")
for index in [19, 23, 25, 28, 30]:
    source = src("p2qa", index)
    if index == 25:
        source = 'audit_raw_full = audit.copy()\n' + source[source.index('cleaned_full ='):]
        source = source.replace('print("Cleaned audit predictors', 'assert bad == 0\nprint("Cleaned audit predictors')
    code(source, f"p2_table_{index}")
code(src("p2qa", 32), "sensitivity_definitions")

code('''
df = modelling.copy()
patients = df.groupby("subject_id").agg(encounters=("stay_id", "size"),
    any_aki=("aki_label", "max"), any_stage2=("aki_stage2_label", "max")).reset_index()
patients["stratum"] = np.select([patients.any_stage2.eq(1), patients.any_aki.eq(1)],
    ["stage2", "stage1_only"], "none")
train_p, hold_p = train_test_split(patients, test_size=0.30, stratify=patients.stratum, random_state=SPLIT_SEED)
val_p, test_p = train_test_split(hold_p, test_size=0.50, stratify=hold_p.stratum, random_state=SPLIT_SEED)
partition_of = pd.concat([pd.Series(name, index=group.subject_id) for name, group in
    [("train", train_p), ("validation", val_p), ("test", test_p)]])
df["partition"] = df.subject_id.map(partition_of)
assert df.partition.notna().all() and df.stay_id.is_unique
assert df.groupby("subject_id").partition.nunique().eq(1).all()
data = {part: df.loc[df.partition.eq(part)].sort_values("stay_id").reset_index(drop=True)
        for part in ("train", "validation", "test")}
split_summary = pd.DataFrame([{"partition": part, "encounters": len(d), "patients": d.subject_id.nunique(),
    "primary_positive": int(d.aki_label.sum()), "secondary_positive": int(d.aki_stage2_label.sum())}
    for part, d in data.items()])
save_csv(split_summary, "split_summary.csv")
save_csv(df[["subject_id", "stay_id", "partition"]], "split_manifest.csv", PRIVATE)
display(split_summary)
X = {part: d[PREDICTORS].copy() for part, d in data.items()}
assert set(PREDICTORS).isdisjoint({"subject_id", "stay_id", "hadm_id", "aki_label", "aki_stage2_label"})
fitted, feature_names, matrices = {}, {}, {}
with threadpool_limits(limits=1):
    for name, builder in PREPROCESSORS.items():
        pre = builder().fit(X["train"])
        fitted[name] = pre
        feature_names[name] = list(pre.get_feature_names_out())
        # Test transformation is deferred until model/threshold freezing.
        matrices[name] = {part: pre.transform(X[part]) for part in ("train", "validation")}
        print(name, {part: matrix.shape for part, matrix in matrices[name].items()})
imputer = fitted["lr_structured"].named_transformers_["num"].named_steps["impute"]
assert np.allclose(imputer.statistics_, X["train"][NUMERIC].median().to_numpy())
assert np.allclose(matrices["lr_structured"]["train"][:, :len(NUMERIC)].mean(axis=0), 0, atol=1e-8)
for name in ("lr_structured", "lr_text"):
    for matrix in matrices[name].values():
        assert np.isfinite(matrix.data if sp.issparse(matrix) else matrix).all()
assert fitted["lr_text"].named_transformers_["text"].named_steps["tfidf"].vocabulary_ == fitted["tree_text"].named_transformers_["text"].named_steps["tfidf"].vocabulary_
write_json(CHECKS / "preprocessing.json", {"fit_partition": "train", "split_seed": SPLIT_SEED,
    "features": feature_names, "test_transformed": False, "imputation_medians": imputer.statistics_.tolist()})
''', "patient_split_preprocessing")

md("""
## Part 3 — four baselines per outcome (Person 3)

The dummy predicts the training prevalence. Age/gender LR is unpenalized. Structured and text-inclusive
LR select `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` by validation AP (ties within 1e-12 favour smaller C).
The text model separates the gain from adding text from the gain from changing the algorithm.
All selected models remain fitted on training patients only; convergence warnings stop the run.
""")
code('''
TARGETS = ["aki_label", "aki_stage2_label"]
MODEL_NAMES = ["dummy", "age_gender_lr", "structured_lr", "structured_text_lr",
    "age_gender_xgboost", "structured_xgboost", "structured_text_xgboost"]
KEYS = [(target, model) for target in TARGETS for model in MODEL_NAMES]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
models, input_names, selected_parameters, tuning_rows = {}, {}, {}, []
lr_age_indices = [feature_names["lr_structured"].index(name) for name in ["num__age", "cat__gender_M"]]

def fit_lr(matrix, y, C=1.0, penalty="l2", max_iter=2000):
    model = LogisticRegression(C=C, penalty=penalty, solver="lbfgs", tol=1e-8,
        max_iter=max_iter, class_weight=None, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with threadpool_limits(limits=1):
            model.fit(matrix, y)
    return model

for target in TARGETS:
    ytr, yva = data["train"][target].to_numpy(), data["validation"][target].to_numpy()
    models[target, "dummy"] = DummyClassifier(strategy="prior").fit(matrices["lr_structured"]["train"], ytr)
    input_names[target, "dummy"] = "lr_structured"
    selected_parameters[target, "dummy"] = {"strategy": "training_prevalence"}
    models[target, "age_gender_lr"] = fit_lr(matrices["lr_structured"]["train"][:, lr_age_indices], ytr, penalty=None)
    input_names[target, "age_gender_lr"] = "lr_age"
    selected_parameters[target, "age_gender_lr"] = {"penalty": None, "features": ["num__age", "cat__gender_M"]}
    for name, matrix_name in [("structured_lr", "lr_structured"), ("structured_text_lr", "lr_text")]:
        best_ap, best_model, best_c = -np.inf, None, None
        for C in C_GRID:
            candidate = fit_lr(matrices[matrix_name]["train"], ytr, C=C, max_iter=20000 if matrix_name == "lr_text" else 2000)
            ap_value = average_precision_score(yva, candidate.predict_proba(matrices[matrix_name]["validation"])[:, 1])
            tuning_rows.append({"target": target, "model": name, "C": C, "validation_ap": ap_value})
            if ap_value > best_ap + 1e-12:
                best_ap, best_model, best_c = ap_value, candidate, C
            print(f"{target}/{name}: C={C:g}, validation AP={ap_value:.6f}", flush=True)
        models[target, name] = best_model
        input_names[target, name] = matrix_name
        selected_parameters[target, name] = {"C": best_c, "validation_ap": best_ap}
''', "lr_training")

md("""
## Part 4 — three XGBoost models per outcome (Person 4, integrated by Person 5)

For each outcome, structured and text-inclusive models independently select from P4's nine candidates
using validation AP. Age/gender XGBoost uses the selected structured model's hyperparameters. Other
settings are P4's `hist`, `logloss`, seed 42, subsample 0.8 and colsample_bytree 0.8. Exact AP ties retain
the first candidate. The Stage ≥2 extension follows the delivered P4 settings and documented search recipe.

**Sparse zero correction:** the complete dense structured block is stored explicitly in CSR, including
zero and NaN entries, before adding the TF-IDF block. We do not call `eliminate_zeros`. The text block
is selected by feature names; previously reported 19/2,772/2,791 dimensions are not assumed.
""")
code('''
def dense_to_explicit_csr(matrix):
    matrix = np.asarray(matrix)
    rows, columns = matrix.shape
    return sp.csr_matrix((matrix.ravel(), np.tile(np.arange(columns, dtype=np.int32), rows),
        np.arange(0, (rows + 1) * columns, columns, dtype=np.int64)), shape=matrix.shape)

tree_age_indices = [feature_names["tree_structured"].index(name) for name in ["num__age", "cat__gender_M"]]
tree_text_indices = [i for i, name in enumerate(feature_names["tree_text"]) if name.startswith("text__")]
combined_tree_names = feature_names["tree_structured"] + [feature_names["tree_text"][i] for i in tree_text_indices]

def construct_model_inputs(part):
    values = {name: parts[part] for name, parts in matrices.items()}
    values["lr_age"] = values["lr_structured"][:, lr_age_indices]
    values["tree_age"] = values["tree_structured"][:, tree_age_indices]
    dense = np.asarray(values["tree_structured"])
    explicit = dense_to_explicit_csr(dense)
    assert explicit.nnz == dense.size
    assert np.array_equal(explicit.data, dense.ravel(), equal_nan=True)
    assert np.array_equal(explicit.toarray(), dense, equal_nan=True)
    text = sp.csr_matrix(values["tree_text"][:, tree_text_indices])
    values["tree_explicit_text"] = sp.hstack([explicit, text], format="csr")
    assert values["tree_explicit_text"].nnz == dense.size + text.nnz
    assert values["tree_explicit_text"].shape[1] == len(combined_tree_names)
    return values

model_inputs = {part: construct_model_inputs(part) for part in ("train", "validation")}
XGB_GRID = [{"n_estimators": n, "max_depth": depth, "learning_rate": rate,
             "subsample": 0.8, "colsample_bytree": 0.8}
    for depth in (2, 3, 4) for n, rate in ([(200, 0.05), (300, 0.05), (400, 0.03)]
        if depth != 4 else [(200, 0.05), (300, 0.03), (400, 0.03)])]

def fit_xgb(matrix, y, parameters):
    classifier = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=MODEL_SEED, n_jobs=N_JOBS, **parameters)
    return classifier.fit(matrix, y)

for target in TARGETS:
    ytr, yva = data["train"][target].to_numpy(), data["validation"][target].to_numpy()
    for name, matrix_name in [("structured_xgboost", "tree_structured"), ("structured_text_xgboost", "tree_explicit_text")]:
        best_ap, best_model, best_parameters = -np.inf, None, None
        for index, parameters in enumerate(XGB_GRID):
            candidate = fit_xgb(model_inputs["train"][matrix_name], ytr, parameters)
            ap_value = average_precision_score(yva, candidate.predict_proba(model_inputs["validation"][matrix_name])[:, 1])
            tuning_rows.append({"target": target, "model": name, "candidate": index, **parameters, "validation_ap": ap_value})
            if ap_value > best_ap:
                best_ap, best_model, best_parameters = ap_value, candidate, parameters.copy()
            print(f"{target}/{name}: candidate {index + 1}/{len(XGB_GRID)}, validation AP={ap_value:.6f}", flush=True)
        models[target, name] = best_model
        input_names[target, name] = matrix_name
        selected_parameters[target, name] = {**best_parameters, "validation_ap": best_ap}
    parameters = {key: value for key, value in selected_parameters[target, "structured_xgboost"].items() if key != "validation_ap"}
    models[target, "age_gender_xgboost"] = fit_xgb(model_inputs["train"]["tree_age"], ytr, parameters)
    input_names[target, "age_gender_xgboost"] = "tree_age"
    selected_parameters[target, "age_gender_xgboost"] = parameters
assert set(models) == set(KEYS)
save_csv(pd.DataFrame(tuning_rows), "validation_model_search.csv")
''', "xgboost_training")

md("""
## Part 5 — validation freeze, test evaluation and sensitivity analyses (Person 5)

AP uses the step-wise average-precision definition, not a trapezoidal PR area. AUROC and Brier are
secondary metrics. Each model/outcome has its own largest validation threshold with Sensitivity ≥0.80;
predictions equal to the threshold are positive. Test data never select hyperparameters or thresholds.

Bootstrap samples patients with replacement within each patient's highest outcome category (none,
Stage 1 only, Stage ≥2), retaining all evaluated encounters for each selected patient. The **same draws**
are used for all 14 model/outcome combinations. There are 1,000 draws per scope; percentile 95% intervals
and paired AP/AUROC/Brier differences are reported. These intervals condition on the trained models,
fixed split and stratum sizes. No models are retrained inside Bootstrap; no multiplicity correction is applied.

Sensitivity subsets S1–S4 reuse the frozen models and thresholds: any vital observed, complete case,
exclude primary positives already positive on their first sample, and ≥2 current samples. S3 uses the
same primary-defined exclusion for both outcomes, matching P2/P5. S5–S6 are exported as descriptive
flags only. These subsets change the population and ascertainment; they are not causal experiments.
""")
code('''
thresholds, validation_probabilities, validation_rows = {}, {}, []
for target, name in KEYS:
    y = data["validation"][target].to_numpy()
    p = models[target, name].predict_proba(model_inputs["validation"][input_names[target, name]])[:, 1].astype(float)
    threshold = choose_threshold(y, p, SENSITIVITY_TARGET)
    metrics = PreparedMetrics(y, p, threshold).evaluate()
    reference = [average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p)]
    assert np.allclose([metrics[k] for k in METRIC_NAMES[:3]], reference, rtol=0, atol=1e-12)
    assert metrics["sensitivity"] >= SENSITIVITY_TARGET - 1e-12
    assert (p[y == 1] >= np.nextafter(threshold, np.inf)).mean() < SENSITIVITY_TARGET
    thresholds[target, name], validation_probabilities[target, name] = threshold, p
    validation_rows.append({"target": target, "model": name, "threshold": threshold, **metrics})
validation_table = pd.DataFrame(validation_rows)
save_csv(validation_table, "validation_metrics_and_thresholds.csv")
freeze = {"created_at_utc": utc_now(), "sensitivity_target": SENSITIVITY_TARGET,
    "rule": "largest validation cutoff attaining sensitivity >= target; positive if probability >= cutoff",
    "selected_models": [{"target": t, "model": m, "parameters": selected_parameters[t, m], "threshold": thresholds[t, m]} for t, m in KEYS],
    "test_predictions_generated": False, "historical_test_results_previously_known": True,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
    "environment": ENVIRONMENT}
freeze_serialized = json.dumps(freeze, sort_keys=True, default=str)
freeze_hash = hashlib.sha256(freeze_serialized.encode()).hexdigest()
write_json(CHECKS / "analysis_freeze.json", freeze)
threshold_snapshot = thresholds.copy()
print("All 14 validation thresholds frozen before test inference.")
display(validation_table[["target", "model", "average_precision", "auroc", "threshold", "sensitivity", "ppv"]])
''', "validation_freeze")

code('''
write_json(CHECKS / "test_inference_start.json", {"started_at_utc": utc_now(), "analysis_freeze_sha256": freeze_hash})
for name, pre in fitted.items():
    matrices[name]["test"] = pre.transform(X["test"])
model_inputs["test"] = construct_model_inputs("test")
test_probabilities = {}
prediction_frames = []
for target in TARGETS:
    frame = data["test"][["stay_id", "subject_id", target]].rename(columns={target: "y_true"}).copy()
    frame.insert(0, "target", target)
    for name in MODEL_NAMES:
        p = models[target, name].predict_proba(model_inputs["test"][input_names[target, name]])[:, 1].astype(float)
        assert p.shape == (len(data["test"]),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        test_probabilities[target, name] = p
        frame[f"y_prob_{name}"] = p
    prediction_frames.append(frame)
save_csv(pd.concat(prediction_frames, ignore_index=True), "test_predictions_all_models.csv", PRIVATE)
assert thresholds == threshold_snapshot
''', "test_inference")

# Patient-resampling primitives are embedded, not imported from the old P5 file.
evaluation_source = (ROOT / "person5_evaluation/current/evaluate.py").read_text(encoding="utf-8")
evaluation_tree = ast.parse(evaluation_source)
bootstrap_primitives = "\n\n".join(ast.get_source_segment(evaluation_source, node) for node in evaluation_tree.body
    if isinstance(node, ast.FunctionDef) and node.name in {"patient_layout", "draw_multiplicities"})
code(bootstrap_primitives + '''

PAIRS = [("structured_text_lr", "structured_lr"), ("structured_xgboost", "structured_lr"),
         ("structured_text_xgboost", "structured_xgboost"), ("structured_text_xgboost", "structured_text_lr")]
SUBSETS = ["S1_has_any_vital", "S2_complete_case", "S3_excl_positive_at_first_sample", "S4_two_or_more_samples"]

def evaluate_scope(labels, probabilities, scope, scope_index):
    assert len(labels) > 0, f"Empty sensitivity subset: {scope}"
    patients, inverse, stages, groups = patient_layout(labels)
    prepared = [PreparedMetrics(labels[t].to_numpy(), probabilities[t, m], thresholds[t, m]) for t, m in KEYS]
    points = [metric.evaluate() for metric in prepared]
    for (target, name), point in zip(KEYS, points):
        y, p = labels[target].to_numpy(), probabilities[target, name]
        assert abs(point["brier_score"] - brier_score_loss(y, p)) < 1e-12
        if np.unique(y).size == 2:
            assert abs(point["average_precision"] - average_precision_score(y, p)) < 1e-12
            assert abs(point["auroc"] - roc_auc_score(y, p)) < 1e-12
    samples = np.full((BOOTSTRAP_REPLICATES, len(KEYS), len(METRIC_NAMES)), np.nan)
    with threadpool_limits(limits=1):
        for replicate in range(BOOTSTRAP_REPLICATES):
            counts = draw_multiplicities(groups, len(patients), BOOTSTRAP_SEED, scope_index, replicate)
            assert counts.sum() == len(patients)
            weights = counts[inverse]
            for j, metric in enumerate(prepared):
                result = metric.evaluate(weights, include_calibration=scope == "test")
                samples[replicate, j] = [result[k] for k in METRIC_NAMES]
            if (replicate + 1) % 100 == 0 or replicate + 1 == BOOTSTRAP_REPLICATES:
                print(f"{scope}: paired patient Bootstrap {replicate + 1}/{BOOTSTRAP_REPLICATES}", flush=True)
    rows, intervals, pairs = [], [], []
    for j, (target, name) in enumerate(KEYS):
        base = {"scope": scope, "target": target, "model": name}
        rows.append({**base, "encounters": len(labels), "patients": len(patients),
            "positives": int(labels[target].sum()), "prevalence": float(labels[target].mean()),
            "threshold": thresholds[target, name], **points[j]})
        for k, metric in enumerate(METRIC_NAMES):
            if scope != "test" and metric in METRIC_NAMES[-3:]:
                continue
            values = samples[:, j, k]
            values = values[np.isfinite(values)]
            lo, hi = np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
            intervals.append({**base, "metric": metric, "estimate": points[j][metric],
                "ci_lower": lo, "ci_upper": hi, "valid_replicates": len(values), "requested_replicates": BOOTSTRAP_REPLICATES})
    for target in TARGETS:
        for a, b in PAIRS:
            ai, bi = KEYS.index((target, a)), KEYS.index((target, b))
            for k, metric in enumerate(METRIC_NAMES[:3]):
                values = samples[:, ai, k] - samples[:, bi, k]
                values = values[np.isfinite(values)]
                lo, hi = np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
                pairs.append({"scope": scope, "target": target, "model_a": a, "model_b": b,
                    "metric": metric, "difference_a_minus_b": points[ai][metric] - points[bi][metric],
                    "ci_lower": lo, "ci_upper": hi, "valid_replicates": len(values), "multiplicity_adjusted": False})
    # Fresh outputs only; these samples are never read to resume another run.
    np.save(CHECKS / f"bootstrap_{scope}.npy", samples)
    write_json(CHECKS / f"bootstrap_{scope}.json", {"patients": len(patients), "encounters": len(labels),
        "patient_strata": {str(s): len(g) for s, g in enumerate(groups)},
        "same_draws_for_all_models_and_targets": True, "method": "stratified patient cluster percentile Bootstrap",
        "calibration_ci": scope == "test", "keys": KEYS, "metrics": METRIC_NAMES})
    return pd.DataFrame(rows), pd.DataFrame(intervals), pd.DataFrame(pairs)
''', "bootstrap_definition")

code('''
test_labels = data["test"][["subject_id", "stay_id"] + TARGETS].copy()
test_flags = flags.set_index("stay_id").loc[test_labels.stay_id].reset_index()
assert np.array_equal(test_flags.subject_id, test_labels.subject_id)
point_frames, interval_frames, pair_frames = [], [], []
for scope_index, scope in enumerate(["test"] + SUBSETS):
    mask = np.ones(len(test_labels), dtype=bool) if scope == "test" else test_flags[scope].to_numpy(dtype=bool)
    points, intervals, pairs = evaluate_scope(test_labels.loc[mask].reset_index(drop=True),
        {key: probability[mask] for key, probability in test_probabilities.items()}, scope, scope_index)
    point_frames.append(points)
    interval_frames.append(intervals)
    pair_frames.append(pairs)
all_metrics = pd.concat(point_frames, ignore_index=True)
all_intervals = pd.concat(interval_frames, ignore_index=True)
all_pairs = pd.concat(pair_frames, ignore_index=True)
save_csv(all_metrics, "all_metrics.csv")
save_csv(all_intervals, "all_confidence_intervals.csv")
save_csv(all_pairs, "all_paired_differences.csv")
save_csv(all_metrics.loc[all_metrics.scope.eq("test")], "test_metrics.csv")
save_csv(all_metrics.loc[~all_metrics.scope.eq("test")], "sensitivity_metrics.csv")
summary_rows = []
for _, row in all_metrics.loc[all_metrics.scope.eq("test")].iterrows():
    entry = {key: row[key] for key in ["target", "model", "encounters", "positives", "prevalence", "threshold"]}
    for metric in ["average_precision", "auroc", "brier_score", "sensitivity", "specificity", "ppv", "npv"]:
        interval = all_intervals.loc[all_intervals.scope.eq("test") & all_intervals.target.eq(row.target)
            & all_intervals.model.eq(row.model) & all_intervals.metric.eq(metric)].iloc[0]
        entry[metric + " (95% CI)"] = f"{interval.estimate:.4f} [{interval.ci_lower:.4f}, {interval.ci_upper:.4f}]"
    summary_rows.append(entry)
comparison_table = pd.DataFrame(summary_rows)
save_csv(comparison_table, "primary_secondary_comparison.csv")
display(comparison_table)
display(all_pairs.loc[all_pairs.scope.eq("test")])
''', "bootstrap_execution")

md("""
## Figures and interpretation checks

Calibration plots use up to ten quantile bins, keeping equal scores together, and retain bin counts.
Joint calibration intercept/slope and offset-only calibration-in-the-large are diagnostics; test predictions
are never recalibrated. A constant-score dummy has no identifiable joint slope/intercept. Intervals may
be undefined when a metric lacks the necessary outcome classes or predicted positives; valid draw counts
are retained. Feature importance and LR coefficients describe fitted associations, not causal effects.
""")

code('''
MODEL_LABELS = {"dummy": "Training-prevalence dummy", "age_gender_lr": "Age/gender LR",
    "structured_lr": "Structured LR", "structured_text_lr": "Structured + text LR",
    "age_gender_xgboost": "Age/gender XGBoost", "structured_xgboost": "Structured XGBoost",
    "structured_text_xgboost": "Structured + text XGBoost"}
TARGET_LABELS = {"aki_label": "Primary: Stage ≥1", "aki_stage2_label": "Secondary: Stage ≥2"}
COLORS = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(MODEL_NAMES)}

def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    display(fig)
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
for row, target in enumerate(TARGETS):
    y = test_labels[target].to_numpy()
    for name in MODEL_NAMES:
        p = test_probabilities[target, name]
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[row, 0].step(recall, precision, where="post", label=MODEL_LABELS[name], color=COLORS[name])
        axes[row, 1].plot(fpr, tpr, label=MODEL_LABELS[name], color=COLORS[name])
    axes[row, 0].axhline(y.mean(), color="grey", linestyle="--", label="Test prevalence (reference)")
    axes[row, 1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    for col, (xlabel, ylabel) in enumerate([("Recall", "Precision"), ("False positive rate", "Sensitivity")]):
        axes[row, col].set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1), title=TARGET_LABELS[target])
    axes[row, 0].legend(fontsize=8)
save_figure(fig, "01_test_pr_roc_both_outcomes")

calibration_rows = []
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for ax, target in zip(axes, TARGETS):
    y = test_labels[target].to_numpy()
    for name in MODEL_NAMES:
        p = test_probabilities[target, name]
        bins = pd.qcut(p, q=10, labels=False, duplicates="drop") if np.ptp(p) > 0 else np.zeros(len(p))
        grouped = pd.DataFrame({"bin": bins, "probability": p, "y": y}).groupby("bin", dropna=False).agg(
            mean_probability=("probability", "mean"), observed_rate=("y", "mean"), count=("y", "size"), positives=("y", "sum")).reset_index()
        grouped.insert(0, "model", name)
        grouped.insert(0, "target", target)
        calibration_rows.append(grouped)
        ax.plot(grouped.mean_probability, grouped.observed_rate, "o-", markersize=3, color=COLORS[name], label=MODEL_LABELS[name])
    ax.plot([0, 1], [0, 1], "--", color="grey")
    relevant = pd.concat([g for g in calibration_rows if g.target.iloc[0] == target])
    upper = min(1.0, max(0.10, relevant[["mean_probability", "observed_rate"]].max().max() * 1.1))
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive proportion", title=TARGET_LABELS[target], xlim=(0, upper), ylim=(0, upper))
    ax.legend(fontsize=8)
save_csv(pd.concat(calibration_rows, ignore_index=True), "calibration_bins.csv")
save_figure(fig, "02_test_calibration_both_outcomes")
''', "performance_figures")

code('''
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
flow_table = pd.DataFrame(flow)
ax.barh(np.arange(len(flow_table)), flow_table.encounters, color="#3977a8")
ax.set_yticks(np.arange(len(flow_table)), flow_table.step)
ax.invert_yaxis()
ax.set(xlabel="ED encounters", title="Cohort construction from fresh extraction")
for i, value in enumerate(flow_table.encounters):
    ax.text(value, i, f" {value:,}", va="center")
ax.set_xlim(0, flow_table.encounters.max() * 1.18)
save_figure(fig, "03_cohort_flow")

importance_rows = []
fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
for row, target in enumerate(TARGETS):
    lr_coef = models[target, "structured_text_lr"].coef_[0]
    lr_order = np.argsort(np.abs(lr_coef))[-15:]
    axes[row, 0].barh([feature_names["lr_text"][i].replace("text__", "") for i in lr_order], lr_coef[lr_order])
    axes[row, 0].set(title=TARGET_LABELS[target] + ": LR coefficients", xlabel="Coefficient (mixed feature scales)")
    gain = models[target, "structured_text_xgboost"].get_booster().get_score(importance_type="gain")
    xgb_gain = np.array([gain.get(f"f{i}", 0.0) for i in range(len(combined_tree_names))])
    xgb_order = np.argsort(xgb_gain)[-15:]
    axes[row, 1].barh([combined_tree_names[i].replace("text__", "") for i in xgb_order], xgb_gain[xgb_order])
    axes[row, 1].set(title=TARGET_LABELS[target] + ": XGBoost gain", xlabel="Mean split gain")
    for feature, value in zip(feature_names["lr_text"], lr_coef):
        importance_rows.append({"target": target, "model": "structured_text_lr", "feature": feature, "measure": "coefficient", "value": value})
    for feature, value in zip(combined_tree_names, xgb_gain):
        importance_rows.append({"target": target, "model": "structured_text_xgboost", "feature": feature, "measure": "gain", "value": value})
save_csv(pd.DataFrame(importance_rows), "feature_associations.csv")
save_figure(fig, "04_feature_associations")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for ax, target in zip(axes, TARGETS):
    for name in ["structured_lr", "structured_text_lr", "structured_xgboost", "structured_text_xgboost"]:
        subset = all_intervals.loc[all_intervals.target.eq(target) & all_intervals.model.eq(name)
            & all_intervals.metric.eq("average_precision")].set_index("scope").loc[["test"] + SUBSETS]
        x = np.arange(len(subset))
        ax.plot(x, subset.estimate, "o-", label=MODEL_LABELS[name], color=COLORS[name])
        ax.vlines(x, subset.ci_lower, subset.ci_upper, color=COLORS[name], alpha=0.6)
    ax.set_xticks(range(5), ["All test", "S1", "S2", "S3", "S4"])
    ax.set(ylabel="AP (95% patient Bootstrap CI)", title=TARGET_LABELS[target])
    ax.legend(fontsize=8)
save_figure(fig, "05_sensitivity_ap")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for ax, target in zip(axes, TARGETS):
    pairs = all_pairs.loc[all_pairs.scope.eq("test") & all_pairs.target.eq(target) & all_pairs.metric.eq("average_precision")]
    y_positions = np.arange(len(pairs))
    ax.hlines(y_positions, pairs.ci_lower, pairs.ci_upper, color="#3977a8")
    ax.plot(pairs.difference_a_minus_b, y_positions, "o", color="#3977a8")
    ax.axvline(0, linestyle="--", color="grey")
    ax.set_yticks(y_positions, [MODEL_LABELS[r.model_a] + " minus\\n" + MODEL_LABELS[r.model_b] for r in pairs.itertuples()])
    ax.set(xlabel="Paired difference in AP (95% CI)", title=TARGET_LABELS[target])
save_figure(fig, "06_paired_ap_differences")
''', "cohort_importance_sensitivity_figures")

code('''
assert len(models) == 14 and thresholds == threshold_snapshot
assert len(all_metrics) == 14 * 5
assert len(all_pairs) == 2 * 4 * 3 * 5
assert all_intervals.requested_replicates.eq(BOOTSTRAP_REPLICATES).all()
assert len(query_audit) == 7 and not any(row["cache_hit"] for row in query_audit)
assert all((labels.aki_stage2_label <= labels.aki_label).all() for labels in data.values())
for target in TARGETS:
    assert np.unique(test_probabilities[target, "dummy"]).size == 1
    assert np.isclose(test_probabilities[target, "dummy"][0], data["train"][target].mean())
    dummy = all_metrics.loc[all_metrics.scope.eq("test") & all_metrics.target.eq(target) & all_metrics.model.eq("dummy")].iloc[0]
    assert np.isclose(dummy.average_precision, test_labels[target].mean()) and np.isclose(dummy.auroc, 0.5)
run_manifest = [{"file": path.relative_to(RUN_DIR).as_posix(), "bytes": path.stat().st_size,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in RUN_DIR.rglob("*") if path.is_file()
    and path.name not in {"artifact_manifest.json", "run_status.json"}]
write_json(CHECKS / "artifact_manifest.json", run_manifest)
write_json(CHECKS / "run_status.json", {"status": "complete", "finished_at_utc": utc_now(),
    "models": 14, "outcomes": TARGETS, "bootstrap_replicates_per_scope": BOOTSTRAP_REPLICATES,
    "scopes": ["test"] + SUBSETS, "source": "fresh BigQuery extraction", "reads_existing_data_or_models": False,
    "fit_partition": "train", "threshold_partition": "validation", "analysis_freeze_sha256": freeze_hash,
    "historical_test_results_previously_known": True})
print("Complete:", RUN_DIR)
print("Comparison table: tables/primary_secondary_comparison.csv")
print("Paired differences and intervals: tables/all_paired_differences.csv")
print("Patient-level outputs are in private/; do not include them in a public submission.")
''', "final_verification")

md("""
### Reading the outputs

- AP must be compared with the evaluated population's prevalence; the learned dummy uses **training** prevalence.
  Test prevalence is a descriptive reference, not a fitted prediction. Brier's oracle constant reference
  `π_test(1−π_test)` is not the dummy's actual score unless training and test prevalence coincide.
- A paired difference CI entirely above zero supports a positive AP/AUROC difference under this resampling
  procedure. For Brier, a negative difference favours model A. A CI crossing zero means uncertainty,
  not demonstrated equivalence. Comparisons are exploratory and unadjusted for multiplicity.
- AP across Stage ≥1 and Stage ≥2, or across sensitivity subsets, is affected by prevalence and case mix.
  Do not interpret a lower AP in the rarer endpoint as sufficient evidence of poorer discrimination.
- Validation sensitivity ≥0.80 does not guarantee test sensitivity ≥0.80. PPV also depends on prevalence
  and false positives. These models do not establish prospective clinical utility or transportability.
- Creatinine-based ascertainment, missing vitals, single-centre data and text reflecting existing clinical
  suspicion remain limitations. Early recognition at ED admission is not necessarily prediction before suspicion.

The notebook produces new CSV summaries and private predictions **as outputs only**. No generated CSV
is subsequently read into this run. Submit this notebook; keep row-level clinical data and Google credentials
in authorized private storage. The optional local builder/check scripts are maintenance tools, not runtime dependencies.
""")

manifest = [{"contributor_source": str(path.relative_to(ROOT)).replace("\\", "/"),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in SOURCES.values()]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (sml)", "language": "python", "name": "sml"},
    "language_info": {"name": "python", "version": "3.12.11"},
    "integration": {"raw_data_only": True, "source_manifest": manifest, "version": 1},
})
for cell in cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
        forbidden = ["pd.read_csv(", "np.load(", "joblib.load(", "read_parquet(", "runpy", "/content/", "google.colab"]
        assert not any(token in cell.source for token in forbidden), cell.metadata.tags
nbf.validate(nb)
path = HERE / "Final_AKI_End_to_End.ipynb"
nbf.write(nb, path)
(HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Built {path.name}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")
