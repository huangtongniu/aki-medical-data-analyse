"""Paths are anchored to this file; stored project paths are relative, portable paths."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Keep imports of teammates' modules read-only, including their __pycache__ folders.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P1 = ROOT / "person1_handoff"
P2 = ROOT / "person2_handoff/person2_split_preprocessing"
P2_QA = ROOT / "person2_handoff/person2_phenotype_review"
P3 = ROOT / "person3_handoff_v2_2026-10-16/person3_handoff"
P4 = ROOT / "person4_handoff/person4_handoff"
OUTPUT = HERE / "outputs"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
CHECKS = OUTPUT / "checks"
PREDICTIONS = OUTPUT / "predictions"

MODEL_NAMES = [
    "prevalence", "age_gender_lr", "structured_lr", "structured_text_lr",
    "age_gender_xgboost", "structured_xgboost", "structured_chief_complaint_xgboost",
]
MODEL_LABELS = {
    "prevalence": "Training prevalence",
    "age_gender_lr": "Age / gender LR",
    "structured_lr": "Structured LR",
    "structured_text_lr": "Structured + text LR",
    "age_gender_xgboost": "Age / gender XGBoost",
    "structured_xgboost": "Structured XGBoost",
    "structured_chief_complaint_xgboost": "Structured + text XGBoost",
}


def make_output_dirs():
    for p in [TABLES, FIGURES, CHECKS, PREDICTIONS]:
        p.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def settings():
    return read_json(HERE / "settings.json")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    names = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost", "matplotlib", "statsmodels"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "environment_name": Path(sys.prefix).name,
        "packages": versions,
        "thread_limits": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def upstream_files():
    """Inventory original deliverables only; no data are copied or modified."""
    files = []
    for folder in [P1, ROOT / "person2_handoff", P3, P4]:
        files.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.extend(ROOT.glob("*.zip"))
    files.extend([ROOT / "Project Method.docx", ROOT / "what should I do.md"])
    return sorted(set(files))


def freeze_sources():
    """Development artifacts only: this function does not read test labels/matrices."""
    paths = [HERE / "settings.json", HERE / "common.py", HERE / "inference.py", HERE / "metrics.py",
             HERE / "run_validation.py", HERE / "run_test_evaluation.py", P2 / "aki_preprocessing.py",
             P2 / "outputs/split_manifest.csv", P1 / "predictor_columns.csv",
             P3 / "outputs/model_metadata.json", P4 / "outputs/model_settings.json"]
    for folder, names in [(P3, MODEL_NAMES[:4]), (P4, MODEL_NAMES[4:])]:
        paths.extend(folder / f"outputs/models/{name}.joblib" for name in names)
        paths.append(folder / "outputs/validation_predictions.csv")
    paths.extend(sorted((P2 / "outputs/preprocessors").glob("*")))
    paths.extend(sorted((P2 / "outputs/matrices").glob("*validation*")))
    return {relative(p): sha256(p) for p in paths}


def check_frozen_sources(freeze):
    for path, expected in freeze["source_sha256"].items():
        require(sha256(ROOT / path) == expected, f"Frozen source changed: {path}")
