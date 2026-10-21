"""Shared paths and provenance for the two-outcome evaluation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / "person5_evaluation"
P1 = ROOT / "person1_handoff"
P2 = ROOT / "person2_handoff/person2_split_preprocessing"
P2_QA = ROOT / "person2_handoff/person2_phenotype_review"
P3_PRIMARY = HERE / "inputs/person3_primary/person3_handoff"
P3_SECONDARY = ROOT / "person3_stage2_handoff/person3_stage2_handoff"
P4 = ROOT / "person4_handoff/person4_handoff"
OUTPUT = HERE / "outputs"
CHECKS = OUTPUT / "checks"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
PREDICTIONS = OUTPUT / "predictions"
TARGETS = ["aki_label", "aki_stage2_label"]
TARGET_LABELS = {"aki_label": "Primary: Stage >=1", "aki_stage2_label": "Secondary: Stage >=2"}
MODEL_NAMES = ["prevalence", "age_gender_lr", "structured_lr", "structured_text_lr",
               "age_gender_xgboost", "structured_xgboost", "structured_chief_complaint_xgboost"]
MODEL_LABELS = {"prevalence": "Training prevalence", "age_gender_lr": "Age / gender LR",
    "structured_lr": "Structured LR", "structured_text_lr": "Structured + text LR",
    "age_gender_xgboost": "Age / gender XGBoost", "structured_xgboost": "Structured XGBoost",
    "structured_chief_complaint_xgboost": "Structured + text XGBoost"}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


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


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "xgboost", "matplotlib", "statsmodels"]
    return {"python": sys.version.split()[0], "environment_name": Path(sys.prefix).name,
            "packages": {n: importlib.metadata.version(n) for n in packages}}


def make_output_dirs():
    for folder in [CHECKS, TABLES, FIGURES, PREDICTIONS]:
        folder.mkdir(parents=True, exist_ok=True)


def save_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def recover_primary_package():
    """Recover the retained primary archive inside P5, without restoring deleted teammate paths."""
    archive = ROOT / "person3_handoff_v2_2026-10-16.zip"
    destination = P3_PRIMARY.parent.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir() or not member.filename.startswith("person3_handoff/"):
                continue
            path = (destination / member.filename).resolve()
            require(path.is_relative_to(destination), "Archive path escapes the P5 input directory")
            data = bundle.read(member)
            if path.exists():
                require(path.read_bytes() == data, f"Recovered archive entry changed: {relative(path)}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    old = read_json(OLD / "outputs/checks/analysis_freeze.json")
    rows = []
    for name in MODEL_NAMES:
        path = (P3_PRIMARY if name in MODEL_NAMES[:4] else P4) / f"outputs/models/{name}.joblib"
        old_key = (f"person3_handoff_v2_2026-10-16/person3_handoff/outputs/models/{name}.joblib"
                   if name in MODEL_NAMES[:4] else relative(path))
        match = sha256(path) == old["source_sha256"][old_key]
        require(match, f"Primary model differs from archived freeze: {name}")
        rows.append({"model_name": name, "file": relative(path), "matches_v1_frozen_hash": match})
    write_json(CHECKS / "primary_package_recovery.json", {"archive": relative(archive),
               "archive_sha256": sha256(archive), "models": rows, "teammate_paths_restored": False})


def source_hashes():
    paths = [HERE / name for name in ["settings.json", "common.py", "inference.py", "metrics.py", "evaluate.py"]]
    paths += [P2 / "aki_preprocessing.py", P2 / "outputs/split_manifest.csv", P1 / "predictor_columns.csv",
              P4 / "P4_inference.py", OLD / "outputs/checks/analysis_freeze.json",
              OLD / "outputs/predictions/validation_predictions_wide.csv"]
    for folder in [P3_PRIMARY, P3_SECONDARY, P4]:
        paths += list((folder / "outputs/models").glob("*.joblib"))
        paths += list((folder / "outputs").glob("*validation*.csv"))
        paths += list((folder / "outputs").glob("*model*.json"))
    paths += list((P2 / "outputs/preprocessors").glob("*"))
    paths += list((P2 / "outputs/matrices").glob("*validation*"))
    paths += [P2_QA / "outputs/sensitivity_subset_flags.csv"]
    return {relative(p): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def check_sources(freeze):
    for path, expected in freeze["source_sha256"].items():
        require(sha256(ROOT / path) == expected, f"Frozen source changed: {path}")


def original_hashes():
    paths = []
    for folder in [P1, ROOT / "person2_handoff", P3_SECONDARY, P4, OLD]:
        paths += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += list(ROOT.glob("*.zip"))
    return {relative(p): sha256(p) for p in sorted(set(paths))}
