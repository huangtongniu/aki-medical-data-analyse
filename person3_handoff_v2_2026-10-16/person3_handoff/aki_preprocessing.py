"""Shared preprocessing for the MIMIC-IV-ED AKI project.

Every transformer here must be fitted on the TRAINING partition only, then
applied with .transform() to validation and test.

Keep this file next to the saved .joblib preprocessors. Unpickling them
imports this module.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Model inputs (identical to aki_handoff/predictor_columns.csv)
PREDICTORS = [
    "age", "gender", "arrival_transport",
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
    "acuity", "pain_numeric", "pain_status", "chiefcomplaint",
]

NUMERIC = ["age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
           "acuity", "pain_numeric"]
# Numeric fields with missing values; each gets a 0/1 missingness indicator
MISSING_FLAGGED = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
                   "acuity", "pain_numeric"]
CATEGORICAL = ["gender", "arrival_transport", "pain_status"]
TEXT = "chiefcomplaint"

# Keeps clinical shorthand such as "s/p", "n/v" and "r/o" as single tokens
TOKEN_PATTERN = r"[a-z0-9]+(?:/[a-z0-9]+)*"


def clean_text(x):
    """Lowercase, trim and fill missing chief complaints with empty text."""
    return pd.Series(x).fillna("").astype(str).str.lower().str.strip().to_numpy()


def _categorical():
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def _text():
    return Pipeline([
        ("clean", FunctionTransformer(clean_text, feature_names_out="one-to-one")),
        ("tfidf", TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                                  min_df=5, sublinear_tf=True)),
    ])


def build_lr_preprocessor(include_text=False):
    """For logistic regression: median imputation + missingness indicators +
    standard scaling of numeric fields, one-hot categoricals, optional TF-IDF."""
    parts = [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("miss", MissingIndicator(features="all"), MISSING_FLAGGED),
        ("cat", _categorical(), CATEGORICAL),
    ]
    if include_text:
        parts.append(("text", _text(), TEXT))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.3 if include_text else 0)


def build_tree_preprocessor(include_text=False):
    """For tree models (e.g. XGBoost): numeric fields passed through with NaN
    kept (XGBoost learns a missing-value direction), one-hot categoricals,
    optional TF-IDF. No scaling or imputation."""
    parts = [
        ("num", "passthrough", NUMERIC),
        ("cat", _categorical(), CATEGORICAL),
    ]
    if include_text:
        parts.append(("text", _text(), TEXT))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.3 if include_text else 0)


PREPROCESSORS = {
    "lr_structured": lambda: build_lr_preprocessor(include_text=False),
    "lr_text": lambda: build_lr_preprocessor(include_text=True),
    "tree_structured": lambda: build_tree_preprocessor(include_text=False),
    "tree_text": lambda: build_tree_preprocessor(include_text=True),
}
