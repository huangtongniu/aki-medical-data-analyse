import joblib
import numpy as np
from scipy import sparse

def dense_to_explicit_csr(X):
    X = np.asarray(X)
    n_rows, n_cols = X.shape

    data = X.ravel()
    indices = np.tile(np.arange(n_cols, dtype=np.int32), n_rows)
    indptr = np.arange(
        0,
        (n_rows + 1) * n_cols,
        n_cols,
        dtype=np.int64)

    return sparse.csr_matrix(
        (data, indices, indptr),
        shape=X.shape)

def prepare_p4_text_matrix(X_structured_dense, X_text_raw):
    X_structured_explicit = dense_to_explicit_csr(
        X_structured_dense)

    X_text_only = X_text_raw[:, X_structured_dense.shape[1]:]

    return sparse.hstack(
        [X_structured_explicit, X_text_only],
        format="csr")

def load_p4_models(model_dir):
    models = {
        "age_gender_xgboost": joblib.load(
            model_dir / "age_gender_xgboost.joblib"),
        "structured_xgboost": joblib.load(
            model_dir / "structured_xgboost.joblib"),
        "structured_chief_complaint_xgboost": joblib.load(
            model_dir / "structured_chief_complaint_xgboost.joblib"),
        "stage2_age_gender_xgboost": joblib.load(
            model_dir / "stage2_age_gender_xgboost.joblib"),
        "stage2_structured_xgboost": joblib.load(
            model_dir / "stage2_structured_xgboost.joblib"),
        "stage2_structured_chief_complaint_xgboost": joblib.load(
            model_dir / "stage2_structured_chief_complaint_xgboost.joblib")}

    return models

def p4_predict(models, X_age_gender, X_structured, X_text_raw):
    X_text_corrected = prepare_p4_text_matrix(
        X_structured,
        X_text_raw)

    predictions = {}

    for model_name, model in models.items():
        if "age_gender" in model_name:
            X = X_age_gender
        elif "chief_complaint" in model_name:
            X = X_text_corrected
        else:
            X = X_structured

        predictions[model_name] = model.predict_proba(X)[:, 1]

    return predictions
