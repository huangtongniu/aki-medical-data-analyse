# Person 2: patient-separated split and training-only preprocessing

*Prepared 13 October 2026. This follows the "Next team member" instructions in `aki_handoff/README.txt`. Input: `aki_handoff/modelling_data.csv` (113,097 encounters, 66,090 patients).*

## Split (frozen; please don't re-split)

The split is 70 / 15 / 15 **by patient** (`subject_id`).

- It is stratified at patient level on the worst observed outcome: none (59,190 patients), stage 1 only (4,770) or stage ≥2 (2,130).
- The seed is `90089`. It was fixed before the split and run once.
- Both targets use the same partitions. The natural outcome frequency is kept, with no resampling.

| Partition | Patients | Encounters | Primary positives | Primary prevalence | Stage ≥2 positives | Stage ≥2 prevalence |
|---|---|---|---|---|---|---|
| train | 46,263 (70.0%) | 79,364 (70.2%) | 5,546 | 6.99% | 1,589 | 2.00% |
| validation | 9,913 (15.0%) | 16,932 (15.0%) | 1,165 | 6.88% | 333 | 1.97% |
| test | 9,914 (15.0%) | 16,801 (14.9%) | 1,155 | 6.87% | 332 | 1.98% |

Checks passed:
- Every encounter is assigned exactly once.
- Zero patients overlap between any two partitions.
- Median age (66) and the share with all six vitals missing (7.7–7.9%) are similar across partitions.

**The test partition stays untouched until the team records that models are frozen.**

## Preprocessing (fitted on train only)

The code is in `aki_preprocessing.py`. The inputs are exactly the 13 fields in `predictor_columns.csv`.

| Variant | Use | Numeric (age, 6 vitals, acuity, pain score) | Missingness | Categorical (gender, transport, pain_status) | Chief complaint | Features |
|---|---|---|---|---|---|---|
| `lr_structured` | LR, structured | train median, then standardised | 0/1 indicator for 8 fields | one-hot, unseen → zeros | – | 27 |
| `lr_text` | LR, text-inclusive | same | same | same | TF-IDF | 2,799 |
| `tree_structured` | XGBoost | passthrough, NaN kept | implicit | one-hot | – | 19 |
| `tree_text` | XGBoost, text-inclusive | same | same | same | TF-IDF | 2,791 |

Details:
- **Acuity** is treated as ordinal numeric; levels 4–5 are rare.
- **Missingness indicators** are added because missing vitals are strongly linked to AKI (13.5% vs 6.4%; see `person2_phenotype_review`).
- **TF-IDF:**
  - text is lowercased;
  - the tokeniser keeps `s/p`, `n/v` and `r/o` as single tokens;
  - it uses 1–2 grams, `min_df = 5` and sublinear tf;
  - the vocabulary of 2,772 terms comes from training text only.

Leakage checks all pass (`outputs/preprocessing_checks.csv`):
- The imputer and scaler statistics equal the training values.
- The TF-IDF vocabulary equals a refit on training text and differs from an all-data fit.
- Unseen categories encode to all zeros.
- No identifier or outcome appears among the features.
- A reloaded preprocessor reproduces the saved matrices exactly.

## Files (`outputs/`)

| File | Content |
|---|---|
| `split_manifest.csv` | `subject_id`, `stay_id`, `partition`. **This is the shared split, used by everyone.** |
| `split_summary.csv` | The table above |
| `matrices/labels_<partition>.csv` | `subject_id`, `stay_id`, `partition`, `aki_label`, `aki_stage2_label`. Row order equals matrix row order. |
| `matrices/X_raw_<partition>.csv` | Cleaned but unprocessed predictors, for refitting inside CV folds |
| `matrices/<variant>_<partition>.npy` / `.npz` | Transformed matrices: dense `.npy`, sparse (text) `.npz` |
| `preprocessors/<variant>.joblib` | Fitted ColumnTransformers. They need `aki_preprocessing.py` on the path. |
| `preprocessors/<variant>_feature_names.csv` | Column names for each matrix |
| `preprocessing_checks.csv` | Leakage and behaviour checks |
| `environment.txt` | Package versions (scikit-learn 1.7.2) and the seed |

## Notes for Person 3 (baselines), Person 4 (XGBoost) and Person 5 (evaluation)

- **Loading:**
  - `np.load("…/lr_structured_train.npy")` or `scipy.sparse.load_npz("…/lr_text_train.npz")`;
  - labels come from `labels_train.csv`, column `aki_label` or `aki_stage2_label`.
- **Cross-validation inside training:** don't reuse the fitted preprocessors. Put `aki_preprocessing.PREPROCESSORS[variant]()` in a `Pipeline` with the model, and use `StratifiedGroupKFold` on `subject_id`, so preprocessing is refit in every fold.
- **XGBoost and sparse input:** XGBoost treats *absent* sparse entries as missing, not 0. Use the dense `tree_structured` `.npy`. For `tree_text`, the same applies to one-hot and TF-IDF zeros.
- **sklearn versions:** if Colab's scikit-learn differs from 1.7.2 and a `.joblib` won't load, refit from `X_raw_train.csv` with the same builder. The output is identical by construction.
- **Prediction files (team format):** `stay_id`, `subject_id`, target, observed label, predicted probability, model name, partition.
- **Data handling:** all files are row-level MIMIC-derived data. Keep them in approved private storage only.
