# Person 4 Handoff — XGBoost and Predictor Comparisons

## Objective

Develop and compare XGBoost models for predicting the primary AKI outcome using Person 2's frozen patient-level split and preprocessing outputs.

## Inputs

Person 4 reuses Person 2's split and preprocessing outputs.

Main inputs:

- `tree_structured_train.npy`
- `tree_structured_validation.npy`
- `tree_text_train.npz`
- `tree_text_validation.npz`
- `labels_train.csv`
- `labels_validation.csv`
- `split_manifest.csv`
- `tree_structured_feature_names.csv`
- `tree_text_feature_names.csv`

Person 3's validation metrics are also loaded for comparison with the baseline models.

## Data

Training set:

- 79,364 encounters
- AKI prevalence: 6.99%

Validation set:

- 16,932 encounters
- AKI prevalence: 6.88%

The frozen patient-level training and validation split from Person 2 was retained. Encounter assignments were checked against the split manifest and no patient overlap was found.

## Predictor Groups

Three XGBoost predictor groups were evaluated:

1. Age/Gender — 2 features
2. Structured predictors — 19 features
3. Structured + Chief Complaint — 2,791 transformed features

Average Precision was used as the primary model-selection metric. AUROC and Brier score were reported as supporting metrics.

## Structured Zero Handling

Person 2's existing chief-complaint vocabulary and TF-IDF representation were retained.

The original text-inclusive matrix is sparse. Because absent sparse entries can be treated as missing by XGBoost, the structured feature block was reconstructed so that structured zero values were explicitly stored rather than treated as absent entries.

The corrected structured block was checked against the original dense structured matrix. Zero values and NaN values were preserved exactly in both the training and validation partitions.

The training and validation split was not changed.

## Structured XGBoost Tuning

A limited hyperparameter search was performed over:

- `n_estimators`
- `max_depth`
- `learning_rate`

`subsample` and `colsample_bytree` were fixed at 0.8.

Selected structured-model parameters:

- `n_estimators = 400`
- `max_depth = 3`
- `learning_rate = 0.03`
- `subsample = 0.8`
- `colsample_bytree = 0.8`

Validation performance:

- Average Precision: 0.124828
- AUROC: 0.645931
- Brier score: 0.062673

## Predictor Group Comparison

Using the fixed structured-XGBoost settings:

| Predictor group | Features | AP | AUROC | Brier |
|---|---:|---:|---:|---:|
| Age/Gender | 2 | 0.074466 | 0.531112 | 0.064048 |
| Structured | 19 | 0.124828 | 0.645931 | 0.062673 |
| Structured + Chief Complaint | 2791 | 0.148056 | 0.671179 | 0.062013 |

The structured + chief complaint model showed the strongest development performance and was therefore tuned further.

## Final Structured + Chief Complaint Model

Selected parameters:

- `n_estimators = 300`
- `max_depth = 3`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`

Validation performance:

- Average Precision: 0.148320
- AUROC: 0.672716
- Brier score: 0.061982

No test data were used during model development or hyperparameter selection.

## Feature Interpretation

Gain-based XGBoost feature importance was calculated for the final structured + chief complaint model.

Highly ranked chief-complaint features included:

- labs
- cardiac arrest
- renal failure
- abnormal labs
- fistula
- hypotension
- hyperkalemia

Structured predictors including systolic blood pressure, arrival transport, triage acuity, respiratory rate, diastolic blood pressure and gender also appeared among the top-ranked features.

Gain importance indicates how useful a feature was for prediction within the fitted model, but it does not show whether the feature increased or decreased predicted risk. Some chief-complaint terms may also reflect clinical information or concerns already present at ED arrival.

## Loading the Final Text-Inclusive Model

The final `structured_chief_complaint_xgboost` model must receive input with the same zero handling used during training.

Do not pass the original sparse `tree_text` matrix directly to the saved XGBoost model.

Before prediction:

1. Load the dense structured matrix.
2. Load the original sparse text-inclusive matrix.
3. Reconstruct the structured block so that zero values are explicitly stored.
4. Keep the original chief-complaint/text columns.
5. Combine the corrected structured block with the text block.
6. Pass the corrected matrix to the saved model.

This preprocessing must remain consistent for validation and future test evaluation.

## Outputs

`outputs/`

- `structured_tuning_record.csv`
- `chief_complaint_tuning_record.csv`
- `predictor_group_comparison.csv`
- `final_model_comparison.csv`
- `validation_predictions.csv`
- `final_feature_importance.csv`
- `model_settings.json`

`outputs/models/`

- `age_gender_xgboost.joblib`
- `structured_xgboost.joblib`
- `structured_chief_complaint_xgboost.joblib`

`outputs/figures/`

- `final_feature_importance.png`

## Final Model

The selected Person 4 model is:

`structured_chief_complaint_xgboost`

The model should remain frozen before independent held-out test evaluation.
