# Person 4 Handoff — XGBoost and Predictor Comparisons

## Objective

Develop and compare XGBoost models for both the primary Stage ≥1 AKI outcome and the secondary Stage ≥2 AKI outcome using Person 2's frozen patient-level split and preprocessing outputs.

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

Person 3's validation metrics for the primary and Stage ≥2 baseline models are also loaded for final model comparisons.

## Data

Training set:

- 79,364 encounters
- Stage ≥1 AKI prevalence: 6.99%
- Stage ≥2 AKI prevalence: 2.00%

Validation set:

- 16,932 encounters
- Stage ≥1 AKI prevalence: 6.88%
- Stage ≥2 AKI prevalence: 1.97%

The frozen patient-level training and validation split from Person 2 was retained for both outcomes. Encounter assignments were checked against the split manifest and no patient overlap was found.

Stage ≥2 labels were also checked to confirm that all Stage ≥2 positives were nested within the primary Stage ≥1 AKI outcome.

## Predictor Groups

For both Stage ≥1 and Stage ≥2, three XGBoost predictor groups were evaluated:

1. Age/Gender — 2 features
2. Structured predictors — 19 features
3. Structured + Chief Complaint — 2,791 transformed features

Average Precision was used as the primary model-selection metric. AUROC and Brier score were reported as supporting metrics.

## Structured Zero Handling

Person 2's existing chief-complaint vocabulary and TF-IDF representation were retained.

The original text-inclusive matrix is sparse. Because absent sparse entries can be treated as missing by XGBoost, the structured feature block was reconstructed so that structured zero values were explicitly stored rather than treated as absent entries.

The corrected structured block was checked against the original dense structured matrix. Zero values and NaN values were preserved exactly in both the training and validation partitions.

The same corrected text-inclusive representation was used for both Stage ≥1 and Stage ≥2 XGBoost models.

The training and validation split was not changed.

# Stage ≥1 Primary Analysis

## Stage ≥1 Structured XGBoost Tuning

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

## Stage ≥1 Predictor Group Comparison

Using the fixed structured-XGBoost settings:

| Predictor group | Features | AP | AUROC | Brier |
|---|---:|---:|---:|---:|
| Age/Gender | 2 | 0.074466 | 0.531112 | 0.064048 |
| Structured | 19 | 0.124828 | 0.645931 | 0.062673 |
| Structured + Chief Complaint | 2791 | 0.148056 | 0.671179 | 0.062013 |

The structured + chief complaint model showed the strongest development performance and was therefore tuned further.

## Final Stage ≥1 Structured + Chief Complaint Model

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

No test data were used during Stage ≥1 model development or hyperparameter selection.

## Stage ≥1 Feature Interpretation

Gain-based XGBoost feature importance was calculated for the final Stage ≥1 structured + chief complaint model.

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

# Stage ≥2 Secondary Analysis

## Stage ≥2 Labels

The secondary target identifies observed creatinine Stage ≥2 AKI.

Training:

- Positive encounters: 1,589
- Prevalence: 2.00%

Validation:

- Positive encounters: 333
- Prevalence: 1.97%

The same frozen patient-level partitions and predictor matrices used for the primary analysis were retained.

## Stage ≥2 Structured XGBoost Tuning

The same limited hyperparameter search space used for the primary analysis was applied to the Stage ≥2 structured model.

Selected structured-model parameters:

- `n_estimators = 200`
- `max_depth = 4`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`

Validation performance:

- Average Precision: 0.056399
- AUROC: 0.683630
- Brier score: 0.019024

## Stage ≥2 Predictor Results

Using the selected structured-XGBoost settings:

| Predictor group | Features | AP | AUROC | Brier |
|---|---:|---:|---:|---:|
| Age/Gender | 2 | 0.025961 | 0.564014 | 0.019269 |
| Structured | 19 | 0.056399 | 0.683630 | 0.019024 |
| Structured + Chief Complaint | 2791 | 0.076075 | 0.701369 | 0.018800 |

The Stage ≥2 structured + chief complaint model showed the strongest development performance and was tuned further.

## Final Stage ≥2 Structured + Chief Complaint Model

Selected parameters:

- `n_estimators = 300`
- `max_depth = 3`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`

Validation performance:

- Average Precision: 0.076271
- AUROC: 0.697470
- Brier score: 0.018790

No test data were used during Stage ≥2 model development or hyperparameter selection.

## Stage ≥2 Feature Interpretation

Gain-based XGBoost feature importance was calculated for the final Stage ≥2 structured + chief complaint model.

Highly ranked chief-complaint features included:

- fistula
- labs
- renal failure
- renal
- clotted
- urinary retention
- abnormal labs
- hyperkalemia
- hypotension

Structured predictors including systolic blood pressure, gender, arrival transport and age also appeared among the top-ranked features.

As with the primary model, gain importance indicates predictive contribution within the fitted model and should not be interpreted as a causal effect.

## Loading the Text-Inclusive XGBoost Models

Both the Stage ≥1 and Stage ≥2 structured + chief complaint XGBoost models must receive input with the same zero handling used during training.

Do not pass the original sparse `tree_text` matrix directly to either saved text-inclusive XGBoost model.

Before prediction:

1. Load the dense structured matrix.
2. Load the original sparse text-inclusive matrix.
3. Reconstruct the structured block so that zero values are explicitly stored.
4. Keep the original chief-complaint/text columns.
5. Combine the corrected structured block with the text block.
6. Pass the corrected matrix to the corresponding saved XGBoost model.

This preprocessing must remain consistent for validation and future test evaluation.

## Outputs

Primary Stage ≥1 outputs in `outputs/`:

- `structured_tuning_record.csv`
- `chief_complaint_tuning_record.csv`
- `predictor_group_comparison.csv`
- `final_model_comparison.csv`
- `validation_predictions.csv`
- `final_feature_importance.csv`
- `model_settings.json`

Stage ≥2 outputs in `outputs/`:

- `stage2_structured_tuning_record.csv`
- `stage2_chief_complaint_tuning_record.csv`
- `stage2_final_model_comparison.csv`
- `stage2_validation_predictions.csv`
- `stage2_final_feature_importance.csv`
- `stage2_model_settings.json`

Primary Stage ≥1 models in `outputs/models/`:

- `age_gender_xgboost.joblib`
- `structured_xgboost.joblib`
- `structured_chief_complaint_xgboost.joblib`

Stage ≥2 models in `outputs/models/`:

- `stage2_age_gender_xgboost.joblib`
- `stage2_structured_xgboost.joblib`
- `stage2_structured_chief_complaint_xgboost.joblib`

Figures in `outputs/figures/`:

- `final_feature_importance.png`
- `stage2_final_feature_importance.png`

## Reproducibility

Software versions are recorded in `software_versions.json`.

The saved Stage ≥1 and Stage ≥2 XGBoost models should reproduce their original validation predictions when reloaded using the same validation inputs.

For the text-inclusive models, the corrected sparse representation with explicitly preserved structured zero values must be used.

## Final Models

The selected final Stage ≥1 model is:

`structured_chief_complaint_xgboost`

The selected final Stage ≥2 model is:

`stage2_structured_chief_complaint_xgboost`

Both models should remain frozen before independent held-out test evaluation.
