# Person 3 primary-outcome baseline handoff

Updated 16 October 2026 for COMP90089 Group 7. Version 2 adds text-inclusive LR
to the three baselines delivered on 13 October.

This handoff contains four primary-outcome baselines: training-prevalence
constant prediction, age/gender logistic regression, and structured L2 logistic
regression, plus structured-and-chief-complaint L2 logistic regression. It reuses
Person 2's patient split and lr_structured/lr_text preprocessing unchanged.
Only training and validation files are loaded by the development workflow.
Stage >=2 and sensitivity models are not included.

## Read first

1. `P3_baseline_models.ipynb`: executed notebook, complete Person 3 code and aggregate results.
2. `outputs/validation_metrics.csv`: the four-model comparison on identical validation encounters.
3. `outputs/tuning_record.csv`: the fixed age/gender fit and both six-value LR searches.
4. `P3_report_contribution.md`: baseline methods and validation results for report integration.
5. `DECISIONS_P3.md`: choices recorded before tuning and proposed shared evaluation protocol.

The task is early identification of observed creatinine-defined AKI, including
first-sample positives, among selected outcome-assessable admitted encounters.
Operational negatives do not prove clinical absence of AKI. Interpret the
results within that population and label definition.

## Development specification

| Model | Predictive inputs | Fitting and tuning |
|---|---|---|
| `prevalence` | None | Constant probability from training encounter prevalence; DummyClassifier prior |
| `age_gender_lr` | Age, male indicator | Unpenalised LR; no tuning; female reference |
| `structured_lr` | 27 shared transformed structured features | L2 LR; C = 0.001, 0.01, 0.1, 1, 10, 100; choose maximum validation AP |
| `structured_text_lr` | The same 27 structured columns plus 2,772 TF-IDF terms | L2 LR; independently select C from the same six values by validation AP |

LR uses lbfgs and tol 1e-8; max_iter is 2000 for the earlier models and 20000 for
the text model. AP ties within 1e-12 select smaller C.
The text limit was increased from 5000 after C = 100 initially did not converge;
the C grid, tolerance and selection rule were retained.
All encounters have equal training weight; no resampling or class weighting is
used. The prevalence baseline learns an encounter frequency, not a patient-level
ever-positive frequency. Final saved classifiers remain fitted on training only.

The earlier pipelines retain Person 2's fitted `lr_structured` transformer. The age/gender
pipeline then passes only `num__age` and `cat__gender_M` to its classifier. Using
the male column with an intercept avoids redundant binary gender columns; this
preserves Person 2's encoding. The structured model uses all 27 shared columns
and does not include chief-complaint text. The new `structured_text_lr` pipeline
retains Person 2's fitted `lr_text` transformer. Its 27 structured columns are
exactly equal to the structured model's columns on every development row. Its
2,772 TF-IDF terms, IDF weights and tokenizer settings match Person 2's tree_text
preprocessor. It does not use Person 4's XGBoost-specific sparse-zero correction.

The original three baselines' validation IDs, labels, probabilities and metrics
are checked against `reference_baselines_v1.json`, generated from the archived
13 October handoff before this update. The old ZIP remains available separately;
use the latest `person3_handoff.zip` for the current four-model handoff.

Validation comprises 16,932 encounters from 9,913 patients, with 1,165 positives.
Both structured and text C values were selected on these validation results, so the displayed
performance is a development estimate, not an independent final estimate.
Patient-level bootstrap intervals and final test comparisons are pending.

## Reproduce

Keep the existing upstream team files beside `person3_handoff`:

```text
Group Project/
  aki_handoff/predictor_columns.csv
  person2_handoff/person2_split_preprocessing/
    aki_preprocessing.py
    outputs/split_manifest.csv
    outputs/preprocessors/lr_structured.joblib
    outputs/preprocessors/lr_structured_feature_names.csv
    outputs/preprocessors/lr_text.joblib
    outputs/preprocessors/lr_text_feature_names.csv
    outputs/preprocessors/tree_text.joblib
    outputs/matrices/labels_train.csv
    outputs/matrices/labels_validation.csv
    outputs/matrices/X_raw_train.csv
    outputs/matrices/X_raw_validation.csv
    outputs/matrices/lr_structured_train.npy
    outputs/matrices/lr_structured_validation.npy
    outputs/matrices/lr_text_train.npz
    outputs/matrices/lr_text_validation.npz
  person3_handoff/...
```

From `ML For Health`, use a dedicated environment:

```sh
python3 -m venv .venv-person3
.venv-person3/bin/python -m pip install -r "Group Project/person3_handoff/requirements.txt"
.venv-person3/bin/python "Group Project/person3_handoff/person3_baselines.py" --project-dir "Group Project"
```

Or select that environment as the notebook kernel and run all cells. The notebook
includes all Person 3 workflow definitions, so the companion executable module
is optional for notebook execution. The shared preprocessing module must stay
available. `aki_preprocessing.py` included here is a byte-identical copy of Person
2's module; upstream files were not changed. Re-execution overwrites only generated
Person 3 outputs and the report contribution.

To rebuild and execute the notebook from the source module:

```sh
.venv-person3/bin/python "Group Project/person3_handoff/build_notebook.py" --execute
```

Independently check the saved probabilities, metrics, predictor isolation and
notebook execution without fitting models again:

```sh
.venv-person3/bin/python "Group Project/person3_handoff/verify_handoff.py" --project-dir "Group Project"
```

Python 3.12.14 was used locally; Person 2 used 3.13.9. The scientific packages
match Person 2: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2 and
joblib 1.5.2. Saved preprocessing was checked against every training/validation
matrix value, and reloaded models reproduce validation probabilities exactly.
`outputs/environment_freeze.txt` records the complete local environment.

The text CSR matrices have the same nonzero pattern and structured values as the
local transform. Sparse TF-IDF normalisation can differ across platforms at machine
precision (maximum observed absolute difference 2.22e-16); the check uses atol 1e-14.
The classifier fits the supplied training matrix, while validation uses the frozen
preprocessor's local transform so exported probabilities match the saved raw-input
pipeline exactly. This does not change the vocabulary, IDF, preprocessing settings
or zero/missing-value semantics. Numerical checks are recorded in model_metadata.json.

## Pipeline use

The models accept Person 1's 13 cleaned input columns, in a pandas DataFrame.
No cleaning from raw BigQuery records is performed by these pipelines; that is
the upstream extraction responsibility. Models return probabilities for labels
in `model.classes_`, with positive probability in column 1 for this binary task.

```python
import joblib
import pandas as pd

model = joblib.load("person3_handoff/outputs/models/structured_lr.joblib")
X_validation = pd.read_csv(
    "person2_handoff/person2_split_preprocessing/outputs/matrices/X_raw_validation.csv"
).drop(columns="stay_id")
p_validation = model.predict_proba(X_validation)[:, 1]
```

Use the pinned scientific environment when loading pipelines, consistent with
[scikit-learn's persistence guidance](https://scikit-learn.org/1.7/model_persistence.html).

## Outputs and prediction schema

| File | Content |
|---|---|
| `outputs/models/prevalence.joblib` | Fitted constant-probability Pipeline |
| `outputs/models/age_gender_lr.joblib` | Fitted age/gender Pipeline |
| `outputs/models/structured_lr.joblib` | Selected fitted structured Pipeline |
| `outputs/models/structured_text_lr.joblib` | Selected fitted structured-plus-text Pipeline |
| `outputs/validation_predictions.csv` | 67,728 rows: 16,932 encounters x 4 models |
| `outputs/validation_metrics.csv` | AP, AUROC, Brier, target, partition and denominators |
| `outputs/tuning_record.csv` | Fixed age/gender fit plus all six C settings for each regularised LR |
| `outputs/handoff_checks.csv` | Input, leakage, alignment, pipeline and CSV checks |
| `outputs/analysis_plan.json` | Bounded specification saved before tuning |
| `outputs/model_metadata.json` | Both selected C values, text specification, environment and upstream SHA-256 hashes |
| `outputs/coefficients.csv` | LR coefficients and intercepts, for development inspection |
| `outputs/calibration_bins.csv` | Plot coordinates for calibration bins |
| `outputs/figures/validation_discrimination_calibration.png` | Validation PR, ROC and calibration curves |
| `outputs/independent_validation_checks.json` | Independent probability, metric and input-isolation checks |
| `outputs/package_portability_check.json` | Fresh-directory model reload and archive integrity checks |
| `reference_baselines_v1.json` | Original three-model prediction fingerprints and metrics |
| `CHANGELOG_P3.md` | Scope and handoff changes in version 2 |

`../person3_handoff.zip` contains this folder for local/private team handoff. It
does not duplicate the original input data or Person 2's matrices; teammates need
their existing upstream files to rerun model development. Its predictions are
row-level MIMIC-derived information and follow the same private-storage rule.

The prediction CSV uses this proposed machine-readable naming for the seven
fields requested by the methodology:

```text
stay_id,subject_id,target,observed_label,predicted_probability,model_name,partition
```

Every target is `aki_label`, every partition is `validation`, and observed labels
are 0/1. Model names are the four names in the model table above. Person 5 can
rename fields to match shared code, preserving their meaning and identifier order.
Read probabilities with `pd.read_csv(..., float_precision="round_trip")` when
checking bit-for-bit correspondence with saved model output.

## Pending shared evaluation

AP is the provisional Person 3 development metric, following the methodology's
recommendation; Person 5 has not yet provided shared evaluation code. AP is
calculated with `average_precision_score` and is distinct from trapezoidal PR
area. See the [official AP definition](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html).

No decision threshold has been chosen. `DECISIONS_P3.md` proposes, for discussion,
a validation-selected threshold achieving at least 90% sensitivity; the team has
not agreed that value. No threshold metrics, recalibration, confidence intervals
or test predictions are included.

The four baselines are ready for review. Person 3 development settings are
recorded, while the team-wide freeze awaits coordination with Persons 4 and 5.
Person 5 conducts final test evaluation and patient-level paired bootstrap
comparisons after that freeze.

Use Person 4's corrected XGBoost outputs for the text/algorithm comparison.
The supplied 15 October P4 handoff still uses the original sparse text input;
the structured-zero correction and affected reruns are Person 4's responsibility.
The text-LR development does not depend on that correction. Chief-complaint
terms may capture conditions already suspected at presentation. Their associations
do not show prediction before clinical suspicion, and coefficients are not causal.

Row-level predictions, upstream data and fitted models must remain in approved
private storage, as required by the existing team handoff. Only aggregate results
and figures belong in the shared methods document.
