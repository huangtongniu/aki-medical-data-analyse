# Stage >=2 integration for Person 5

The supplied Person 5 package evaluates **seven primary models using `aki_label`**. It does not yet evaluate newly trained secondary models. Preserve that completed evaluation and its freeze records.

## Person 3 inputs

- Outcome: `aki_stage2_label`, existing Person 1/2 labels, same full cohort. Stage 2 or 3 = 1; Stage 1 and operational negative = 0.
- Models: `prevalence`, `age_gender_lr`, `structured_lr`, `structured_text_lr` in this package's `outputs/models`.
- Raw predictors: exactly Person 2's 13 cleaned columns, excluding stay_id, subject_id and labels. `aki_preprocessing.py` must be importable when unpickling.
- Pipelines include preprocessing; do not preprocess twice or substitute tree matrices. `pipeline.target_name_` is `aki_stage2_label`.
- Validation export: `stay_id,subject_id,target,observed_label,predicted_probability,model_name,partition`. Exactly 16,932 rows per model, 333 positives, 67,728 rows overall.
- Same primary model names are retained. Join/group by **target + model_name** as well as IDs and partition, and use distinct output directories for each outcome.
- `model_metadata.json` records source hashes and selected penalties. All four pipelines remain training-only fits.

## Before secondary test evaluation

1. Obtain Person 4's newly fitted Stage >=2 XGBoost models and the matching inference code/representation. Also obtain the **corrected primary** XGBoost source and model used in the existing Person 5 evaluation: the older local Person 4 text model does not match its frozen hash.
2. Adapt target selection, labels, model paths, validation references, report labels and bootstrap outputs in the primary-only evaluator. Do not merely rename primary predictions or score primary-trained models against the secondary labels as substitutes for the new models.
3. Reproduce secondary validation predictions by stay_id and subject_id; verify the secondary labels and split match between Persons 3/4. Confirm identical text vocabulary, settings and IDF, with appropriate algorithm-specific numeric preprocessing.
4. Freeze a separate secondary protocol before secondary test evaluation: AP as the main metric, AUROC/Brier/calibration, a validation-only operating threshold rule, patient-cluster resampling with identical patient draws for paired comparisons, comparison list and any sensitivity subsets. Reusing the *rule* of 80% validation sensitivity is a consistency proposal; each new secondary model requires its own validation cutoff. Record who agrees the protocol.
5. Save the new freeze and results under a separate secondary directory. Keep the primary source-hash checks and freeze artifacts intact. Handle bootstrap draws without both outcome classes explicitly and report valid replicate counts.
6. Report secondary results in their own table with the secondary prevalence. Use test estimates and patient-level intervals for the final report; this handoff's validation results are development estimates.

## Final notebook

For the single submission notebook, include the complete acquisition/phenotype, split/preprocessing, baseline, XGBoost and evaluation code in a consistent order. Expose input/output paths and environment requirements in setup. Reproduction should write to a fresh output directory, preserving the archived primary evaluation. Keep recorded hyperparameter searches/selection rules and fixed validation thresholds distinct from evaluation; no tuning should respond to test scores.

This is an integration note, not an update to Person 5's evaluator. No shared or frozen primary files have been edited by the secondary workflow.
