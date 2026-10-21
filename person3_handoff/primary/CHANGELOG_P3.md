# Person 3 handoff changes

## Version 2 dated 16 October 2026

Added `structured_text_lr` for the primary AKI outcome after the team requested a
text-inclusive LR comparator and the user authorised the extension. The original
three baselines and their modelling settings are retained.

- Reused Person 2's fitted lr_text transformer and saved training/validation CSR matrices.
- Verified the 27 structured columns against lr_structured and the 2,772 TF-IDF terms,
  tokenizer settings and IDF weights against tree_text.
- Defined the text model's L2 C grid as 0.001, 0.01, 0.1, 1, 10, 100 before tuning.
  Selection uses validation AP, with ties within 1e-12 favouring smaller C. Optimisation
  uses lbfgs, tol 1e-8 and max_iter 20000. No class weighting or resampling is used.
  The initial max_iter 5000 was insufficient for C = 100; only the numerical
  iteration budget was increased, retaining all candidate values and the AP rule.
- Added the fitted raw-input text Pipeline, tuning rows and validation probabilities.
  The combined prediction export now has four models and 67,728 rows, using the same
  seven-column schema and identical 16,932 validation encounters per model.
- Updated the executed notebook, validation plots, coefficients, report contribution,
  metadata and handoff instructions. Added original-baseline preservation checks.
- Retained AP as the provisional development metric. Thresholds, recalibration,
  confidence intervals and final test evaluation remain for shared evaluation.
- Did not modify Person 1/2's data, labels, split or preprocessing, Person 4's files,
  or Project Method.docx. Stage >=2 and sensitivity work remain deferred.

The old handoff is archived separately as `person3_handoff_v1_2026-10-13.zip` in the
parent folder. `reference_baselines_v1.json` records its prediction fingerprints,
metrics and archive SHA-256. The current `person3_handoff.zip` replaces the previous
ZIP for Person 3 outputs; it does not duplicate the upstream data or P2 matrices.

Person 4 is independently correcting XGBoost structured-zero handling. Its older
text-inclusive results should be replaced with the corrected run before final
text/algorithm comparisons. The LR text representation needs no such correction.
