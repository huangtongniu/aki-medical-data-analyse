# Secondary analysis decisions

Recorded 20 October 2026, before secondary model tuning.

1. Add all four Person 3 baseline families for the existing secondary `aki_stage2_label` endpoint: training prevalence, age/gender LR, structured LR and structured-plus-text LR.
2. Use the complete existing modelling cohort, frozen patient split and train-only fitted Person 2 preprocessing. Stage 1 is a secondary negative; do not restrict to primary-positive encounters.
3. Training: 79,364 encounters / 46,263 patients / 1,589 secondary positives. Validation: 16,932 encounters / 9,913 patients / 333 secondary positives. Secondary test outcomes are not opened here.
4. Use the same six-value C grid and maximum validation AP rule as primary baseline development. Independently tune each regularised model to the secondary labels. No class weights, resampling, candidate expansion or train-plus-validation refitting.
5. Use lbfgs, tol 1e-8; max_iter 2,000 for age/gender and structured LR, 20,000 for text LR. A nonconvergent fit stops execution. Retain all candidates' results.
6. Save a distinct package; do not overwrite primary models, thresholds, predictions, metadata or Person 5 freeze records. Each saved pipeline explicitly identifies its target.
7. No secondary threshold, patient-bootstrap interval or test score is produced by Person 3. Person 5 should freeze the secondary evaluation protocol before accessing secondary test outcomes. The primary 80% validation sensitivity rule can be proposed for consistency, but primary probability cutoffs cannot be reused.
8. The secondary endpoint predates this implementation in the team methodology. Implementation and this development plan follow the already completed primary test evaluation. Do not claim the secondary model-development freeze predates all test-result inspection.
9. Person 4 owns Stage >=2 XGBoost and its sparse-zero representation. LR retains the existing lr_text representation. Shared encounters, endpoint, split, vocabulary, TF-IDF settings and IDF must align between algorithms.
10. Validation performance is a development estimate. Secondary AP is prevalence-dependent and is not directly comparable with primary AP as a measure of task difficulty. Chief-complaint features describe predictive associations, potentially including information already suspected at presentation.
