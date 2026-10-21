# Person 3 report-ready contribution

English contribution for team editing, 20 October 2026. The primary results below come from Person 5's frozen **test** evaluation. The secondary paragraph is a **validation-only** development note and must be replaced or supplemented after Person 5 completes secondary test evaluation. Do not paste every supporting note into the final report.

## Baseline methods

We developed four baseline models using the frozen patient-separated training and validation partitions: training prevalence, age/gender logistic regression (LR), structured LR, and structured-plus-chief-complaint LR. The constant model predicted the training encounter prevalence. Age/gender LR used standardised age and a male indicator without regularisation. Both multivariable models used training-median imputation, standardisation, missingness indicators and one-hot encoding, yielding 27 structured features. The text model added 2,772 training-derived TF-IDF unigram and bigram features. The text vocabulary, IDF weights and settings were shared with XGBoost. Each multivariable LR independently selected its L2 penalty from C = 0.001, 0.01, 0.1, 1, 10 and 100 by maximum validation average precision (AP); ties within 1e-12 favoured smaller C. All models were fitted on training data without class weighting or resampling. The selected primary C values were 10 for structured LR and 0.1 for text-inclusive LR.

## Primary test results

The test set contained 16,801 encounters from 9,914 patients, including 1,155 primary AKI outcomes (6.87%). Structured LR achieved AP 0.1109, AUROC 0.6224 and Brier score 0.06311. Adding chief-complaint text increased AP to 0.1328 (95% patient-cluster bootstrap CI 0.1202–0.1495), with AUROC 0.6499 and Brier score 0.06245. The paired AP difference was 0.0219 (95% CI 0.0134–0.0326). Intervals used 1,000 patient-cluster resamples with identical draws across models, conditional on the fitted models.

## Interpretation

Chief-complaint text improved discrimination within LR, although overall discrimination remained modest. Complaints may contain renal diagnoses, abnormal laboratory findings or suspicions already known at arrival; their associations do not establish prediction before clinical suspicion. Operational negatives may include unobserved disease. Threshold-based utility should be judged using the shared test evaluation, including sensitivity, precision and the proportion flagged, rather than AP alone.

## Secondary methods addition

For the secondary Stage ≥2 endpoint, all four baselines were newly fitted using the existing Stage 2-or-3 label, unchanged patient partitions and preprocessing. Stage 1 and operational negatives remained in the full cohort as label 0. Regularised models independently repeated the same bounded validation AP search. Selected C values were 100 for structured LR and 1 for text-inclusive LR. Implementation followed completion of the primary test evaluation; the secondary development plan preceded secondary tuning.

## Secondary development note — not final test results

Validation included 333 secondary positives among 16,932 encounters (1.97%). Structured LR achieved AP 0.0430 and AUROC 0.6263; text-inclusive LR achieved AP 0.0721 and AUROC 0.6659. These estimates were used for model selection and have no patient-level confidence intervals in this handoff. Secondary test evaluation is pending.

## Optional primary table rows

These are Person 5's test point estimates. In the final combined model table, use its confidence intervals and include the XGBoost rows too.

| Model | Test AP | Test AUROC | Test Brier |
|---|---:|---:|---:|
| Training prevalence | 0.0687 | 0.5000 | 0.06402 |
| Age/gender LR | 0.0739 | 0.5331 | 0.06397 |
| Structured LR | 0.1109 | 0.6224 | 0.06311 |
| Structured + text LR | 0.1328 | 0.6499 | 0.06245 |

## Integration notes — exclude from report prose

- Keep the final report within the course word limit; the above is a contribution, not a complete report.
- Person 5's final primary operating-point rule was the largest validation cutoff achieving at least **80% sensitivity**, retaining ties. Earlier Person 3 proposals of 90% were not adopted. Test sensitivity need not reach the validation target.
- The main results should use Person 5's test estimates and clustered uncertainty. Person 3's original validation numbers belong in development/tuning details.
- Avoid treating secondary AP as directly comparable with primary AP: the outcome prevalence differs.
- Add the team's agreed references and CRediT contribution table in the combined report. Suggested baseline roles: Methodology, Software, Formal analysis, Validation, Visualisation, Writing – original draft.
- Source tables and checksums are recorded in `report_source_provenance.json`; the original Person 5 files were read without modification. No secondary test inference was performed.
