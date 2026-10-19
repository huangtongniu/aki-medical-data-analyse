# Person 3 baseline methods and validation results

## Methods contribution

Four baseline models were developed for the primary outcome, observed creatinine-defined AKI
within 24 hours of ED arrival or earlier hospital discharge. All models used Person 2's fixed
patient-separated partitions (seed 90089), with 79,364 training encounters from 46,263 patients
and 16,932 validation encounters from 9,913 patients. Person 2's lr_structured and lr_text
preprocessing were reused without change:
training-median numeric imputation, numeric standardisation, missingness indicators, and
one-hot categorical encoding. Identifiers, laboratory measurements and ascertainment fields
were excluded from model inputs. The fitted preprocessing was verified against the saved
training and validation matrices.

The prevalence baseline assigned every encounter the training outcome frequency
(6.9881%). An unpenalised logistic regression used standardised age and
the male-gender indicator only, with female as reference. Structured logistic regression used
all 27 transformed structured features and an L2 penalty. Six regularisation values
(C = 0.001, 0.01, 0.1, 1, 10, 100) were compared, selecting maximum validation average
precision (AP); ties within 1e-12 favoured stronger regularisation. The selected C was
10 for structured LR. The additional structured-plus-text model used
the same 27 structured columns plus 2,772 frozen training TF-IDF terms, for 2,799 columns.
Text was lowercased and represented by 1-2 grams with min_df 5 and sublinear term frequency,
using Person 2's tokenizer. The vocabulary, IDF weights and settings were verified to match
Person 2's tree_text representation. Its C was selected independently from the same six
values using the same AP rule; the selected value was 0.1. Both
LR models used lbfgs with tol 1e-8; iteration limits were 2,000 for structured LR and 20,000
for text LR. The text limit was increased from 5,000 after the first C = 100 fit
did not converge; no candidate values, tolerance or selection rules were changed.
Models were fitted on training data only, without resampling or
class weighting. AP, AUROC, Brier score and calibration curves were calculated on validation
data. AP is the recall-increment-weighted metric implemented by average_precision_score,
not a trapezoidal precision–recall area. All logistic regressions converged.

## Validation results contribution

The validation set contained 1,165 positive encounters (6.8805%). The prevalence baseline
achieved AP 0.0688, AUROC 0.5000, and Brier score
0.0641. Age/gender logistic regression achieved AP 0.0732,
AUROC 0.5281, and Brier score 0.0640. Structured logistic regression
achieved AP 0.1170, AUROC 0.6223, and Brier score
0.0630. Structured-plus-text LR achieved AP 0.1475,
AUROC 0.6632, and Brier score 0.0621. The within-LR text
comparison changed AP by +0.0305
and AUROC by +0.0410, with a Brier-score change of
-0.00090. See outputs/validation_metrics.csv
and the validation figure for complete values. These are encounter-level development
results: both regularised LR models were selected using this validation set, so their
performance is not an independent
final estimate. Repeated encounters also require patient-level resampling for uncertainty
estimates; confidence intervals have not been calculated in this handoff.

## Interpretation and work remaining

Age/gender alone added little discrimination (AUROC 0.528). Structured predictors
increased AP to 1.70 times the
prevalence baseline, but AUROC 0.622 still indicates modest discrimination.
The Brier reduction was only 0.00103 in absolute
terms (1.60%
relative). The text comparison keeps the structured columns, outcome, encounters and
partition fixed, while allowing each model's penalty to be selected by the same predefined
search. It assesses adding text within the LR modelling approach. A common-C comparison
is also available in outputs/tuning_record.csv. It does not establish clinical utility
or a statistically reliable improvement. In the original structured search, AP for
C = 10 and 100 differed by less than 0.000001; selection followed the recorded rule
and should not be interpreted as evidence for a distinct optimal penalty. Calibration
plots summarise quantile groups and lack uncertainty intervals; useful probability
calibration and clinical operating thresholds still require shared evaluation.

This study identifies observed AKI,
including cases already positive at the first sample, in a selected outcome-assessable
admitted population. Operational negatives do not establish clinical absence of AKI.
Regression coefficients describe associations and depend on coding and regularisation.
Chief complaints may contain renal diagnoses, abnormal laboratory findings or clinical
suspicions already known at presentation. Their predictive association does not show that
the model anticipates AKI before any clinical suspicion. Early identification is the
appropriate task description.

Comparison with text-inclusive XGBoost must use Person 4's corrected input representation
and updated results; the older dense-versus-sparse comparison also changed structured-zero
handling. No such correction is needed for LR. Algorithm-specific numeric preprocessing
can differ, while encounters, targets and the frozen text representation remain aligned.

Person 5 should agree the final metric and threshold protocol with Persons 3 and 4 before
final test evaluation. AP is currently a provisional Person 3 tuning choice. No threshold
has been selected, no probability recalibration has been fitted, and no test predictors,
labels, matrices or predictions were opened or generated by this code. Stage >=2 and
sensitivity evaluations remain outside the present scope. The text-LR addition was
authorised on 16 October 2026 following the team discussion; the earlier three-model
handoff is retained separately, and baseline preservation checks compare its original
prediction and metric fingerprints with the updated outputs.

## Implementation references

- [scikit-learn 1.7.2 LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn 1.7.2 average_precision_score](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html)
- [scikit-learn 1.7.2 DummyClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.dummy.DummyClassifier.html)

Proposed CRediT contribution: Methodology, Software, Formal analysis, Validation,
Visualisation, Writing – original draft (baseline methods and results).
