# Person 3 development decisions

Original plan recorded 13 October 2026 before baseline tuning. Text-LR extension
recorded 16 October 2026 before inspecting its tuning results.

The original stage covered only the primary outcome and three basic baselines.
On 16 October, following the team's request, the user authorised adding structured
plus chief-complaint LR using Person 2's existing lr_text preprocessing. Stage >=2
and sensitivity models remain deferred. The original handoff is preserved as
`../person3_handoff_v1_2026-10-13.zip`, with reference fingerprints in
`reference_baselines_v1.json`.

| Decision | Choice | Status |
|---|---|---|
| Cohort and labels | Revised Person 1/2 cohort; primary aki_label | Existing team specification |
| Split | Reuse Person 2's patient-separated manifest, seed 90089 | Existing team specification |
| Preprocessing | Reuse fitted lr_structured and lr_text unchanged | Existing team specification |
| Constant comparator | Training encounter prevalence, DummyClassifier prior | Person 3 implementation |
| Age/gender comparator | Unpenalised LR; shared age scaling and male one-hot column | Person 3 implementation |
| Structured comparator | L2 LR; all 27 transformed structured features | Person 3 implementation |
| Structured + text comparator | L2 LR; the same 27 columns plus 2,772 frozen TF-IDF terms | Authorised 16 October |
| Tuning search | C = 0.001, 0.01, 0.1, 1, 10, 100; no search expansion after results | Fixed before tuning |
| Text optimisation | lbfgs, tol 1e-8, max_iter 20000; select its own C using the same grid and AP rule | Initial limit revised for convergence only; grid/rule unchanged |
| Selection metric | Validation AP; ties within 1e-12 choose smaller C | Provisional; Person 5 agreement pending |
| Supporting validation metrics | AUROC, Brier, quantile calibration plot | Development assessment |
| Weighting and sampling | No class weighting, patient weighting or resampling | Person 3 implementation |
| Final fit | Training only; validation remains development data | Person 3 implementation |
| Thresholds | None selected and no threshold metrics generated | Team agreement pending |
| Calibration | Assess original probabilities; no recalibration | Team agreement pending |
| Confidence intervals | Patient bootstrap and paired comparison delegated to Person 5 | Existing team responsibility |
| Final test | Pending team model freeze; no test data opened by P3 code | Existing team responsibility |

The notebook and executable module implement the same workflow. The appropriate fitted
Person 2 preprocessor is retained in each pipeline, but the age/gender pipeline
passes only age and the male indicator to its classifier. The prevalence classifier
ignores predictor values. Selecting shared transformed columns is not a change to
Person 2's preprocessing.

## Text representation and comparison scope

The text-inclusive LR keeps training-median imputation, numeric scaling,
missingness indicators and one-hot encoding unchanged. It uses Person 2's existing
lowercased TF-IDF with 1-2 grams, min_df 5, sublinear term frequency and the original
tokenizer. Both vocabulary and IDF weights are frozen from training text. Verify
that these settings, terms and weights equal the existing tree_text preprocessor.
No XGBoost-specific sparse-zero correction is applied to LR.

Pre-fit reproduction checks found a maximum absolute 2.22e-16 difference in sparse
TF-IDF normalisation across platforms, with identical nonzero positions, exact
structured values, vocabulary and IDF. This is checked with atol 1e-14. The model
fits the supplied training matrix; prediction uses the original fitted
preprocessor's local transform, matching raw-input pipeline inference. These
numerical choices were recorded before text-LR tuning results were inspected.

The original three baselines retain their settings and are checked against the
v1 prediction and metric fingerprints. The updated output adds one model named
`structured_text_lr`; all four models use the same 16,932 validation encounters.
Each regularised LR selects its own C from the identical bounded grid. The full
tuning table also allows a common-C comparison without extra fitting.

The initial 5,000-iteration text optimisation limit was reached by C = 100 without
convergence. The final limit is 20,000 for every text candidate; tolerance 1e-8,
solver, six C values and AP selection rule are unchanged. This is a numerical
convergence repair, not an expansion of the statistical model search. The full
final tuning record reports each candidate's actual iteration count.

Person 4 must correct structured-zero handling in text-inclusive XGBoost and
provide updated tuning, predictions, selected model and feature importance.
Until then, its earlier text-inclusive comparison is not used as definitive
evidence for a text or algorithm gain. The numeric preprocessing remains
appropriate to each algorithm while the text representation and cohort match.
Chief complaints may contain information already known at presentation; describe
predictive associations and early identification, not prediction before suspicion.

## Proposed evaluation protocol for Person 5

Use AP as the primary ranking metric, and AUROC, Brier and calibration as supporting
metrics, consistently for LR and XGBoost. AP must be distinguished from trapezoidal
PR area. Natural outcome frequencies must be retained. Define model success against
the training-prevalence comparator and assess the practical false-positive burden;
do not infer clinical utility solely from a higher AP.

A candidate threshold rule is the largest model-specific cutoff achieving at least
90% sensitivity on validation, using prediction >= cutoff and retaining all ties.
The 90% value is a proposal, not an agreed clinical requirement; it can be changed
before testing. A constant baseline cannot provide a useful graded operating point.
If the team adopts a rule, record the cutoff, sensitivity, specificity, PPV, NPV and
fraction flagged for each model and freeze the rule before test evaluation.

After all model choices are fixed, Person 5 should evaluate the same test encounters
with patient-level bootstrap confidence intervals and paired resamples. Sensitivity
subsets should be agreed with Person 2 before test inspection. Any changes to the
development metric or preprocessing should be recorded here and in the team's
shared decision log before rerunning development models.
