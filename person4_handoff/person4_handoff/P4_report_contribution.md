# Person 4 XGBoost methods and validation results

## Methods contribution

XGBoost models were developed for both the primary Stage ≥1 AKI outcome and the secondary Stage ≥2 AKI outcome. All models used Person 2's fixed patient-separated training and validation partitions. The training set contained 79,364 encounters and the validation set contained 16,932 encounters.

Person 2's tree-model preprocessing was reused. Numeric missing values were retained as NaN so that XGBoost could handle missingness internally, while categorical variables were represented using the saved one-hot encoded features. Person 2's existing chief-complaint vocabulary and TF-IDF representation were also retained.

For the structured-plus-chief-complaint models, the structured feature block was reconstructed so that zero values were explicitly preserved in the sparse matrix rather than being treated as absent entries. The reconstructed structured values were checked against the original dense matrices, and both zero values and NaN values were preserved correctly. The same corrected representation was used for both Stage ≥1 and Stage ≥2 models.

For each outcome, three predictor groups were compared on the same training and validation encounters: age and gender only, 19 structured features, and 2,791 transformed structured-plus-chief-complaint features. XGBoost models used a binary logistic objective and histogram-based tree construction. Average precision (AP) was used as the primary model-selection metric, with AUROC and Brier score reported as supporting metrics.

For the primary Stage ≥1 outcome, a limited hyperparameter search was first performed for the structured model. Combinations of tree depth, learning rate and number of estimators were evaluated, while subsample and column subsampling were fixed at 0.8. The selected structured model used 400 estimators, maximum depth 3 and learning rate 0.03.

The predictor groups were then compared using these fixed XGBoost settings. Because the structured-plus-chief-complaint model showed the strongest validation performance, a further limited search was performed for this model. The selected Stage ≥1 configuration used 300 estimators, maximum depth 3, learning rate 0.05, subsample 0.8 and column subsampling 0.8.

The same development procedure was repeated for the Stage ≥2 secondary outcome using the same patient partitions and predictor matrices. The selected Stage ≥2 structured model used 200 estimators, maximum depth 4 and learning rate 0.05. After predictor-group comparison, the structured-plus-chief-complaint model was tuned further. The selected final Stage ≥2 configuration used 300 estimators, maximum depth 3, learning rate 0.05, subsample 0.8 and column subsampling 0.8.

No test data were used during model development or hyperparameter selection for either outcome.

## Stage ≥1 validation results

Using the fixed XGBoost settings, the age/gender predictor group achieved AP 0.0745, AUROC 0.5311 and Brier score 0.0640. The structured predictor model achieved AP 0.1248, AUROC 0.6459 and Brier score 0.0627.

After correcting the sparse representation so that structured zero values were preserved consistently, the structured-plus-chief-complaint model achieved AP 0.1481, AUROC 0.6712 and Brier score 0.0620.

After limited model-specific tuning, the final Stage ≥1 structured-plus-chief-complaint XGBoost model achieved AP 0.1483, AUROC 0.6727 and Brier score 0.0620 on the validation set.

For the structured-only models, XGBoost achieved AP 0.1248, AUROC 0.6459 and Brier score 0.0627, compared with AP 0.1170, AUROC 0.6223 and Brier score 0.0630 for structured logistic regression.

For the text-inclusive models, XGBoost achieved AP 0.1483, AUROC 0.6727 and Brier score 0.0620, compared with AP 0.1475, AUROC 0.6632 and Brier score 0.0621 for text-inclusive logistic regression.

## Stage ≥2 validation results

The Stage ≥2 outcome was less common, with 1,589 positive encounters in training (2.00%) and 333 positive encounters in validation (1.97%).

Using the selected structured-XGBoost settings, the Stage ≥2 age/gender model achieved AP 0.0260, AUROC 0.5640 and Brier score 0.0193. The structured model achieved AP 0.0564, AUROC 0.6836 and Brier score 0.0190. Adding chief-complaint features improved performance to AP 0.0761, AUROC 0.7014 and Brier score 0.0188 before additional model-specific tuning.

After limited tuning, the final Stage ≥2 structured-plus-chief-complaint XGBoost model achieved AP 0.0763, AUROC 0.6975 and Brier score 0.0188.

For the Stage ≥2 structured-only models, XGBoost achieved AP 0.0564, AUROC 0.6836 and Brier score 0.0190, compared with AP 0.0430, AUROC 0.6263 and Brier score 0.0191 for structured logistic regression.

For the Stage ≥2 text-inclusive models, XGBoost achieved AP 0.0763, AUROC 0.6975 and Brier score 0.0188, compared with AP 0.0721, AUROC 0.6659 and Brier score 0.0189 for text-inclusive logistic regression.

Across both outcomes, XGBoost achieved higher validation performance than the corresponding logistic-regression models, while the inclusion of chief-complaint information produced a substantial additional improvement.

## Interpretation and work remaining

For the Stage ≥1 outcome, the predictor-group comparison suggests that richer clinical information contributed substantially to predictive performance. Age and gender alone provided limited discrimination, while structured features improved both AP and AUROC. After keeping the representation of the structured predictors consistent, adding chief-complaint features further improved XGBoost performance.

Feature-importance analysis of the final Stage ≥1 model showed that many highly ranked features were derived from chief-complaint text. Terms such as labs, cardiac arrest, renal failure, abnormal labs, fistula, hypotension and hyperkalemia appeared among the highest-ranked features. Structured predictors including systolic blood pressure, arrival transport, triage acuity, respiratory rate, diastolic blood pressure and gender also appeared among the top-ranked features.

For the Stage ≥2 model, highly ranked chief-complaint features included fistula, labs, renal failure, renal, clotted, urinary retention, abnormal labs, hyperkalemia and hypotension. Structured predictors including systolic blood pressure, gender, arrival transport and age also appeared among the top-ranked features.

Gain-based feature importance shows how useful a feature was for prediction within the fitted model, but it does not show whether the feature increased or decreased the predicted risk. Some chief-complaint terms may also reflect clinical information or concerns already present when the patient arrived at the ED. These findings therefore indicate which features contributed to the models' predictions, but they do not show a causal relationship.

The final Stage ≥1 and Stage ≥2 XGBoost configurations should now be frozen before independent test-set evaluation. Validation performance should still be treated as a development estimate because predictor selection and hyperparameter selection used the validation partition.
