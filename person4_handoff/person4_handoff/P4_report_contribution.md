# Person 4 XGBoost methods and validation results

## Methods contribution

XGBoost models were developed for the primary outcome of observed creatinine-defined AKI within 24 hours of ED arrival or earlier hospital discharge. All models used Person 2's fixed patient-separated training and validation partitions. The training set contained 79,364 encounters and the validation set contained 16,932 encounters.

Person 2's tree-model preprocessing was reused. Numeric missing values were retained as NaN so that XGBoost could handle missingness internally, while categorical variables were represented using the saved one-hot encoded features. For the structured-plus-chief-complaint model, Person 2's existing chief-complaint vocabulary and TF-IDF representation were retained. However, the structured feature block was reconstructed so that zero values were explicitly preserved in the sparse matrix rather than being treated as absent entries. The reconstructed structured values were checked against the original dense matrices, and both zero values and NaN values were preserved correctly.

Three predictor groups were compared on the same training and validation encounters: age and gender only, 19 structured features, and 2,791 transformed structured-plus-chief-complaint features. XGBoost models used a binary logistic objective and histogram-based tree construction. Average precision (AP) was used as the primary model-selection metric, with AUROC and Brier score reported as supporting metrics.

A limited hyperparameter search was performed using development data only. For the structured model, combinations of tree depth, learning rate and number of estimators were evaluated, while subsample and column subsampling were fixed at 0.8. The selected structured model used 400 estimators, maximum depth 3 and learning rate 0.03. The predictor groups were then compared using these fixed XGBoost settings.

Because the structured-plus-chief-complaint model showed the strongest validation performance, a further limited search was performed for this model. The selected configuration used 300 estimators, maximum depth 3, learning rate 0.05, subsample 0.8 and column subsampling 0.8. No test data were used during model development or hyperparameter selection.

## Validation results contribution

Using the fixed XGBoost settings, the age/gender predictor group achieved AP 0.0745, AUROC 0.5311 and Brier score 0.0640. The structured predictor model achieved AP 0.1248, AUROC 0.6459 and Brier score 0.0627.

After correcting the sparse representation so that structured zero values were preserved consistently, the structured-plus-chief-complaint model achieved AP 0.1481, AUROC 0.6712 and Brier score 0.0620.

After limited model-specific tuning, the final structured-plus-chief-complaint XGBoost model achieved AP 0.1483, AUROC 0.6727 and Brier score 0.0620 on the validation set.

For the structured-only models, structured XGBoost achieved AP 0.1248, AUROC 0.6459 and Brier score 0.0627, compared with AP 0.1170, AUROC 0.6223 and Brier score 0.0630 for structured logistic regression.

For the text-inclusive models, the final XGBoost model achieved AP 0.1483, AUROC 0.6727 and Brier score 0.0620, compared with AP 0.1475, AUROC 0.6632 and Brier score 0.0621 for text-inclusive logistic regression.

XGBoost therefore improved performance over logistic regression for both the structured-only and text-inclusive comparisons. Adding chief-complaint information produced a larger improvement in predictive performance for both modelling approaches.

## Interpretation and work remaining

The predictor-group comparison suggests that richer clinical information contributed substantially to predictive performance. Age and gender alone provided limited discrimination, while structured features improved both AP and AUROC. After keeping the representation of the structured predictors consistent, adding chief-complaint features further improved XGBoost performance.

Feature-importance analysis of the final model showed that many highly ranked features were derived from chief-complaint text. Terms such as labs, cardiac arrest, renal failure, abnormal labs, fistula, hypotension and hyperkalemia appeared among the highest-ranked features. Structured predictors including systolic blood pressure, arrival transport, triage acuity, respiratory rate, diastolic blood pressure and gender also appeared among the top features.

Gain-based feature importance shows how useful a feature was for prediction within the fitted model, but it does not show whether the feature increased or decreased the predicted risk. Some chief-complaint terms may also reflect clinical information or concerns already present when the patient arrived at the ED. These findings therefore indicate which features contributed to the model's predictions, but they do not show a causal relationship.

The final XGBoost configuration should now be frozen before independent test-set evaluation. Validation performance should still be treated as a development estimate because both predictor selection and hyperparameter selection used the validation partition.
