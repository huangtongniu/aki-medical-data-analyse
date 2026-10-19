"""Create standalone scientific figures and a report from frozen evaluation outputs."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

from common import (CHECKS, FIGURES, HERE, MODEL_LABELS, MODEL_NAMES, P1, P2, P2_QA,
                    P3, P4, TABLES, environment, make_output_dirs, read_json, save_csv,
                    settings, write_json)

COLORS = {"prevalence": "#747474", "age_gender_lr": "#C2813B", "structured_lr": "#21857C",
          "structured_text_lr": "#7559A6", "age_gender_xgboost": "#B59D27",
          "structured_xgboost": "#3D75A5", "structured_chief_complaint_xgboost": "#BB4F60"}
MAIN_MODELS = ["prevalence", "structured_lr", "structured_text_lr", "structured_xgboost", "structured_chief_complaint_xgboost"]


def finish_figure(fig, stem):
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{stem}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(headers, rows):
    def cell(x):
        return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def number(x, digits=4):
    return f"{x:.{digits}f}" if np.isfinite(x) else "NA"


def feature_label(name):
    fields = {"age": "Age", "temperature": "Temperature", "heartrate": "Heart rate",
              "resprate": "Respiratory rate", "o2sat": "Oxygen saturation", "sbp": "Systolic BP",
              "dbp": "Diastolic BP", "acuity": "Acuity", "pain_numeric": "Pain score"}
    if name.startswith("miss__missingindicator_"):
        field = name.removeprefix("miss__missingindicator_")
        return "Missing: " + fields.get(field, field)
    if name.startswith("cat__arrival_transport_"):
        return "Transport: " + name.removeprefix("cat__arrival_transport_").lower()
    if name in {"cat__gender_F", "cat__gender_M"}:
        return "Female" if name.endswith("_F") else "Male"
    if name.startswith("num__"):
        field = name.removeprefix("num__")
        return fields.get(field, field)
    return name.removeprefix("text__")


def interval(ci, name, metric, digits=4, scope="test"):
    row = ci[(ci.scope == scope) & (ci.model_name == name) & (ci.metric == metric)].iloc[0]
    if not np.isfinite(row.estimate):
        return "NA"
    return f"{row.estimate:.{digits}f} [{row.ci_lower:.{digits}f}, {row.ci_upper:.{digits}f}]"


def cohort_figure():
    counts = [("All ED encounters", 425087), ("Linked adult admitted encounters", 203016),
              ("Valid encounter timing", 202974), ("No qualifying prior ESKD / dialysis documentation", 195340),
              ("Labelled modelling encounters", 113097)]
    save_csv(pd.DataFrame(counts, columns=["step", "encounters"]), TABLES / "cohort_flow_counts.csv")
    fig, ax = plt.subplots(figsize=(9, 8.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    positions = [.90, .73, .56, .39, .22]
    exclusions = ["Not linked to hospital admission\nn = 222,071", "Invalid timing\nn = 42",
                  "Prior ESKD / dialysis evidence\nn = 7,634",
                  "Indeterminate: 72,171\nNo usable current creatinine: 10,072"]
    for i, ((label, count), y) in enumerate(zip(counts, positions)):
        ax.add_patch(FancyBboxPatch((.04, y - .045), .57, .095, boxstyle="round,pad=0.012,rounding_size=0.006",
                                   linewidth=1, edgecolor="#8A9CA6", facecolor="#F0F5F7"))
        ax.text(.325, y + .004, f"{label}\nn = {count:,}", ha="center", va="center", fontsize=10)
        if i < len(counts) - 1:
            mid = (y + positions[i + 1]) / 2
            ax.annotate("", xy=(.325, positions[i + 1] + .055), xytext=(.325, y - .057), arrowprops={"arrowstyle": "->", "color": "#536574"})
            ax.annotate("", xy=(.68, mid), xytext=(.325, mid), arrowprops={"arrowstyle": "->", "color": "#8D959C"})
            ax.text(.70, mid, exclusions[i], va="center", fontsize=9)
    ax.text(.325, .08, "66,090 patients; 7,866 primary positives\n2,254 stage ≥2 positives", ha="center", va="center", fontsize=10)
    ax.set_title("Cohort construction", loc="left", fontsize=15, pad=16)
    fig.text(.09, .015, "Redrawn from verified Person 1 extraction counts; one row per ED encounter.", fontsize=9, color="#555555")
    finish_figure(fig, "figure1_cohort_flow")


def discrimination_figure(wide, metrics):
    y = wide.y_true.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    indexed = metrics.set_index("model_name")
    for name in MAIN_MODELS:
        p = wide[f"y_prob_{name}"].to_numpy()
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].step(recall, precision, where="post", color=COLORS[name], lw=1.6,
                     label=f"{MODEL_LABELS[name]} (AP {indexed.loc[name, 'average_precision']:.3f})")
        axes[1].plot(fpr, tpr, color=COLORS[name], lw=1.6,
                     label=f"{MODEL_LABELS[name]} (AUC {indexed.loc[name, 'auroc']:.3f})")
    axes[1].plot([0, 1], [0, 1], color="#AAAAAA", ls="--", lw=1)
    axes[0].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="A  Precision-recall", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="B  Receiver operating characteristic", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[1] else "upper right", fontsize=8, frameon=False)
        ax.grid(alpha=.15)
    fig.suptitle(f"Held-out internal test set: {len(wide):,} encounters / {wide.subject_id.nunique():,} patients", fontsize=12)
    finish_figure(fig, "figure2_test_pr_roc")


def calibration_figure(wide):
    rows = []
    for name in MODEL_NAMES:
        p = wide[f"y_prob_{name}"].to_numpy()
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        bins = np.zeros(len(p), dtype=int) if len(edges) == 1 else np.searchsorted(edges[1:-1], p, side="left")
        for b in np.unique(bins):
            mask = bins == b
            rows.append({"model_name": name, "bin": int(b), "encounters": int(mask.sum()),
                         "positives": int(wide.y_true.to_numpy()[mask].sum()),
                         "mean_probability": float(p[mask].mean()), "observed_fraction": float(wide.y_true.to_numpy()[mask].mean())})
    table = pd.DataFrame(rows)
    save_csv(table, TABLES / "test_calibration_bins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    largest = .20
    for name in MAIN_MODELS:
        group = table[table.model_name.eq(name)]
        axes[0].plot(group.mean_probability, group.observed_fraction, marker="o", ms=4, lw=1.4, color=COLORS[name], label=MODEL_LABELS[name])
        largest = max(largest, group.mean_probability.max(), group.observed_fraction.max())
        if name != "prevalence":
            axes[1].hist(wide[f"y_prob_{name}"], bins=np.linspace(0, 1, 101), density=True,
                         histtype="step", lw=1.3, color=COLORS[name], label=MODEL_LABELS[name])
    upper = min(1, largest * 1.08)
    axes[0].plot([0, upper], [0, upper], ls="--", color="#777777", lw=1)
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed AKI fraction", xlim=(0, upper), ylim=(0, upper), title="A  Quantile calibration bins")
    axes[1].set(xlabel="Predicted probability", ylabel="Density", title="B  Distribution of predictions", xlim=(0, .4))
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Original test probabilities; no recalibration applied", fontsize=12)
    finish_figure(fig, "figure3_test_calibration")


def interpretation_figure():
    coefficients = pd.read_csv(P3 / "outputs/coefficients.csv")
    importance = pd.read_csv(P4 / "outputs/final_feature_importance.csv").head(15).iloc[::-1]
    structured = coefficients[coefficients.model_name.eq("structured_lr") & ~coefficients.feature.str.contains("intercept", case=False)].copy()
    structured = structured.assign(magnitude=structured.coefficient.abs()).nlargest(15, "magnitude").sort_values("coefficient")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout="constrained")
    axes[0].barh(structured.feature.map(feature_label), structured.coefficient,
                 color=np.where(structured.coefficient > 0, "#BB4F60", "#3D75A5"))
    axes[0].axvline(0, color="#888888", lw=.8)
    axes[0].set(xlabel="Regularized LR coefficient", title="A  Structured LR: largest absolute coefficients")
    axes[1].barh(importance.feature.map(feature_label), importance.importance, color="#3D75A5")
    axes[1].set(xlabel="XGBoost gain importance", title="B  Final text XGBoost: highest gain")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Frozen development-model interpretation; panels use different scales", fontsize=12)
    finish_figure(fig, "figure4_model_interpretation")
    text = coefficients[coefficients.model_name.eq("structured_text_lr") & coefficients.feature.str.startswith("text__")]
    negative = text.nsmallest(12, "coefficient").sort_values("coefficient")
    positive = text.nlargest(12, "coefficient").sort_values("coefficient")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    for ax, frame, color, title in zip(axes, [negative, positive], ["#3D75A5", "#BB4F60"], ["A  Most negative text coefficients", "B  Most positive text coefficients"]):
        ax.barh(frame.feature.str.removeprefix("text__"), frame.coefficient, color=color)
        ax.axvline(0, color="#888888", lw=.8)
        ax.set(xlabel="L2 logistic-regression coefficient", title=title)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Structured + text LR: predictive associations, not causal effects", fontsize=12)
    finish_figure(fig, "figureS1_text_lr_coefficients")


def paired_figure(pairs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    names = {"structured_text_lr": "Text LR", "structured_lr": "Structured LR", "structured_xgboost": "Structured XGB", "structured_chief_complaint_xgboost": "Text XGB"}
    for ax, metric, title in zip(axes, ["average_precision", "auroc"], ["A  Difference in AP", "B  Difference in AUROC"]):
        group = pairs[pairs.metric.eq(metric)].iloc[::-1].reset_index(drop=True)
        labels = [f"{names[r.model_a]} minus\n{names[r.model_b]}" for r in group.itertuples()]
        for i, r in enumerate(group.itertuples()):
            ax.hlines(i, r.ci_lower, r.ci_upper, color="#3D75A5", lw=2)
            ax.plot(r.difference_a_minus_b, i, "o", color="#BB4F60", ms=6)
        ax.axvline(0, ls="--", lw=1, color="#888888")
        ax.set(yticks=np.arange(len(group)), yticklabels=labels, xlabel="A minus B (95% paired patient-bootstrap CI)", title=title)
        ax.grid(axis="x", alpha=.15)
    finish_figure(fig, "figure5_paired_model_differences")


def sensitivity_figure(test, sensitivity, ci):
    scopes = ["test"] + settings()["subsets"]
    labels = ["All test", "S1: any vital", "S2: complete", "S3: exclude\nfirst-positive", "S4: ≥2\nsamples"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    four = MAIN_MODELS[1:]
    for j, name in enumerate(four):
        points = []
        for i, scope in enumerate(scopes):
            r = ci[(ci.scope == scope) & (ci.model_name == name) & (ci.metric == "average_precision")].iloc[0]
            x = i + (j - 1.5) * .05
            ax.vlines(x, r.ci_lower, r.ci_upper, color=COLORS[name], alpha=.65, lw=1)
            ax.plot(x, r.estimate, "o", color=COLORS[name], ms=4)
            points.append(r.estimate)
        ax.plot(np.arange(len(scopes)) + (j - 1.5) * .05, points, color=COLORS[name], lw=1, label=MODEL_LABELS[name])
    baseline = [test.prevalence.iloc[0]] + [sensitivity[sensitivity.scope.eq(s)].prevalence.iloc[0] for s in scopes[1:]]
    ax.plot(range(len(scopes)), baseline, "s--", color="#777777", ms=4, label="Subgroup prevalence (uninformative AP)")
    ax.set(xticks=range(len(scopes)), xticklabels=labels, ylabel="Average precision", title="Frozen-model subgroup evaluation with original validation cutoffs")
    ax.grid(axis="y", alpha=.15)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2)
    finish_figure(fig, "figure6_sensitivity_ap")


def descriptive_outputs():
    # Selected P2 table rows contain adequate aggregate cell sizes for the report.
    t1 = pd.read_csv(P2_QA / "outputs/table1_characteristics_pos_vs_neg.csv", keep_default_na=False)
    selections = [("Age, years", "median [IQR]"), ("Gender", "M"), ("Arrival transport", "AMBULANCE"),
                  ("Temperature, °F", "missing"), ("Systolic BP, mmHg", "median [IQR]"),
                  ("Diastolic BP, mmHg", "median [IQR]"), ("Triage acuity (ESI)", "1"), ("All six vital signs missing", "Yes")]
    records = []
    for variable, level in selections:
        r = t1[t1.variable.eq(variable) & t1.level.astype(str).eq(level)].iloc[0]
        records.append([f"{variable}: {level}", r["Operational negatives (N=105,231)"], r["Primary positives (N=7,866)"], r["SMD positive vs negative"]])
    save_csv(pd.DataFrame(records, columns=["characteristic", "operational_negative", "primary_positive", "SMD"]), TABLES / "report_descriptive_characteristics.csv")
    return markdown_table(["Characteristic", "Operational negative (105,231)", "Primary positive (7,866)", "SMD"], records)


def write_report(test, validation, sensitivity, ci, pairs, wide):
    cfg = settings()
    freeze = read_json(CHECKS / "analysis_freeze.json")
    result = read_json(CHECKS / "test_evaluation_complete.json")
    labels = test.set_index("model_name")
    methods_table = markdown_table(["Model", "AP [95% CI]", "AUROC [95% CI]", "Brier [95% CI]"],
        [[MODEL_LABELS[n], interval(ci, n, "average_precision"), interval(ci, n, "auroc"), interval(ci, n, "brier_score", 5)] for n in MODEL_NAMES])
    operating = markdown_table(["Model", "Cutoff", "Sensitivity", "Specificity", "PPV", "NPV", "Flagged", "FP / 1,000"],
        [[MODEL_LABELS[n], f"{labels.loc[n, 'threshold']:.6f}", *[number(labels.loc[n, m], 3) for m in ["sensitivity", "specificity", "ppv", "npv", "flagged_fraction"]], number(labels.loc[n, "false_positives_per_1000"], 1)] for n in MODEL_NAMES])
    calibration_table = markdown_table(["Model", "Joint intercept [95% CI]", "Slope [95% CI]", "Calibration-in-large [95% CI]"],
        [[MODEL_LABELS[n], interval(ci, n, "calibration_intercept", 3), interval(ci, n, "calibration_slope", 3), interval(ci, n, "calibration_in_the_large", 3)] for n in MODEL_NAMES])
    pair_rows = []
    for r in pairs[pairs.metric.isin(["average_precision", "auroc"])].itertuples():
        pair_rows.append([f"{MODEL_LABELS[r.model_a]} minus {MODEL_LABELS[r.model_b]}", r.metric,
                          f"{r.difference_a_minus_b:.5f}", f"[{r.ci_lower:.5f}, {r.ci_upper:.5f}]"])
    pair_table = markdown_table(["Comparison", "Metric", "Difference", "95% paired CI"], pair_rows)
    subset_rows = []
    for scope in cfg["subsets"]:
        group = sensitivity[sensitivity.scope.eq(scope)]
        r = group.iloc[0]
        subset_rows.append([scope, f"{int(r.encounters):,}", f"{int(r.positives):,} ({r.prevalence:.2%})",
                            interval(ci, "structured_text_lr", "average_precision", scope=scope),
                            interval(ci, "structured_chief_complaint_xgboost", "average_precision", scope=scope)])
    subset_table = markdown_table(["Subset", "Encounters", "Positive n (%)", "Text LR AP [95% CI]", "Text XGBoost AP [95% CI]"], subset_rows)
    descriptive_table = descriptive_outputs()
    key_pair = pairs[(pairs.model_a == "structured_chief_complaint_xgboost") & (pairs.model_b == "structured_text_lr") & (pairs.metric == "average_precision")].iloc[0]
    pairing_sentence = "included zero" if key_pair.ci_lower <= 0 <= key_pair.ci_upper else "excluded zero"
    best = test.sort_values("average_precision", ascending=False).iloc[0]
    draft = f"""# Early identification of observed creatinine-defined AKI at ED admission

COMP90089 Group 7 report draft · Person 5 integration · 19 October 2026

## 1 Background and study objective

This retrospective study assessed whether information available at emergency department (ED) admission could identify observed creatinine-defined acute kidney injury (AKI) within 24 hours of ED arrival or earlier hospital discharge. The population comprised adults subsequently admitted to hospital with an outcome-assessable encounter. The prediction anchor is described consistently as **at ED admission**; follow-up starts at the recorded ED arrival time. The study includes cases already positive at the first current sample and therefore concerns early identification rather than exclusively incident, previously unsuspected AKI.

The comparison was designed to separate the information gained from richer predictors from differences between modelling approaches: prevalence and age/gender baselines, structured logistic regression (LR), structured-plus-chief-complaint LR, and corresponding XGBoost predictor groups. Person 5 evaluated the seven existing training-fitted models with a common protocol. No additional model development was conducted for this evaluation.

## 2 Data sources and cohort

Person 1 extracted MIMIC-IV-ED and MIMIC-IV Hospital v2.2 data through the authorised BigQuery/Colab workflow. Python performed linkage, cohort selection, predictor cleaning and outcome derivation. The analysis unit was an ED encounter identified by `stay_id`; `subject_id` identified patients with potentially repeated encounters. Adult, hospital-admitted encounters with valid timing were retained, excluding evidence of prior ESKD or chronic dialysis from a different admission discharged before ED arrival. Other chronic kidney disease was retained.

The cohort decreased from 425,087 ED encounters to 203,016 linked admitted encounters, 202,974 encounters after excluding 42 timing errors, and 195,340 eligible encounters after excluding 7,634 with qualifying renal-history evidence. Of these, 113,097 encounters from 66,090 patients were labelled: 7,866 primary positives and 105,231 operational negatives. Another 72,171 were indeterminate and 10,072 had no usable current creatinine. The labelled primary positivity of 6.96% is not an estimate for all admitted ED encounters.

![Figure 1 Cohort construction](outputs/figures/figure1_cohort_flow.png)

Figure 1 was redrawn from the verified extraction counts. An independently supplied, mentor-approved waterfall image was not present in the reviewed folder.

## 3 Outcome, descriptive findings and preprocessing

### 3.1 Creatinine-only operational phenotype

The observation window extended from ED arrival to the earlier of arrival plus 24 hours or hospital discharge. Chemistry creatinine items 50912 and 52546 were used; whole-blood item 52024 was excluded from the primary definition. Usable results were positive values no greater than 150 mg/dL, a broad extraction screen. Conflicting patient/sample-time values were removed; a shared event was assigned to the most recent eligible ED arrival, breaking exact ties by smaller stay ID.

A positive outcome required an observed rise of at least 0.3 mg/dL relative to the minimum strictly earlier usable result within 48 hours, or at least 1.5 times the minimum earlier result within seven days. Boundary times were included and Decimal arithmetic preserved threshold comparisons. References could precede arrival. Operational negatives required at least two distinct current sample times, a timed comparison, and no observed qualifying rise. Neither baseline creatinine nor outcomes were imputed. Operational negatives indicate no qualifying rise observed in available samples, not established clinical absence of AKI.

The secondary label was observed creatinine stage >=2 in the same primary labelled population: 2,254 encounters. Stage 1 and operational negatives were zero for this secondary label. Urine output and renal replacement therapy were not included. No stage >=2-specific fitted models were supplied, so the present seven-model evaluation addresses only the primary outcome.

### 3.2 Independent review and descriptive results

Person 2's independent review passed 30 count reconciliations, 41 row-level rule checks with zero violations, independent cleaning reproduction with zero differing rows across 13 predictors, and 26 synthetic boundary cases. Source laboratory rows were not included in the local handoff, so this comprised exported-data checks, code review and synthetic cases rather than a fresh source-database phenotype extraction.

{descriptive_table}

Missingness differed by outcome: six vital signs were all missing in 15.2% of positives versus 7.3% of operational negatives. AKI frequency was 13.5% among encounters missing all six vitals versus 6.4% with at least one vital recorded, supporting explicit missingness indicators for LR. The labelled cohort was older and had longer hospital stays than excluded encounters: median ages 66, 55 and 52 years and hospital stays 3.7, 1.1 and 0.4 days in labelled, indeterminate and no-creatinine groups, respectively.

Ascertainment also differed. Positives had a median three current samples versus two among negatives; the last negative sample preceded the window endpoint by a median 7.5 hours. Among 7,866 positives, 2,627 (33.4%) were positive at their first current sample, 725 (9.2%) were initially assessable without a rise and later positive, and 4,514 (57.4%) were initially unassessable and later positive. The first-positive sample time is an ascertainment time, not necessarily biological onset. Detailed characteristics, missingness, selection and sampling tables remain in Person 2's `outputs/table1_...` through `table5_...` CSV files and `P2_tables.xlsx`.

### 3.3 Patient-separated development and test partitions

Person 2 fixed a single 70/15/15 patient-level split, stratifying on the worst observed patient outcome (none, stage 1 only, stage >=2) with seed 90089. Training contained 79,364 encounters / 46,263 patients / 5,546 positives; validation contained 16,932 / 9,913 / 1,165; testing contained 16,801 / 9,914 / 1,155. Every patient's encounters stayed together. Person 5 verified encounter/subject alignment and zero patient overlap; no split was changed.

Thirteen cleaned inputs covered age, gender, arrival transport, six vital signs, acuity, numeric pain, pain-entry status and chief complaint. Identifiers, creatinine values, sampling fields and outcome-derived fields were excluded. P2 fitted preprocessing on training only. LR used training-median imputation, numeric standardisation, eight missingness indicators and categorical one-hot encoding, giving 27 structured columns. Tree models retained numeric NaN and used one-hot categoricals, giving 19 columns. Acuity was ordinal numeric. Chief complaints were lowercased and represented using training-fitted 1-2 gram TF-IDF, `min_df=5`, sublinear TF and a clinical shorthand-preserving tokenizer. The 2,772 terms yielded 2,799 LR or 2,791 tree columns.

## 4 Model development and unified evaluation

### 4.1 Frozen model development

P3 provided a training-prevalence constant, unpenalised age/gender LR, structured L2 LR and structured-plus-text L2 LR. The two regularised models independently selected C from 0.001, 0.01, 0.1, 1, 10 and 100 by validation AP; selected values were 10 and 0.1. LR used lbfgs and tolerance 1e-8. The text iteration limit was increased for convergence, without expanding the search. The models were training-fitted without resampling or class weighting.

P4 compared age/gender, structured and structured-plus-text XGBoost using binary logistic objective and histogram trees. A recorded nine-candidate search selected 400 trees, depth 3, learning rate 0.03 for the structured model. The same settings supported a predictor-group comparison. A separate nine-candidate text search selected 300 trees, depth 3, learning rate 0.05; row and column subsampling were 0.8. Model seed 42 was separate from split seed 90089. Validation selected settings; the test set did not drive model or predictor selection.

### 4.2 Inference adapters and structured-zero consistency

P3 joblib files contain raw-input preprocessing/classifier Pipelines, so P5 passed exactly the 13 cleaned raw columns. P4 files contain classifiers: age/gender used named columns `num__age` and `cat__gender_M` (current indices 0 and 10); structured XGBoost received the 19-column dense matrix. For text XGBoost, P5 reproduced P4's explicit-CSR reconstruction of the structured block, preserving zero and NaN entries, and joined the unchanged 2,772 TF-IDF columns. Passing the original sparse text matrix directly would change the structured-zero representation. This adapter reproduced validation predictions before test access. Sparse missing-value semantics are described in the [XGBoost FAQ](https://xgboost.readthedocs.io/en/stable/faq.html).

Execution used the existing `sml` environment: Python {environment()['python']}, scikit-learn {environment()['packages']['scikit-learn']} and XGBoost {environment()['packages']['xgboost']}. Maximum validation probability differences were approximately 1.1e-16 for P3 and 2.1e-8 for P4. Structured LR exhibited a cross-platform tie-ranking effect: its handoff AP was 0.116995 versus local AP 0.117000 despite probability differences below 9e-17. Both original and reproduced validation metrics were retained; scores were not rounded or changed. Probability checks and a documented 1e-5 ranking-metric tolerance passed before testing. Upstream code and model files were unchanged.

### 4.3 Metrics, thresholds, calibration and uncertainty

Average precision (AP), computed as recall-increment-weighted precision rather than trapezoidal PR area, was the primary discrimination metric, with AUROC and Brier score supporting. The implementation was independently checked against scikit-learn's [AP](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html), AUROC and Brier functions, including tied scores and integer sample weights.

Before test evaluation, the user confirmed a target validation sensitivity of at least 0.80. For each model, the largest cutoff attaining that sensitivity was selected, including all ties under probability >= cutoff. These cutoffs, model artifacts, adapters, metric code and sensitivity definitions were recorded in `outputs/checks/analysis_freeze.json`. No test-based cutoff adjustment, refitting or recalibration occurred. Sensitivity, specificity, PPV, NPV, false-positive rate, fraction flagged and false positives per 1,000 encounters used the frozen cutoffs. The constant baseline flags every encounter at its qualifying cutoff, giving no useful graded operating point and undefined NPV.

Calibration used ten quantile bins, with ties retained. An unpenalised diagnostic logistic regression of outcome on logit predicted probability estimated joint intercept and slope; a separate offset-only fit with slope fixed at one estimated calibration-in-the-large. Ideal values are 0, 1 and 0, respectively. Probabilities were clipped to [1e-12, 1-1e-12] only for taking logits, without modifying reported scores. Joint slope/intercept are unidentifiable for the constant model and reported NA; its calibration-in-the-large remains defined. These are diagnostic fits, not probability recalibration. Definitions follow the [prediction-model evaluation methods literature](https://www.bmj.com/content/384/bmj-2023-074820).

Uncertainty used {cfg['bootstrap_replicates']:,} nonparametric patient-cluster bootstrap replicates, seed {cfg['bootstrap_seed']}. Each replicate sampled the same number of patients with replacement and included every encounter of each sampled patient with its multiplicity. All seven models shared the identical resample within each scope. Percentile 95% intervals were computed for primary test metrics, including calibration diagnostics where identifiable. Direct paired differences were calculated from the same replicates for four prespecified model contrasts; no p-values were calculated. Models, thresholds and preprocessing stayed fixed. Intervals therefore describe evaluation uncertainty conditional on the fitted models and validation-selected cutoffs, not uncertainty in the entire model-development process.

Sensitivity analyses evaluated frozen models at unchanged cutoffs in P2-defined S1 (at least one vital), S2 (complete predictor cases), S3 (excluding positives already positive at first sample) and S4 (at least two current samples for positives too). S1/S3/S4 were the recommended phenotype checks; S2 was supplementary. Each subgroup received its own patient-cluster bootstrap with shared resamples across models. Subgroup calibration has point estimates only. These outcome/sampling-derived restrictions are retrospective sensitivity checks, not proposed prospective eligibility rules.

## 5 Results

### 5.1 Held-out test discrimination and overall accuracy

The test set contained {len(wide):,} encounters from {wide.subject_id.nunique():,} patients, including {int(wide.y_true.sum()):,} positives ({wide.y_true.mean():.2%}). These are held-out **internal** test results from the same source population, not external validation.

{methods_table}

The highest observed test AP was {best.average_precision:.4f} for {MODEL_LABELS[best.model_name]}. The prespecified paired AP difference between text XGBoost and text LR was {key_pair.difference_a_minus_b:.5f}, with 95% CI [{key_pair.ci_lower:.5f}, {key_pair.ci_upper:.5f}]; the interval {pairing_sentence}. Model comparisons should be read from paired intervals rather than overlap between separate model intervals.

![Figure 2 Test PR and ROC curves](outputs/figures/figure2_test_pr_roc.png)

### 5.2 Operating characteristics at validation-selected cutoffs

{operating}

At the selected text-XGBoost cutoff, sensitivity was {labels.loc['structured_chief_complaint_xgboost', 'sensitivity']:.2%}, specificity {labels.loc['structured_chief_complaint_xgboost', 'specificity']:.2%}, PPV {labels.loc['structured_chief_complaint_xgboost', 'ppv']:.2%} and NPV {labels.loc['structured_chief_complaint_xgboost', 'npv']:.2%}. It flagged {labels.loc['structured_chief_complaint_xgboost', 'flagged_fraction']:.2%} of encounters: {int(labels.loc['structured_chief_complaint_xgboost', 'true_positives']):,} true-positive and {int(labels.loc['structured_chief_complaint_xgboost', 'false_positives']):,} false-positive flags, with {int(labels.loc['structured_chief_complaint_xgboost', 'false_negatives']):,} missed observed positives. This is about {labels.loc['structured_chief_complaint_xgboost', 'false_positives'] / labels.loc['structured_chief_complaint_xgboost', 'true_positives']:.1f} false-positive flags per true positive, a substantial workload despite the target sensitivity.

Cutoffs above are displayed to six decimals for readability; the CSV and frozen protocol retain full precision. Their selection guaranteed the validation sensitivity target, not a test sensitivity of exactly 0.80. `FP / 1,000` counts false-positive flags per 1,000 total test encounters, distinct from the false-positive rate among negatives. Confidence intervals for threshold metrics are in `outputs/tables/test_confidence_intervals.csv`.

### 5.3 Calibration

{calibration_table}

![Figure 3 Test calibration and probability distributions](outputs/figures/figure3_test_calibration.png)

The plot shows original probabilities, without fitted recalibration. The observed Brier differences and calibration estimates should be interpreted together; a lower Brier score alone is not proof of clinically adequate calibration. Quantile-bin lines are descriptive and have no bin-level confidence bands; patient-bootstrap intervals for the global calibration diagnostics are tabulated.

### 5.4 Paired model comparisons

{pair_table}

![Figure 5 Paired differences](outputs/figures/figure5_paired_model_differences.png)

Comparisons include text gain within LR, algorithm differences for structured inputs, text gain within XGBoost, and algorithm differences with text included. Numeric preprocessing remains algorithm-appropriate while patient partitions, outcomes and the text vocabulary are shared. The complete paired table also reports Brier-score differences; positive AP/AUROC differences favour the first model, whereas negative Brier differences favour the first model.

### 5.5 Sensitivity analyses

{subset_table}

![Figure 6 Sensitivity analysis](outputs/figures/figure6_sensitivity_ap.png)

These analyses used the same frozen models and original validation cutoffs, without retraining. Subgroup AP must be interpreted against each subgroup's prevalence. In particular, S3 removes a defined group of positive encounters and thereby changes prevalence and case mix; its raw AP is not directly comparable with full-population AP as if the target population were unchanged. Full seven-model subgroup metrics, threshold metrics and paired intervals are supplied in the CSV outputs.

### 5.6 Model interpretation

![Figure 4 Frozen model interpretation](outputs/figures/figure4_model_interpretation.png)

![Supplementary Figure S1 Text LR coefficients](outputs/figures/figureS1_text_lr_coefficients.png)

The final XGBoost gain rankings included chief-complaint terms such as labs, cardiac arrest, renal failure and abnormal labs, together with blood pressure, arrival transport, acuity and respiratory rate. LR coefficients provide signed regularised associations; their scale depends on preprocessing and coding. Gain is non-directional and cannot be compared numerically with LR coefficient magnitudes. These interpretations were inherited from frozen development models, without test-driven feature selection.

## 6 Discussion and limitations

The held-out evaluation quantifies how the supplied models identify the study's observed creatinine-defined outcome among selected admitted encounters. Richer clinical inputs and chief-complaint information should be assessed through the prespecified within-method and between-method contrasts, with the paired intervals above defining the precision of observed differences. The evaluation does not establish a clinically validated decision threshold, clinical net benefit or benefit from deploying a model.

Text XGBoost had the largest observed AP and AUROC in this test population. Its AP advantage over text LR was {key_pair.difference_a_minus_b:.4f}, and the paired interval {pairing_sentence}. Adding text also improved AP within each modelling approach in the prespecified comparisons. Nevertheless, the final model's AUROC of {labels.loc['structured_chief_complaint_xgboost', 'auroc']:.3f} represents limited discrimination, and the high-sensitivity operating point produced many false-positive flags. Improved ranking and modest Brier reductions should therefore be distinguished from an established useful clinical action rule. The four model contrasts were prespecified, but their individual 95% intervals were not adjusted for multiplicity.

Selection and ascertainment are substantial limitations. Entry into supervised modelling required sufficient information to assign an operational label, favouring older patients with longer stays and repeat testing. Excluded patients differed in age, missingness and observation time. Creatinine sampling was intermittent and differed by outcome. Operational negatives can include unobserved rises, and incomplete prior history can leave unrecorded ESKD or dialysis. Labels use creatinine only, without urine output, treatment information or clinical adjudication. Broad predictor screens preserve some unusual values, and missingness can capture care processes as well as patient characteristics.

One third of positives were already positive at the first sample. Chief complaints containing renal diagnoses, abnormal laboratory findings or other clinical concerns may represent knowledge or suspicion already present at ED admission. Their predictive value supports early identification within the operational study task, not an assertion that the model anticipates AKI before clinical suspicion or predicts only new biological onset. S3 addresses one aspect of this issue but does not recover true onset times.

Validation was used for hyperparameter, predictor-group and threshold choices. Its performance remains a development estimate. Test predictions were generated after a recorded freeze and were not used to revise model choices. Repeated encounters were addressed through patient-separated partitions and patient-cluster resampling, although the resampling intervals condition on fixed models and cutoffs. This is a single-source internal holdout; temporal transportability, external validation and clinical utility remain untested. Stage >=2 labels were prepared upstream, but secondary-outcome models were not part of the seven supplied artifacts and are not claimed as completed analyses.

## Reproducibility and contribution record

Person 1 supplied extraction, cohort, labels and cleaning; Person 2 supplied independent phenotype QA, descriptive tables, the fixed patient manifest and training-only preprocessing; Person 3 supplied four frozen baseline Pipelines and development outputs; Person 4 supplied three XGBoost classifiers, the structured-zero correction and gain interpretation; Person 5 supplied inference adapters, unified checks, frozen validation thresholds, held-out evaluation, paired patient bootstrap, subgroup evaluation, figures and this integrated draft.

All Person 5 code and generated artifacts are inside `person5_evaluation/`; input paths are resolved relative to that folder's project parent. {result['original_files_verified_unchanged']} original handoff files were checked by SHA-256 before and after evaluation with no changes. The current run used existing `sml` packages and did not install or upgrade dependencies. The raw patient-level predictions and original MIMIC-derived materials remain private project artifacts; this report presents aggregate findings. Full protocol, environments, provenance, diagnostic warnings and completion checks are retained under `outputs/checks/`.

Primary report tables: `test_metrics.csv`, `test_confidence_intervals.csv`, `test_paired_differences.csv`, `sensitivity_metrics.csv`, `all_confidence_intervals.csv`, and `all_paired_differences.csv`. Validation results and thresholds are separate. Both 300-dpi PNG and vector PDF versions of every figure are available.

Local method sources: [project method](../Project%20Method.docx), [P1 handoff](../person1_handoff/README.txt), [P2 QA](../person2_handoff/person2_phenotype_review/README_P2_QA_summary.md), [P2 preprocessing](../person2_handoff/person2_split_preprocessing/README_split_preprocessing.md), [P3 methods](../person3_handoff_v2_2026-10-16/person3_handoff/P3_report_contribution.md), and [P4 methods](../person4_handoff/person4_handoff/P4_report_contribution.md).
"""
    (HERE / "report_draft.md").write_text(draft, encoding="utf-8")
    summary = f"""# Person 5 已完成工作与结果

整理日期：2026 年 10 月 19 日。

已在当前 `sml` 环境完成七个现有主结局模型的统一评估，代码、结果、图表和英文报告初稿均保存在本目录。队友原文件未修改，未重新训练模型、拟合预处理器、重新划分患者或用测试结果调整阈值。

## 已完成

- 输入适配：P3 接收 13 个已清洗字段；P4 年龄性别按名称索引 `[0, 10]`；文本 XGBoost 显式保留结构化零值与 NaN。
- 验证集复现：七个模型概率均通过容差检查；统一 AP/AUROC/Brier 与 scikit-learn 交叉核验。
- 冻结阈值：用户确认“验证集敏感度至少 0.80 的最大阈值”，已记录完整精度和测试前冻结文件。
- 独立测试：16,801 次就诊、9,914 名患者、1,155 次阳性；已有长表和宽表预测。
- 患者级 Bootstrap：每个分析范围 1,000 次，各模型使用相同患者重采样，输出 95% 百分位区间及四组模型配对差值。
- 敏感性分析：S1/S3/S4 主分析及 S2 完全病例补充；冻结模型与阈值，不重新训练。主测试包含校准诊断 CI；子群校准只报告点估计。
- 图表：队列流程、测试 PR/ROC、校准、模型解释、配对差值、敏感性 AP，以及文本 LR 系数补充图；均有 PNG 与 PDF。
- 报告整合：`report_draft.md` 含方法、描述结果、主测试结果、阈值指标、校准、配对比较、敏感性和局限性。
- 原文件保护：{result['original_files_verified_unchanged']} 个原交接文件的 SHA-256 均保持一致。

## 主测试结果

{methods_table}

## 验证集冻结阈值下的测试指标

{operating}

完整阈值指标 CI 见 `outputs/tables/test_confidence_intervals.csv`。常数基线在满足敏感度目标的阈值下全部判阳性，因此没有实用分层能力，NPV 不可定义；常规联合校准截距和斜率也不可识别，单独报告 calibration-in-the-large。

## 配对比较

{pair_table}

差值置信区间来自同一次患者抽样下的直接模型相减，不是两个独立 CI 相减，也未额外生成 p 值。不能把数值上更高的 AP 自动解释为已证实更优或具备临床效益。

## 敏感性分析

{subset_table}

S1 为至少一项生命体征；S2 为完整病例；S3 排除首次当前样本即阳性的病例；S4 对阳性也要求至少两次当前采样。子群患病率不同，AP 需结合子群患病率解释。

## 需要在统稿中保留的说明

1. 本次是同源数据的独立内部测试集评估，不是外部验证。
2. 主诉中的肾衰竭、化验异常等词可能反映急诊入院时已有认知；研究表述为早期识别。
3. 结构化 LR 的跨环境概率差异低于 9e-17，但极少量并列分数次序使 AP 相差约 4.83e-6；`handoff_validation_metrics.csv` 保留原交接值，`validation_metrics.csv` 保留本机值，没有对概率舍入。测试前已记录该数值现象。
4. 当前 XGBoost 为 {environment()['packages']['xgboost']}，交接记录为 3.4.1；所有模型先通过验证概率复现，版本及加载提示均已留档。
5. 队列 waterfall 根据已核实计数重绘；目录中未见独立的导师确认图，因此没有把重绘图标记为已获导师确认。
6. Stage >=2 的专门模型尚未由 P3/P4 交付，本次没有把主结局模型冒充次结局模型，也未新增模型训练。
7. 模型已冻结；如需修改阈值或模型方案，应作为后续另行记录的分析，不能用当前测试结果反向选择。

## 交付入口

- [英文报告初稿](report_draft.md)
- [测试结果表](outputs/tables/test_metrics.csv)
- [95% 置信区间](outputs/tables/test_confidence_intervals.csv)
- [配对差值](outputs/tables/test_paired_differences.csv)
- [敏感性结果](outputs/tables/sensitivity_metrics.csv)
- [测试预测宽表](outputs/predictions/test_predictions_all_models.csv)
- [冻结方案](outputs/checks/analysis_freeze.json)
- [运行与复现说明](README.md)
"""
    (HERE / "person5_work_summary_zh.md").write_text(summary, encoding="utf-8")


def main():
    make_output_dirs()
    if not (CHECKS / "test_evaluation_complete.json").exists():
        raise RuntimeError("Complete frozen test evaluation before generating report")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300})
    test = pd.read_csv(TABLES / "test_metrics.csv")
    validation = pd.read_csv(TABLES / "validation_metrics.csv")
    sensitivity = pd.read_csv(TABLES / "sensitivity_metrics.csv")
    ci = pd.read_csv(TABLES / "all_confidence_intervals.csv")
    pairs = pd.read_csv(TABLES / "test_paired_differences.csv")
    wide = pd.read_csv(HERE / "outputs/predictions/test_predictions_all_models.csv", float_precision="round_trip")
    cohort_figure()
    discrimination_figure(wide, test)
    calibration_figure(wide)
    interpretation_figure()
    paired_figure(pairs)
    sensitivity_figure(test, sensitivity, ci)
    write_report(test, validation, sensitivity, ci, pairs, wide)
    write_json(CHECKS / "report_generation.json", {"figures": [p.name for p in sorted(FIGURES.glob("*"))],
               "reports": ["report_draft.md", "person5_work_summary_zh.md"], "source": "frozen evaluation outputs"})
    print("Report draft, Chinese summary and seven PNG/PDF figure pairs generated.", flush=True)


if __name__ == "__main__":
    main()
