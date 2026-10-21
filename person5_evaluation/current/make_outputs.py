"""Comparison tables and standalone figures for both frozen outcomes."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from common import (CHECKS, FIGURES, HERE, MODEL_LABELS, MODEL_NAMES, PREDICTIONS, TABLES,
                    TARGETS, TARGET_LABELS, read_json, save_csv, utc_now, write_json)

COLORS = {"prevalence": "#777777", "age_gender_lr": "#A8702C", "structured_lr": "#198678",
          "structured_text_lr": "#7854A6", "age_gender_xgboost": "#AC9B29",
          "structured_xgboost": "#3475A8", "structured_chief_complaint_xgboost": "#BE4C58"}
MAIN = ["structured_lr", "structured_text_lr", "structured_xgboost", "structured_chief_complaint_xgboost"]


def finish(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def md_table(headers, rows):
    safe = lambda value: str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(safe, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(safe, row)) + " |" for row in rows])


def main():
    complete = read_json(CHECKS / "evaluation_complete.json")
    test = pd.read_csv(TABLES / "test_metrics.csv", float_precision="round_trip")
    ci = pd.read_csv(TABLES / "test_confidence_intervals.csv", float_precision="round_trip")
    pairs = pd.read_csv(TABLES / "test_paired_differences.csv", float_precision="round_trip")
    wide = pd.read_csv(PREDICTIONS / "test_predictions_all_models.csv", float_precision="round_trip")
    sub = pd.read_csv(TABLES / "sensitivity_metrics.csv", float_precision="round_trip")
    all_ci = pd.read_csv(TABLES / "all_confidence_intervals.csv", float_precision="round_trip")
    combined = test.copy()
    for metric in ci.metric.unique():
        part = ci[ci.metric.eq(metric)][["target", "model_name", "ci_lower", "ci_upper", "valid_replicates"]]
        part = part.rename(columns={k: f"{metric}_{k}" for k in ["ci_lower", "ci_upper", "valid_replicates"]})
        combined = combined.merge(part, on=["target", "model_name"], validate="one_to_one")
    save_csv(combined, TABLES / "combined_model_comparison.csv")
    for target in TARGETS:
        for frame, name in [(combined, "model_comparison.csv"), (pairs, "paired_differences.csv"), (sub, "sensitivity_metrics.csv")]:
            save_csv(frame[frame.target.eq(target)], TABLES / target / name)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for row, target in enumerate(TARGETS):
        group = wide[wide.target.eq(target)]
        y = group.y_true.to_numpy()
        indexed = test[test.target.eq(target)].set_index("model_name")
        for name in MAIN:
            p = group[f"y_prob_{name}"].to_numpy()
            precision, recall, _ = precision_recall_curve(y, p)
            fpr, tpr, _ = roc_curve(y, p)
            axes[row, 0].step(recall, precision, where="post", color=COLORS[name], lw=1.5,
                label=f"{MODEL_LABELS[name]}: {indexed.loc[name, 'average_precision']:.3f}")
            axes[row, 1].plot(fpr, tpr, color=COLORS[name], lw=1.5,
                label=f"{MODEL_LABELS[name]}: {indexed.loc[name, 'auroc']:.3f}")
        axes[row, 0].axhline(y.mean(), ls="--", lw=1, color=COLORS["prevalence"], label=f"Prevalence: {y.mean():.3f}")
        axes[row, 1].plot([0, 1], [0, 1], ls="--", lw=1, color="#999999")
        axes[row, 0].set(xlabel="Recall", ylabel="Precision", title=f"{TARGET_LABELS[target]} | PR (legend: AP)", xlim=(0, 1), ylim=(0, 1))
        axes[row, 1].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{TARGET_LABELS[target]} | ROC (legend: AUC)", xlim=(0, 1), ylim=(0, 1))
        axes[row, 0].legend(fontsize=8, frameon=False, loc="upper right")
        axes[row, 1].legend(fontsize=8, frameon=False, loc="lower right")
        for ax in axes[row]:
            ax.grid(alpha=.15)
    fig.suptitle("Held-out internal test set: 16,801 encounters / 9,914 patients", fontsize=13)
    finish(fig, "figure1_primary_secondary_pr_roc")

    bins = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    for ax, target in zip(axes, TARGETS):
        group = wide[wide.target.eq(target)]
        upper = 0
        for name in MODEL_NAMES:
            p, y = group[f"y_prob_{name}"].to_numpy(), group.y_true.to_numpy()
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            assignment = np.zeros(len(p), dtype=int) if len(edges) == 1 else np.searchsorted(edges[1:-1], p, side="left")
            model_bins = []
            for b in np.unique(assignment):
                mask = assignment == b
                entry = {"target": target, "model_name": name, "bin": int(b), "encounters": int(mask.sum()),
                    "positives": int(y[mask].sum()), "mean_probability": float(p[mask].mean()), "observed_fraction": float(y[mask].mean())}
                bins.append(entry)
                model_bins.append(entry)
            frame = pd.DataFrame(model_bins)
            upper = max(upper, frame.mean_probability.max(), frame.observed_fraction.max())
            ax.plot(frame.mean_probability, frame.observed_fraction, marker="o", ms=3.5, lw=1.3,
                    ls="--" if name.startswith("age_gender") else "-", color=COLORS[name], label=MODEL_LABELS[name])
        upper = min(1, upper * 1.12)
        ax.plot([0, upper], [0, upper], ls=":", color="#333333", lw=1, label="Perfect calibration")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title=TARGET_LABELS[target],
               xlim=(0, upper), ylim=(0, upper))
        ax.grid(alpha=.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Calibration assessment: up to ten quantile bins; models are not recalibrated", fontsize=12)
    finish(fig, "figure2_primary_secondary_calibration")
    save_csv(pd.DataFrame(bins), TABLES / "test_calibration_bins.csv")

    comparisons = ["Text LR minus structured LR", "Structured XGB minus structured LR", "Text XGB minus structured XGB", "Text XGB minus text LR"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for row, target in enumerate(TARGETS):
        for column, metric in enumerate(["average_precision", "auroc"]):
            ax = axes[row, column]
            frame = pairs[pairs.target.eq(target) & pairs.metric.eq(metric)].reset_index(drop=True)
            for index, entry in frame.iterrows():
                ax.hlines(index, entry.ci_lower, entry.ci_upper, color="#3475A8", lw=2)
                ax.plot(entry.difference_a_minus_b, index, "o", color="#BE4C58", ms=5)
            ax.axvline(0, ls="--", color="#888888", lw=1)
            ax.set_yticks(range(4), comparisons, fontsize=8)
            ax.invert_yaxis()
            ax.set(xlabel="Difference (95% patient-stratified bootstrap CI)", title=f"{TARGET_LABELS[target]} | {'AP' if column == 0 else 'AUROC'}")
            ax.grid(axis="x", alpha=.15)
    finish(fig, "figure3_paired_model_differences")

    scopes = ["test", "S1_has_any_vital", "S2_complete_case", "S3_excl_positive_at_first_sample", "S4_two_or_more_samples"]
    scope_labels = ["Main test", "S1: any vital", "S2: complete", "S3: exclude first +", "S4: >=2 samples"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, target in zip(axes, TARGETS):
        for offset, name in zip([-.24, -.08, .08, .24], MAIN):
            frame = all_ci[all_ci.target.eq(target) & all_ci.model_name.eq(name) & all_ci.metric.eq("average_precision")].set_index("scope").loc[scopes]
            x = np.arange(len(scopes)) + offset
            ax.errorbar(x, frame.estimate, yerr=[frame.estimate - frame.ci_lower, frame.ci_upper - frame.estimate],
                        fmt="o", ms=4, capsize=2, lw=1.1, color=COLORS[name], label=MODEL_LABELS[name])
        prevalences = pd.concat([test, sub]).query("target == @target and model_name == 'prevalence'").set_index("scope").loc[scopes, "prevalence"]
        ax.plot(range(len(scopes)), prevalences, "k_", ms=12, label="Subset prevalence")
        ax.set_xticks(range(len(scopes)), scope_labels, rotation=22, ha="right", fontsize=8)
        ax.set(ylabel="Average precision", title=TARGET_LABELS[target], ylim=(0, None))
        ax.grid(axis="y", alpha=.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Frozen-model sensitivity analyses; AP must be interpreted with subgroup prevalence", fontsize=12)
    finish(fig, "figure4_sensitivity_analyses")

    def interval(target, model, metric, percent=False):
        row = ci[ci.target.eq(target) & ci.model_name.eq(model) & ci.metric.eq(metric)].iloc[0]
        if not np.isfinite(row.estimate):
            return "NA"
        multiplier, digits = (100, 1) if percent else (1, 4)
        return f"{row.estimate * multiplier:.{digits}f} [{row.ci_lower * multiplier:.{digits}f}, {row.ci_upper * multiplier:.{digits}f}]"

    lines = ["# Person 5 新版评估结果", "", "本次评估包括 Primary（Stage ≥1）和 Secondary（Stage ≥2），每种结局各七个冻结模型。所有区间为 95% CI。", "",
        "## 分析范围与方法", "",
        "- 总队列 113,097 次就诊、66,090 名患者；主要阳性 7,866，次要阳性 2,254。",
        "- 测试集 16,801 次就诊、9,914 名患者；主要阳性 1,155，次要阳性 332。Stage 1 在次要结局中记为阴性。",
        "- 预测时点统一为急诊入院时（at ED admission）。",
        "- 各模型分别使用验证集敏感度至少 80% 的最大阈值；主结局阈值沿用旧冻结值，次要结局重新在验证集选择。",
        "- 1,000 次患者级分层 Bootstrap，按患者在评估人群中的最高分期分为 0、1、≥2 三层；每层有放回抽取同样数量患者，保留其所有合格就诊。两种结局和十四个模型共享每次抽样。",
        "- 旧版采用未分层患者 Bootstrap。本版按新要求使用分层抽样，因此 CI 可以变化；旧版文件保持不变，主结局模型、概率与阈值均经过复核。",
        "- 次要模型是在主结局测试结果已经可见之后补充开发。此次冻结早于次要模型测试推理，不能表述为早于所有测试信息接触。",
        "- CI 不包含重新训练和阈值选择的不确定性；未做多重比较调整。常数模型的 NPV、联合校准参数不可定义。", ""]
    for target in TARGETS:
        label = "主要结局 Stage ≥1" if target == TARGETS[0] else "次要结局 Stage ≥2"
        group = test[test.target.eq(target)].set_index("model_name")
        lines += [f"## {label}", "", f"测试阳性率：{group.iloc[0].prevalence:.2%}。", "",
            md_table(["模型", "AP [95% CI]", "AUROC [95% CI]", "Brier [95% CI]"],
                [[MODEL_LABELS[name]] + [interval(target, name, metric) for metric in ["average_precision", "auroc", "brier_score"]] for name in MODEL_NAMES]), "",
            md_table(["模型", "冻结阈值", "敏感度 % [CI]", "特异度 % [CI]", "PPV % [CI]", "NPV % [CI]"],
                [[MODEL_LABELS[name], f"{group.loc[name, 'threshold']:.6f}"] + [interval(target, name, metric, True) for metric in ["sensitivity", "specificity", "ppv", "npv"]] for name in MODEL_NAMES]), ""]
        part = pairs[pairs.target.eq(target) & pairs.metric.eq("average_precision")]
        lines += ["配对 AP 差值：", "", md_table(["模型 A − 模型 B", "ΔAP", "95% CI", "区间是否支持 A 更高"],
            [[f"{MODEL_LABELS[r.model_a]} − {MODEL_LABELS[r.model_b]}", f"{r.difference_a_minus_b:.4f}",
              f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]", "是" if r.ci_lower > 0 else "否 / 尚不确定"] for r in part.itertuples()]), ""]
        comparison = pairs[pairs.target.eq(target) & pairs.model_a.eq("structured_chief_complaint_xgboost")
                           & pairs.model_b.eq("structured_text_lr")].set_index("metric")
        auc, brier = comparison.loc["auroc"], comparison.loc["brier_score"]
        lines += [f"文本 XGBoost 相比文本 LR：ΔAUROC = {auc.difference_a_minus_b:.4f} "
                  f"[{auc.ci_lower:.4f}, {auc.ci_upper:.4f}]；ΔBrier = {brier.difference_a_minus_b:.6f} "
                  f"[{brier.ci_lower:.6f}, {brier.ci_upper:.6f}]（Brier 越低越好）。", ""]
        if target == TARGETS[1]:
            lines += ["这里 AP 的配对区间支持文本 XGBoost 优于文本 LR；但 AUROC、Brier 的差值区间均跨零，"
                      "还不能确认这两项也有改善。上述区间未作多重比较调整。", ""]
        operating = group.loc["structured_chief_complaint_xgboost"]
        lines += [f"文本 XGBoost 在冻结阈值下标记了 {operating.flagged_fraction:.2%} 的测试就诊，"
                  f"敏感度为 {operating.sensitivity:.2%}，PPV 为 {operating.ppv:.2%}。"
                  "验证集达到 80% 敏感度不保证测试集也达到 80%；这些结果不能直接证明临床部署获益。", ""]
    lines += ["## 敏感性分析", "", "S1–S4 沿用 P2 提供的记录标志，模型和阈值不变。S3 按主要结局的首次采样阳性定义；用于次要结局时，也会删除部分 Stage 1（次要阴性）记录。它不是重新定义的“首次采样未达 Stage ≥2”人群。", "",
        "下表列出文本 XGBoost 的子集结果；完整表包含每种结局的全部七个模型。", ""]
    for target in TARGETS:
        selected = sub[sub.target.eq(target) & sub.model_name.eq("structured_chief_complaint_xgboost")]
        rows = []
        for entry in selected.itertuples():
            ap_ci = all_ci[all_ci.scope.eq(entry.scope) & all_ci.target.eq(target)
                           & all_ci.model_name.eq(entry.model_name) & all_ci.metric.eq("average_precision")].iloc[0]
            rows.append([entry.scope.split("_")[0], int(entry.encounters), int(entry.patients), int(entry.positives),
                         f"{entry.prevalence:.2%}", f"{entry.average_precision:.4f} [{ap_ci.ci_lower:.4f}, {ap_ci.ci_upper:.4f}]",
                         f"{entry.auroc:.4f}", f"{entry.sensitivity:.2%}", f"{entry.ppv:.2%}"])
        lines += [f"### {TARGET_LABELS[target]}", "", md_table(
            ["子集", "就诊数", "患者数", "阳性数", "阳性率", "AP [95% CI]", "AUROC", "敏感度", "PPV"], rows), ""]
    lines += ["S1：至少一项生命体征非缺失；S2：完全病例；S3：排除主结局首次采样已阳性；S4：至少两次采样。", "",
        "子集结果见 [完整敏感性表](outputs/tables/sensitivity_metrics.csv) 和 [全部配对区间](outputs/tables/all_paired_differences.csv)。不同子集、不同结局的 AP 不能脱离各自阳性率直接比较；不能仅凭点估计排序，断言某个模型在每个子集中均更好。", "",
        "## 图与数据", "",
        "![PR 与 ROC](outputs/figures/figure1_primary_secondary_pr_roc.png)", "",
        "![校准](outputs/figures/figure2_primary_secondary_calibration.png)", "",
        "![配对差值](outputs/figures/figure3_paired_model_differences.png)", "",
        "![敏感性分析](outputs/figures/figure4_sensitivity_analyses.png)", "",
        "- [两种结局完整对比表](outputs/tables/combined_model_comparison.csv)",
        "- [验证集阈值](outputs/tables/validation_thresholds.csv)",
        "- [主测试集配对差值](outputs/tables/test_paired_differences.csv)",
        "- [运行说明与限制](README.md)", ""]
    (HERE / "results_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(CHECKS / "outputs_generation.json", {"created_at_utc": utc_now(), "comparison_rows": len(combined),
        "targets": TARGETS, "figures": 4, "formats": ["png", "pdf"], "calibration_bins": len(bins),
        "evaluation_completed_at_utc": complete["completed_at_utc"]})
    print("Combined comparisons, Chinese result summary and four PNG/PDF figure pairs created.", flush=True)


if __name__ == "__main__":
    main()
