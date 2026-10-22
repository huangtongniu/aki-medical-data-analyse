# 原版本数据重新提取后的汇总结果

来源：完成运行 `20261021T092359_830334Z`，使用原 ED 库与 HOSP 2.2。该目录只含聚合指标与六张 PNG 图，原冻结模型评估继续保存在 `../../current/`。

- [主次结局模型对比](tables/primary_secondary_comparison.csv)
- [配对差值及置信区间](tables/all_paired_differences.csv)
- [敏感性分析](tables/sensitivity_metrics.csv)
- [全部指标置信区间](tables/all_confidence_intervals.csv)
- [图表](figures/)

两种结局、14 个模型；主测试集及 S1–S4 各 1,000 次共享患者 Bootstrap，阈值只由验证集 Sensitivity ≥0.80 规则确定。各模型在本次运行中重新训练；这些结果不替代历史冻结模型的评估记录。
