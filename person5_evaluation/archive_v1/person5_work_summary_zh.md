# Person 5 已完成工作与结果

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
- 原文件保护：108 个原交接文件的 SHA-256 均保持一致。

## 主测试结果

| Model | AP [95% CI] | AUROC [95% CI] | Brier [95% CI] |
| --- | --- | --- | --- |
| Training prevalence | 0.0687 [0.0647, 0.0726] | 0.5000 [0.5000, 0.5000] | 0.06402 [0.06052, 0.06737] |
| Age / gender LR | 0.0739 [0.0684, 0.0804] | 0.5331 [0.5154, 0.5500] | 0.06397 [0.06047, 0.06729] |
| Structured LR | 0.1109 [0.1020, 0.1227] | 0.6224 [0.6048, 0.6410] | 0.06311 [0.05959, 0.06637] |
| Structured + text LR | 0.1328 [0.1202, 0.1495] | 0.6499 [0.6326, 0.6686] | 0.06245 [0.05906, 0.06568] |
| Age / gender XGBoost | 0.0753 [0.0697, 0.0814] | 0.5393 [0.5212, 0.5563] | 0.06395 [0.06040, 0.06726] |
| Structured XGBoost | 0.1229 [0.1116, 0.1378] | 0.6422 [0.6240, 0.6604] | 0.06275 [0.05926, 0.06599] |
| Structured + text XGBoost | 0.1444 [0.1292, 0.1628] | 0.6649 [0.6477, 0.6820] | 0.06208 [0.05868, 0.06531] |

## 验证集冻结阈值下的测试指标

| Model | Cutoff | Sensitivity | Specificity | PPV | NPV | Flagged | FP / 1,000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training prevalence | 0.069881 | 1.000 | 0.000 | 0.069 | NA | 1.000 | 931.3 |
| Age / gender LR | 0.060825 | 0.815 | 0.228 | 0.072 | 0.943 | 0.775 | 718.6 |
| Structured LR | 0.052481 | 0.791 | 0.317 | 0.079 | 0.954 | 0.690 | 635.7 |
| Structured + text LR | 0.051644 | 0.784 | 0.380 | 0.085 | 0.960 | 0.631 | 577.4 |
| Age / gender XGBoost | 0.061636 | 0.823 | 0.206 | 0.071 | 0.940 | 0.796 | 739.0 |
| Structured XGBoost | 0.049853 | 0.805 | 0.343 | 0.083 | 0.960 | 0.667 | 611.7 |
| Structured + text XGBoost | 0.050693 | 0.805 | 0.388 | 0.089 | 0.964 | 0.625 | 569.5 |

完整阈值指标 CI 见 `outputs/tables/test_confidence_intervals.csv`。常数基线在满足敏感度目标的阈值下全部判阳性，因此没有实用分层能力，NPV 不可定义；常规联合校准截距和斜率也不可识别，单独报告 calibration-in-the-large。

## 配对比较

| Comparison | Metric | Difference | 95% paired CI |
| --- | --- | --- | --- |
| Structured + text LR minus Structured LR | average_precision | 0.02192 | [0.01342, 0.03263] |
| Structured + text LR minus Structured LR | auroc | 0.02752 | [0.01572, 0.03942] |
| Structured XGBoost minus Structured LR | average_precision | 0.01205 | [0.00439, 0.02001] |
| Structured XGBoost minus Structured LR | auroc | 0.01977 | [0.00849, 0.03128] |
| Structured + text XGBoost minus Structured XGBoost | average_precision | 0.02146 | [0.01064, 0.03666] |
| Structured + text XGBoost minus Structured XGBoost | auroc | 0.02273 | [0.01282, 0.03265] |
| Structured + text XGBoost minus Structured + text LR | average_precision | 0.01160 | [0.00251, 0.02143] |
| Structured + text XGBoost minus Structured + text LR | auroc | 0.01497 | [0.00524, 0.02504] |

差值置信区间来自同一次患者抽样下的直接模型相减，不是两个独立 CI 相减，也未额外生成 p 值。不能把数值上更高的 AP 自动解释为已证实更优或具备临床效益。

## 敏感性分析

| Subset | Encounters | Positive n (%) | Text LR AP [95% CI] | Text XGBoost AP [95% CI] |
| --- | --- | --- | --- | --- |
| S1_has_any_vital | 15,499 | 975 (6.29%) | 0.1090 [0.0975, 0.1245] | 0.1216 [0.1081, 0.1391] |
| S2_complete_case | 13,355 | 797 (5.97%) | 0.1081 [0.0942, 0.1255] | 0.1148 [0.1009, 0.1332] |
| S3_excl_positive_at_first_sample | 16,405 | 759 (4.63%) | 0.1060 [0.0931, 0.1231] | 0.1172 [0.1019, 0.1375] |
| S4_two_or_more_samples | 16,742 | 1,096 (6.55%) | 0.1310 [0.1186, 0.1470] | 0.1428 [0.1286, 0.1613] |

S1 为至少一项生命体征；S2 为完整病例；S3 排除首次当前样本即阳性的病例；S4 对阳性也要求至少两次当前采样。子群患病率不同，AP 需结合子群患病率解释。

## 需要在统稿中保留的说明

1. 本次是同源数据的独立内部测试集评估，不是外部验证。
2. 主诉中的肾衰竭、化验异常等词可能反映急诊入院时已有认知；研究表述为早期识别。
3. 结构化 LR 的跨环境概率差异低于 9e-17，但极少量并列分数次序使 AP 相差约 4.83e-6；`handoff_validation_metrics.csv` 保留原交接值，`validation_metrics.csv` 保留本机值，没有对概率舍入。测试前已记录该数值现象。
4. 当前 XGBoost 为 3.1.2，交接记录为 3.4.1；所有模型先通过验证概率复现，版本及加载提示均已留档。
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
