# Person 5：主结局与 Stage ≥2 完整评估

本版读取 2026-10-21 更新的 P3/P4 交接包，用现有 `sml` 环境评估两种结局各七个模型。所有脚本按自身位置解析相对项目路径。原主结局评估已归入 `../archive_v1/`，其结果、冻结记录和队友文件内容保持原样。

## 运行入口

在已激活 `sml` 的终端，从项目根目录运行：

```powershell
python -B person5_evaluation/run_all.py
```

或者在当前文件夹运行 `python -B run_all.py`。Notebook 入口为 [P5_primary_secondary_evaluation.ipynb](P5_primary_secondary_evaluation.ipynb)，选择 `sml` 内核后执行 Run All。它完成 P5 评估，依赖本文件夹脚本和队友数据；不是 Part 1–5 的单文件最终提交 Notebook。本轮范围由用户确认为新版 P5 评估及运行 Notebook。

如果 Notebook 界面没有注册名为 `sml` 的内核，先在终端激活 `sml`，然后从项目根目录运行 `python -B person5_evaluation/execute_notebook.py`。该脚本使用当前 Python 中的 IPython 顺序执行全部代码单元，并将文字、表格和图片输出保存回 Notebook，不安装或修改用户的内核配置。本次采用此方式，因为运行环境中 Jupyter 创建本机通信连接时阻塞；实际执行的是同一组代码单元，未通过 Jupyter 界面启动内核。运行环境版本记录在 `requirements_sml.txt`；本次没有安装或升级依赖。

执行顺序为：九项统计/输入检查、验证预测复现与阈值冻结、测试推理与 Bootstrap、生成汇总表和图、独立结果复核。已有冻结和缓存时先校验其哈希，再复用；不会重新选择测试阈值。首次完整执行包含五组各 1,000 次患者抽样，每次同时评价十四个模型。

## 新旧包与推理接口

- P3 主结局材料此前从保留 zip 恢复到 P5 输入目录，现在统一移至 `../../person3_handoff/primary/`。原 zip 位于 `../../person3_handoff/archives/`；七个主结局模型均必须与旧 P5 冻结哈希一致。
- 新 P3 包为 `../../person3_handoff/secondary`，只包含新增的四个 Stage ≥2 基线；它不替代主结局模型。
- P3 的八个模型都接收 P2 的 13 个清洗后原始字段，Pipeline 内部完成预处理。Stage ≥2 Pipeline 的 `target_name_` 必须为 `aki_stage2_label`。
- P4 使用 `../../person4_handoff/person4_handoff/P4_inference.py` 提供的 `load_p4_models` 和 `p4_predict`，一次加载两种结局共六个分类器。年龄和性别按特征名索引，实际位置为 `[0, 10]`。
- P4 文本输入的前 19 列显式保留零和 NaN，后 2,772 列保持原 TF-IDF。额外验证结构块、零值、NaN 和文本块没有变化。
- 所有预测按 `(target, model_name, stay_id)` 区分，并检查 `subject_id`、标签、分区与行顺序。

## 固定的分析约定

1. 两个结局均使用完整的同一队列。Stage ≥2 的阴性包括 Stage 1 和原操作性阴性，不限制为主结局阳性病例。
2. AP 为非插值 average precision；辅助指标为 AUROC、Brier。另计算校准联合截距/斜率、固定斜率 1 的 calibration-in-the-large，以及十等频箱校准曲线。校准拟合只用于诊断，不调整预测。
3. 用户此前确认的阈值规则继续采用：验证集敏感度至少 0.80 的最高概率阈值，概率 `>=` 阈值即报警，保留同分值。每个次要模型独立选择自己的阈值，主结局则保留原冻结阈值。
4. 本版按更新请求使用**分层患者 Bootstrap**：在每个评估人群中，将患者按其最高观察分期分为 0、1、≥2 三层，各层有放回抽取同样数量患者。抽中患者的全部合格就诊随抽样重数保留。同次抽样用于两种结局及全部十四个模型。
5. 采用 1,000 次抽样、种子 9008905、百分位 95% CI。指标仍为就诊层面，抽样单元为患者。区间以模型、阈值和各层患者数固定为条件，不覆盖训练或阈值选择的不确定性。
6. 主测试集为 13 项指标计算 CI；子集为前十项指标计算 CI，校准参数仅提供点估计。无法定义的指标留空并报告有效重复数，尤其常数模型的 NPV 和联合校准参数。
7. 每种结局预设四组配对：文本 LR−结构化 LR，结构化 XGBoost−结构化 LR，文本 XGBoost−结构化 XGBoost，文本 XGBoost−文本 LR。差值直接在同次抽样中计算。未做多重比较调整，不输出正式 p 值。
8. S1、S2、S3、S4 原样沿用 P2 标志。S3 是“排除主结局首次采样已阳性”，应用于 Stage ≥2 时会同时删除部分 Stage 1 次要阴性；不能称为另行定义的次要结局首次采样阴性子群。

旧版是未分层患者 Bootstrap，本版是分层患者 Bootstrap，所以主结局 CI 可能变化；主结局模型、预测与阈值不变。旧版文件未被替换。此次次要模型实现晚于主结局结果公布，冻结记录如实说明时间顺序；不能声称早于全部测试信息接触。

## 结果入口

- [中文结果汇总](results_summary_zh.md)
- [十四模型完整对比表](outputs/tables/combined_model_comparison.csv)
- [验证集阈值](outputs/tables/validation_thresholds.csv)
- [测试配对差值](outputs/tables/test_paired_differences.csv)
- [敏感性分析](outputs/tables/sensitivity_metrics.csv)
- [校准分箱数据](outputs/tables/test_calibration_bins.csv)
- [独立验证记录](outputs/checks/final_verification.json)

`outputs/predictions/` 保存联合长表、按结局标识的宽表，以及各结局独立子目录。`outputs/figures/` 保存四张主/次结局联合图，每张均有 300 dpi PNG 和矢量 PDF。`outputs/checks/` 保存冻结方案、接口诊断、源文件哈希、原文件完整性记录、Bootstrap 缓存和验证结果。

项目统计口径为 113,097 次就诊、66,090 名患者、主要阳性 7,866、次要阳性 2,254；预测时点为急诊入院时（at ED admission）。阶段性已知疾病和主诉中的临床怀疑信息限制了结果解释，应称为早期识别。两种结局阳性率不同，AP 不能直接跨结局比较。

2026-10-21 目录整理：当前路径为 `person5_evaluation/current/`。新增 `layout.py` 解析历史冻结路径；原冻结 JSON 未改写，路径适配的源码变更及原文件副本见 [迁移记录](../archive/structure_20261021/migration.json)。[五人工作总览](../../队友工作与成果梳理.md) 与 [Person 5 工作说明](../工作说明.md) 提供最新目录入口。
