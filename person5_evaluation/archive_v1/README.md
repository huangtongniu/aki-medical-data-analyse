# Person 5 统一评估与报告

本目录使用项目现有的 `sml` 环境，读取相邻队友目录，完成七个既有主结局模型的验证、测试、患者级 Bootstrap、敏感性分析、图表和报告初稿。所有输入路径由当前脚本所在位置向项目根目录解析，不依赖硬编码盘符，也不要求终端工作目录固定。没有修改、移动或覆盖队友文件，没有安装或升级环境包。

## 阅读入口

- `person5_work_summary_zh.md`：中文工作汇总和主要结果。
- `report_draft.md`：英文报告初稿，含方法、结果表、图和讨论。
- `outputs/tables/test_metrics.csv`：七模型测试点估计与固定阈值指标。
- `outputs/tables/test_confidence_intervals.csv`：测试集指标的 95% 区间。
- `outputs/tables/test_paired_differences.csv`：四组预设模型对的配对差值区间。
- `outputs/tables/sensitivity_metrics.csv`：S1–S4 冻结模型子群结果。
- `outputs/checks/analysis_freeze.json`：打开测试评估前保存的冻结方案、阈值与源文件哈希。

## 运行

在已经激活 `sml` 的终端，从项目根目录运行：

```powershell
python -B person5_evaluation/run_all.py
```

也可在本目录运行：

```powershell
python -B run_all.py
```

`run_all.py` 依次运行统计单元检查、验证与冻结、测试与 Bootstrap、报告绘图和结果核验。如果测试已开始，则不再重选验证阈值；测试推理和 Bootstrap 可从有哈希校验的缓存继续。现有结果已经完整运行，重复运行主要复核缓存并重建图表与报告。

单独运行各步：

```powershell
python -B test_evaluation.py
python -B run_validation.py
python -B run_test_evaluation.py
python -B make_report.py
python -B verify_outputs.py
```

测试开始后，`run_validation.py` 会拒绝重新冻结阈值；需要复核已有完整结果时使用 `run_all.py` 或 `verify_outputs.py`。脚本对单次进程限制数值计算线程，避免小型矩阵计算中线程过量；没有改变系统或环境的永久设置。

## 文件与职责

| 文件 | 用途 |
|---|---|
| `settings.json` | 测试前预设：80% 敏感度规则、1,000 次 Bootstrap、种子、四个子群与四组对比 |
| `common.py` | 相对路径、文件哈希、环境和输出工具 |
| `inference.py` | P3 完整 Pipeline 与 P4 分类器的输入适配、ID 检查和零值重建 |
| `metrics.py` | 带并列分数的 AP/AUROC、Brier、阈值指标和诊断校准拟合 |
| `run_validation.py` | 概率复现、原交接指标复算、本机指标、阈值和冻结日志 |
| `run_test_evaluation.py` | 冻结后的测试推理、患者级配对 Bootstrap 和敏感性分析 |
| `make_report.py` | 300 dpi PNG、矢量 PDF、英文报告和中文汇总 |
| `verify_outputs.py` | 不重新拟合模型，独立复核结果、区间、缓存、相对链接和原文件完整性 |
| `test_evaluation.py` | 与 scikit-learn 和 statsmodels 独立对照的统计检查 |
| `run_all.py` | 可继续执行的入口 |
| `requirements_sml.txt` | 此次实际使用的依赖版本记录，不是安装操作 |

目录依赖保持现状：P1 为 `../person1_handoff`，P2 为 `../person2_handoff`，P3 为 `../person3_handoff_v2_2026-10-16/person3_handoff`，P4 为 `../person4_handoff/person4_handoff`。无须重新解压或复制大数据。

## 推理与验证

P3 四个 `.joblib` 均接受 13 个已清洗的原始字段。P4 年龄性别按特征名选择，目前索引为 `[0, 10]`；结构化模型使用 19 列稠密矩阵；文本模型使用显式保存零值和 NaN 的结构化 CSR 块加原文本列。没有导入 P4 含 `!unzip` 和顶层训练操作的 Colab 导出脚本。

P4 交接预测缺少 `target` 列，读入时根据其代码与 P2 标签核对后，在 P5 输出中补为 `aki_label`。所有预测按 `stay_id` 和 `subject_id` 校验，不通过原始行位置猜测对应关系。

本机 XGBoost 3.1.2 与 P4 记录的 3.4.1 不同；验证概率最大误差约 2e-8，在预设 1e-7 容差内。结构化 LR 的跨平台机器精度并列分数使 AP 相差约 4.83e-6：`handoff_validation_metrics.csv` 保留原交接指标，`validation_metrics.csv` 保留本机推理指标；本次没有舍入概率，也没有更换模型或调整参数。该现象和容差在测试前写入冻结方案。

## 统计约定

- 主结局固定为 `aki_label`。本次评估七个提供的主结局模型；没有训练 Stage ≥2 新模型。
- AP 是非插值 average precision，以召回增量加权；不使用梯形 PR 面积。
- 用户明确确认阈值规则：验证集敏感度至少 0.80 时的最大概率阈值，`probability >= threshold`，保留全部并列值。
- 每个 Bootstrap 以 `subject_id` 为单位有放回抽取同样数量的患者；患者的所有就诊随抽样重数一起保留。每次迭代的七个模型使用同一份患者抽样。
- 实现使用患者抽样重数作为行权重，与真正重复患者的全部行在数学上等价，且已用含并列概率的显式行复制检查验证。
- 主测试集计算 13 项指标的百分位 95% CI；常数基线的联合校准参数和 NPV 不可定义，留空并在报告标为 NA。
- 校准联合截距和斜率由 `outcome ~ intercept + slope * logit(probability)` 的无惩罚诊断拟合估计；calibration-in-the-large 是固定斜率 1 的单独偏移拟合。它们不会反向修改测试预测。
- 子群 S1/S3/S4 加 S2 补充使用各自 1,000 次患者级 Bootstrap；各子群内七模型共享抽样。子群报告校准点估计，不额外生成校准 CI；其他十项指标及配对 AP/AUROC/Brier 区间均提供。
- 四组配对为：文本 LR−结构化 LR；结构化 XGBoost−结构化 LR；文本 XGBoost−结构化 XGBoost；文本 XGBoost−文本 LR。使用同一次抽样的直接差值；没有生成 p 值，单项 95% 区间未做多重比较调整。
- 区间以固定模型和已选验证阈值为条件，不包括重新训练或阈值选择的不确定性。

## 输出结构

```text
outputs/
├── checks/        冻结方案、环境、输入检查、源文件哈希、Bootstrap 缓存及继续执行进度
├── predictions/   验证与测试的逐行预测长表/宽表
├── tables/        验证、阈值、测试、配对差值、Bootstrap 重复与敏感性结果
└── figures/       七张图各一份 PNG 和 PDF
```

`test_predictions_all_models.csv` 为用户要求的测试宽表：`stay_id`、`subject_id`、`y_true` 和七个 `y_prob_<model>`；标准七列长表另存为 `test_predictions_long.csv`。Bootstrap 检查缓存可继续执行，无需为生成报告重新做模型开发。

队列图依据核实的 P1 计数重绘，未标记为已获导师确认的原图。特征解释复用冻结的 P3 系数与 P4 gain，不使用测试集选择特征。逐行预测是私有 MIMIC 衍生材料，和原交接数据采用相同存储范围。
