# 五人整合的最终提交 Notebook

提交文件：[Final_AKI_End_to_End.ipynb](Final_AKI_End_to_End.ipynb)。代码全部内嵌，共 35 个代码单元；不需要把队友脚本一起提交。

它从 BigQuery 的原始表重新提取数据，重新构建标签、清洗、划分患者、拟合预处理、训练模型并评估。**不读取现有 CSV、NPY、NPZ 或 joblib，也不复用之前的模型和 Bootstrap 缓存。** 流程中新生成的 CSV 是输出，不是输入。

## 覆盖内容

| 部分 | 来源与工作 |
| --- | --- |
| P1 | 七张原始表查询；成人住院 ED 队列；既往 ESKD/透析排除；严格时间窗与 Decimal 肌酐判定；主、次结局；原始特征清洗 |
| P2 | 24 个边界案例、26 条就诊测试；41 项行级规则检查；独立清洗复核；描述性表 1–5；按患者 70/15/15 分层划分，seed 90089；仅训练集拟合预处理 |
| P3 | 每种结局四个基线：训练阳性率常数、年龄性别 LR、结构化 LR、文本 LR；重新在验证集选择 C |
| P4 | 每种结局三个 XGBoost；结构化和文本模型各九组候选；显式保留稀疏结构化块的零值与 NaN；P5 按 P4 交付说明补写次结局训练串联 |
| P5 | 两种结局各七模型；验证 Sensitivity ≥0.80 最大阈值；测试 AP、AUROC、Brier、阈值指标和校准；患者级配对 Bootstrap；S1–S4；对比表与六组图 |

预测时点统一写为 **at ED admission**。源码的 `triage` 仅是原始表和字段的名称。

## 运行方式

在 Jupyter 或 VS Code 中选择已有 `sml` 内核，设置自己的 `GOOGLE_CLOUD_PROJECT` 或首个代码单元的 `BILLING_PROJECT`，再 Run All。GitHub 公开副本使用空项目占位。登录使用 `pandas_gbq` 浏览器认证，当前版本通过 `reauth=True` 重新选择有 2.2 权限的账号；凭据不写入 Notebook。

也可从项目根目录执行：

```powershell
conda activate sml
python -B person5_evaluation/final_submission/execute_final_notebook.py
```

该入口用新的 `sml` Python 进程和 IPython 依次执行每个单元，遇错停止。它不启动 Jupyter 的本机通信内核，也不绕过认证。执行输出另存于 `verification/Final_AKI_End_to_End.executed.ipynb`，逐单元状态见 `verification/real_run_execution.json`。

所有数据输出路径均相对当前工作目录：`final_submission_runs/<UTC运行编号>/`。每次运行创建新目录；不会读取上一次的产物。辅助入口会把工作目录设为本文件所在目录，因此其输出落在这里。

## 数据版本与当前验证状态

2026-10-21 用户明确要求保留原数据版本。Notebook 保持 `physionet-data.mimiciv_ed` 与 `physionet-data.mimiciv_2_2_hosp`，没有切换到 3.1。

2026-10-22 发布前复核：**最新真实数据运行已完成**，本地运行编号为 `20261021T092359_830334Z`。保存的 Notebook 有 35 个连续执行的代码单元（计数 27–61），没有错误输出；结束状态为 `complete`，产物清单的全部 SHA-256 校验一致。源表记录确认使用原 ED 库与 HOSP 2.2。

重新提取得到 113,097 次就诊、66,090 名患者、主结局阳性 7,866、次结局阳性 2,254，与历史队列一致。已完成 14 个模型，以及主测试集和 S1–S4 各 1,000 次患者 Bootstrap。

早先自动执行的 403 错误记录保留在本地 `verification/`，只反映当时的认证状态，不代表最新运行状态。当前 Notebook 的认证单元已改为 `reauth=True`；构建脚本同步保留该修改。本轮发布核对已有执行记录与产物，未重新发起整套查询和训练，也不把这次核对描述为新的干净内核执行。

可公开的核对摘要见 [verification_summary.json](verification_summary.json)，完整本地运行记录、查询日志与患者级产物不随 GitHub 发布。

历史队列数量检查保持启用；重新提取的人数若不吻合会停止并记录差异。原 P5 `current/` 和 `archive_v1/` 的冻结模型和结果不改写。

另有独立合成原始表的全流程检查：35/35 个代码单元、14 个最终模型、全部 LR/XGBoost 调参候选、S1–S4、六组 PNG/PDF 图均运行成功。测试中主动阻止读取旧 CSV、矩阵和模型。合成检查每个范围用 20 次 Bootstrap；正式 Notebook 默认每个范围 1,000 次。合成检查使用较早的缓存认证配置，其结果只用于验证分析代码，不充当真实数据结果。

## GitHub 发布内容

公开副本包含清空执行输出的 Notebook、辅助代码、环境说明，以及 [published_results](published_results/) 中的模型汇总表和六张 PNG 图。本地已执行 Notebook 保持原样。历史 ZIP、失败运行目录、合成测试目录、逐例预测、患者划分、子集标志、完整词表/系数、Bootstrap 明细和缓存均留在本地。

## 主要输出与限制

- `tables/primary_secondary_comparison.csv`：两个结局的完整模型对比，含 95% CI。
- `tables/all_paired_differences.csv`：主测试集与 S1–S4 的配对 AP/AUROC/Brier 差值及 CI。
- `tables/sensitivity_metrics.csv`、`calibration_bins.csv`：子集结果及校准分箱数据。
- `figures/`：PR/ROC、校准、队列流程、特征关联、敏感性和配对差值图，PNG/PDF。
- `checks/`：数据来源/查询记录、环境、模型搜索与阈值冻结、Bootstrap 数组和产物哈希。
- `private/`：新生成的患者划分和测试预测；行级子集标志同样是私人数据，禁止随公开提交上传。

所有候选模型都只在训练患者上拟合；验证集选择参数及阈值，随后才计算测试预测。同一 Bootstrap 抽样用于全部十四模型，按患者最高结局级别分层。CI 条件于这次数据划分、已训练模型和固定分层人数；多重比较未校正。校准拟合只是诊断，不会用测试标签重校准预测。常数模型的联合校准斜率/截距不可识别，保留为缺失。

旧测试结果此前已被团队查看，因此这次是既定方案的重新计算，不能声称是新一次完全盲法外部验证。

维护用文件 `build_final_notebook.py`、`check_synthetic_pipeline.py`、`check_bigquery_access.py` 和执行入口不属于 Notebook 运行依赖。`source_manifest.json` 记录整合时参考的队友源码哈希；`requirements-sml.txt` 记录现有环境版本，未自动安装或升级软件。
