# COMP90089 团队项目

当前按五位队友归档。请先看 [队友工作与成果梳理](队友工作与成果梳理.md)，再通过下表进入各人的工作说明或对应目录。

GitHub 发布范围见 [公开版本说明](GITHUB_PUBLICATION.md)。原冻结模型、患者级数据和完整本地归档留在授权工作区；公开仓库提供代码、汇总结果和说明，复现时需补齐本地输入。

| 文件夹 | 负责内容 | 工作与成果说明 |
| --- | --- | --- |
| `person1_handoff/` | 原始提取、队列、结局标签和预测变量清洗 | [Person 1](person1_handoff/工作说明.md) |
| `person2_handoff/` | 表型审核、描述分析、患者划分和训练集预处理 | [Person 2](person2_handoff/工作说明.md) |
| `person3_handoff/` | 主、次结局的常数与逻辑回归基线 | [Person 3](person3_handoff/工作说明.md) |
| `person4_handoff/` | 主、次结局 XGBoost 与修正后的推理接口 | [Person 4](person4_handoff/工作说明.md) |
| `person5_evaluation/` | 统一测试评估、Bootstrap、敏感性分析和结果解读 | [Person 5](person5_evaluation/工作说明.md) |

当前结果入口：[P5 Notebook](person5_evaluation/current/P5_primary_secondary_evaluation.ipynb)、[中文结果解读](person5_evaluation/current/P5_新版结果解读_团队讨论.md)、[十四模型对比表](person5_evaluation/current/outputs/tables/combined_model_comparison.csv)。

激活已有 `sml` 环境，在项目根目录执行：

```powershell
python -B person5_evaluation/run_all.py
```

执行并保存 Notebook 全部单元输出：

```powershell
python -B person5_evaluation/execute_notebook.py
```

2026-10-21 整理：P3 两种结局归入同一文件夹；P5 当前评估位于 `current/`，旧主结局评估位于 `archive_v1/`，旧发布副本和目录迁移记录位于 `archive/`。原始交接 zip 分别保存在对应队友的归档目录。`.git/` 为版本控制元数据，不计入五个工作文件夹。

[Project Method.docx](Project%20Method.docx) 保留为原始方法材料；[what should I do.md](what%20should%20I%20do.md) 为早期任务记录，当前完成状态以各人工作说明为准。
