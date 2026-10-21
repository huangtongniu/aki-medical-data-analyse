# Person 5 统一工作目录

先读 [工作说明](工作说明.md)。当前执行版本是 `current/`。

正式评估继续使用原冻结模型和阈值。三组特征探索的做法与结果见 [探索实验思路与结论](探索实验思路与结论_20261021.md)；探索模型不采用，相关实验产物按用户要求清理。

| 目录 | 用途 |
| --- | --- |
| [current](current) | 两种结局十四模型评估、Notebook、结果表、图及中文解释 |
| [archive_v1](archive_v1) | 原主结局七模型评估，保留原文件内容和冻结记录 |
| [archive/github_upload_20261021](archive/github_upload_20261021) | 整理前的旧发布副本，保留原发布清单和旧源码；不是当前发布包 |
| [archive/structure_20261021](archive/structure_20261021) | 移动前 SHA-256 清单、原 P5 文件备份、路径映射及验证记录 |

在已激活 `sml` 的终端，从项目根目录运行：

```powershell
python -B person5_evaluation/run_all.py
python -B person5_evaluation/execute_notebook.py
```

第一条执行评估流程，第二条执行并保存 [当前 Notebook](current/P5_primary_secondary_evaluation.ipynb) 的全部输出。已有结果先校验哈希再复用。无需重新训练、重选阈值或重复生成 1,000 次 Bootstrap 缓存。

从 Notebook 界面使用 `sml` 也可 Run All。本机自动运行采用 IPython 顺序执行，因为 Jupyter 内核创建本机通信连接曾阻塞，详见 [current/README.md](current/README.md)。

目录迁移仅调整路径、文档和来源核验。原分析冻结记录不改写；两个冻结源码文件的路径适配另记在迁移记录中，保留改动前的副本和前后哈希。原主结局历史脚本保留旧布局，回顾时阅读其输出；日常运行使用本页入口。
