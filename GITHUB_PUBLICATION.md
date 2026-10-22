# GitHub 版本说明

更新日期：2026-10-22。公开仓库按五位队友的目录组织，原冻结模型的正式评估位于 `person5_evaluation/current/`。新增 [五人整合提交 Notebook](person5_evaluation/final_submission/Final_AKI_End_to_End.ipynb)，从原版本 BigQuery 数据重新提取、训练并评估两种结局；它的结果另存于 `final_submission/published_results/`，不覆盖原冻结结果。特征探索仍只保留 [思路与结论](person5_evaluation/探索实验思路与结论_20261021.md)。

公开内容包括源代码、清空运行输出的 Notebook、方法与工作说明、主次结局汇总指标、置信区间、敏感性分析和图表。发布清单及文件 SHA-256 见 `PUBLICATION_MANIFEST.json`。P5 历史主结局代码与必要来源记录保留用于理解新旧比较。

患者级数据、标签、患者划分、原始文本、逐例预测、拟合模型、预处理器、交接压缩包、完整词表和 Bootstrap 缓存仅保存在获授权的本地工作区，不包含在当前公开版本中。部分原始交接说明中的文件链接因此只适用于完整本地项目。Notebook 公开副本清空输出；本地已执行 Notebook 保持原样。

本地 Notebook 与少量说明文本保留原始内容，因此可能在 `git status` 中显示与公开提交不同。这是刻意保留的差异；后续公开更新也应先检查输出和数据范围，避免用 `git add .` 把本地运行输出直接重新上传。

复现 `current/` 的冻结模型评估需在获授权的本地环境补齐 P1–P4 数据、冻结模型及原始缓存。新增 `final_submission/` Notebook 则不依赖旧 CSV 或模型，但需要自己的 BigQuery 计费项目、MIMIC-IV 2.2 与 ED 数据访问权限和 `sml` 环境。公开代码将个人计费项目替换为空占位，Notebook 执行输出清空。本次新增内容不包含原始临床数据、患者级预测、历史压缩包或完整运行目录。

清理范围是远端当前文件树中的旧 `github_upload/` 副本、旧 P3 路径、旧 P5 根目录脚本、压缩包、系统缓存和受限数据产物。独有的旧 P4 主结局训练材料保留在 `person4_handoff/legacy_primary_training/`，明确标为历史版本；完整本地归档保留。

本次使用普通提交，保留 Git 历史。因此旧提交中曾上传的文件仍可能通过历史访问；当前文件树删除不等于历史清除。
