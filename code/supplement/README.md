# Supplementary research source · 论文文件夹补充代码

This source snapshot comes from the author's `Desktop/论文` directory. It supplements the original source selection with 17 Python files; none is byte-identical to the 19 files in the first source release. Files are copied without algorithm changes. Original relative paths, sizes and SHA-256 values are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

| 模块 | 入口 | 内容 |
|---|---|---|
| 原始数据预处理 | [prepare_ecg_ppg_pretrain_npz.py](preprocessing/prepare_ecg_ppg_pretrain_npz.py) | 预训练 NPZ 准备；同目录还保留 permissive 与快速多源版本 |
| 虚拟 R 峰对齐 | [train_dual_view_virtual_r_alignment.py](alignment/train_dual_view_virtual_r_alignment.py) | Stage 1 双视图表示对齐；PPG-only 视图由冻结的锚点网络重新分割，以 CSFM 双模态表示为 teacher |
| PAT 数据与划分 | [pat_core.py](pat/dataset/pat_core.py) / [make_folds.py](pat/dataset/make_folds.py) | PAT 数据构建、核心处理、折划分及基线评测源码 |
| PAT 模型评测 | [run_splitenc8_pat.py](pat/splitenc8/run_splitenc8_pat.py) | `ppg_only`、`ecg_ppg_direct`、`ecg_only` 三种 track；预测指标以 ms 报告 |
| ECG→PPG 协议与模型 | [common_raw10s.py](translation/common_raw10s.py) / [splitenc8_sc_model.py](translation/splitenc8_sc_model.py) | 原始 10 秒窗口、记录组划分和 S-C 波形恢复模型 |
| ECG→PPG 训练与评估 | [train_splitenc8_sc_ddp.py](translation/train_splitenc8_sc_ddp.py) / [evaluate_splitenc8_sc.py](translation/evaluate_splitenc8_sc.py) | `ECG2PPG-COMMON-RAW10S-GROUP-v3-ALLBEATS` 实验版本 |

## 版本与使用范围

- 这是历史研究源码的公开选集，不是新运行的实验。代码存在、日志存在或文件名包含 `best` 都不等于已验证论文结论。
- Stage 1 表示对齐不训练 foundation reconstruction/prediction heads；不能将它与 Stage 2 重建预训练混为同一个目标。
- 三种 PAT 输入 track 必须分别解释。预测头、标签定义、折划分、checkpoint 与数据集需对应原记录，不能混合汇总。
- 翻译源码来自 **v3 ALLBEATS** 目录。该版本取消 v2 的 25 拍样本上限，使用 ECG-only 输入和 PPG 监督，并按源记录组划分数据。这里公开实现，不将其自动提升为当前论文正式结果或新性能声明。
- 这些脚本依赖原始数据、CSFM/锚点网络等外部模型资产，以及原服务器目录。保持同组文件的相对位置；对照完整归档恢复依赖，再检查参数与路径。没有提供未经验证的统一依赖锁文件或即开即用的训练命令。
- 原始服务器脚本、配置和实验记录保存在补充 Release 内。本次没有生成新的服务器执行包，也没有运行训练、推理或论文编译。

See [supplement archive notes](../SUPPLEMENT.md) and the main [reproducibility notes](../../docs/REPRODUCIBILITY.md).
