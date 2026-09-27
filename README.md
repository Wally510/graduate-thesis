# Graduate thesis archive

论文项目、实验结果与模型权重的公开归档。大文件保存在下面两个 Releases 中。

- [论文、代码和实验结果](https://github.com/Wally510/graduate-thesis/releases/tag/project-archive-v1)：9 个 ZIP，共 9.50 GiB。
- [模型权重](https://github.com/Wally510/graduate-thesis/releases/tag/model-weights-v1)：37 个 ZIP，共 47.56 GiB。

每个 ZIP 均可单独解压；完整恢复需下载对应 Release 内的全部 ZIP，解压到同一父目录。请先阅读附件 `RESTORE.md`，用 `SHA256SUMS.txt` 校验下载文件。不要将 ZIP 文件拼接。

先恢复项目中去重和压缩的结果文件，再恢复权重的原始路径。归档保留逐文件 SHA-256 清单。归档过程没有重新训练、推理或改变论文正文；未执行完整科学复现。第三方代码和模型的原有许可条件仍适用，归档不额外授予许可。
