# 影像、模型权重与运行环境

仓库现在公开。大文件按三类发布到 GitHub Releases：`image_data`、`model_weights`、`environment`。

上传正在进行；每类完整上传并通过 GitHub 端 SHA-256 校验后，其 Release 才会公开。最终链接、文件数与体积会写入本页。

## 保存与去重方式

每类归档保留 `E_01/` 和 `D_01/` 两个来源目录。内容完全相同的文件在 tar 内使用硬链接去重；本仓库的还原程序将其恢复为独立文件，避免修改一份环境时影响另一份。每类附带 `*_files.jsonl`，记录文件路径、大小、SHA-256 和重复文件对应关系。

归档按 512 MiB 分卷，文件名为 `<category>.tar.gz.part0001`、`part0002` 等。每个分卷的 SHA-256 位于 Release 说明中。请下载同一类的所有分卷，用本仓库的 `restore_release_assets.py` 校验并直接解压；无需先生成一个巨大的合并文件。

```powershell
python restore_release_assets.py --parts-dir E:\Downloads\research-assets --category image_data --output E:\RestoredResearch\image_data --checksums release_assets.json
```

对另外两类分别将 `--category` 替换为 `model_weights`、`environment`，并使用各自的空输出目录。输出目录必须为空，避免覆盖已有研究文件。清单中的路径与原始文件 SHA-256 可以用于后续核查。

## 数据来源与许可证

| 数据集 | 原始发布与引用 | 数据许可证 |
| --- | --- | --- |
| T2-weighted Kidney MRI Segmentation | Alexander J. Daniel 等，University of Nottingham；[Zenodo 5153568](https://zenodo.org/records/5153568)，[配套论文](https://doi.org/10.1002/mrm.28768) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| CHAOS | Ali Emre Kavur、M. Alper Selver、Oğuz Dicle、Mustafa Barış、N. Sinem Gezer；[Zenodo 3431873](https://zenodo.org/records/3431873) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| AMOS22 | Yuanfeng Ji 等；[Zenodo 7262581](https://zenodo.org/records/7262581)，[论文](https://arxiv.org/abs/2206.08023) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

下载的原始压缩包已按 Zenodo 官方 MD5 核验。`raw/` 保留本地解压文件；`prepared/`、`splits/`、`audit/` 是本地预处理、折分和审计产物，具体变换见项目的 `data` 与 `external_data.py` 模块。原始数据的许可证继续适用；CHAOS 的非商业与相同方式共享要求也适用于相应派生产物。

## 模型与环境说明

权重来自原始目录的实验检查点及 SSDU 的 `saved_models/`。第三方权重保留其上游来源与权利，不能将本次归档视为给它们授予新许可证；SSDU 来源为 [byaman14/SSDU](https://github.com/byaman14/SSDU)。

运行环境保存本地 Python/Conda/venv 文件快照与原包的许可证文件。环境包含 Windows 和 Linux 文件，可能保留原机器路径；跨机器运行时应优先按项目 `pyproject.toml`、`uv.lock` 或 `requirements.txt` 重建环境。归档本身不保证在其他操作系统上直接运行。

排除损坏或未完成的下载残片、凭据文件及私钥内容。原始目录始终保留不变。
