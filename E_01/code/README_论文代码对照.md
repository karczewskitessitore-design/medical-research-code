# MRI 自监督重建论文与代码对照指南

本目录于 2026-08-02 获取并固定了三套 MRI 重建源码。本文档说明两篇本地论文与代码仓库的对应关系、主要入口、数据格式、依赖和许可证。

本次仅获取源码并进行静态核对：没有安装环境、下载 fastMRI 数据、运行训练或修改任何上游源码。

## 1. 最重要的对应关系

| 本地论文/资料 | 对应代码 | 关系 |
|---|---|---|
| [2023-Wang-arXive-k-band self-supervised MRI.pdf](<../2023-Wang-arXive-k-band self-supervised MRI.pdf>) | [K-band](./K-band/) | 论文内嵌链接指向的官方实现 |
| [2020-Yaman-MRM-Self‐supervised learning.pdf](<../2020-Yaman-MRM-Self‐supervised learning.pdf>) | [SSDU](./SSDU/) | 论文作者提供的官方实现 |
| Millard 与 Chiew 的 Noisier2Noise 论文 | [Noisier2Noise_for_recon](./Noisier2Noise_for_recon/) | 用户指定的相关实现；包含 SSDU 配置，但不是 k-band 论文的官方代码 |

因此，不能把 `Noisier2Noise_for_recon` 直接视为 2023 年 k-band 论文的实现。k-band 的正确官方仓库是 `mikgroup/K-band`。

## 2. 固定版本与获取记录

三个仓库均保留完整 `.git` 历史，并处于固定提交的 detached HEAD 状态，以避免默认分支后续变化影响复现。

| 目录 | Origin | 上游默认分支 | 固定提交 | 提交时间 |
|---|---|---|---|---|
| `K-band` | <https://github.com/mikgroup/K-band.git> | `main` | `5542a0befb65125236b316690a53123ba0dcac63` | 2023-09-03 |
| `SSDU` | <https://github.com/byaman14/SSDU.git> | `main` | `060ec76d479156560e62f4d95708f7260925e31d` | 2022-06-08 |
| `Noisier2Noise_for_recon` | <https://github.com/charlesmillard/Noisier2Noise_for_recon.git> | `master` | `c94567f46b2102ec9d23a9d5364afc8ed612649e` | 2023-01-27 |

可在任一仓库内用以下命令确认版本：

```powershell
git remote get-url origin
git rev-parse HEAD
git status --short
```

`git status --short` 应当没有输出。

## 3. k-band 论文与代码

### 3.1 方法概要

k-band 用仅覆盖部分 k 空间的有限分辨率 band 数据训练重建网络。不同训练样本使用随机 band 方向，使整个训练集在统计意义上覆盖完整 k 空间；训练时再对 band 内数据进行可变密度欠采样，并使用推导出的 k 空间损失权重。推理阶段则把网络用于覆盖完整 k 空间范围的欠采样数据，从而得到高分辨率重建。

论文中的两个关键条件是：

1. 不同样本的 band 方向应随机化，使训练集覆盖全部 k 空间位置。
2. 训练损失应采用对应的 k 空间权重，使子集上的随机梯度近似全监督训练梯度。

### 3.2 代码结构与数据管线

主要文件：

- [`kband/generate_maps.py`](./K-band/kband/generate_maps.py)：筛选 fastMRI 数据，调用 BART/ESPIRiT 生成灵敏度图，并产生线圈合并目标。
- [`kband/generate_data.py`](./K-band/kband/generate_data.py)：裁剪并汇总训练、验证和测试 HDF5 数据。
- [`kband/generate_masks.py`](./K-band/kband/generate_masks.py)：生成 k-band、MoDL、SSDU、square 和 vertical 等方法的采样/损失掩膜。
- [`deepinpy/main.py`](./K-band/deepinpy/main.py)：训练与推理入口。
- [`deepinpy/configs`](./K-band/deepinpy/configs/)：论文实验配置，包括 k-band、MoDL、SSDU 和对照方法。

仓库给出的处理顺序如下。路径均为占位符，执行前必须换成实际路径：

```powershell
cd K-band

python kband/generate_maps.py `
  --train <fastMRI训练目录> `
  --test <fastMRI测试目录> `
  --train_out <预处理训练输出目录> `
  --test_out <预处理测试输出目录> `
  -d knee

python kband/generate_data.py --config knee

python kband/generate_masks.py `
  --config knee `
  --type kband `
  --undersampling 2d

python deepinpy/main.py --config deepinpy/configs/Kband_Rv4_Rb4.json
```

注意：`generate_data.py`、`generate_masks.py` 和示例 JSON 中仍包含作者机器上的 `/mikRAID/...` 绝对路径。真正运行前需要复制配置并替换路径；不要直接覆盖本目录保留的固定上游源码。

`generate_data.py` 生成的网络数据 HDF5 字段为：

```text
ksp:  [Nsamples, Ncoils, NX, NY]
imgs: [Nsamples, NX, NY]
maps: [Nsamples, Ncoils, NX, NY]
```

`generate_masks.py` 生成的掩膜 HDF5 字段为：

```text
masks:      [Nsamples, NX, NY]
loss_masks: [Nsamples, NX, NY]
```

典型配置 `Kband_Rv4_Rb4.json` 使用 1600 个训练样本、400 个验证样本、400 个推理样本，`loss_function` 为 `kspace_L1`，学习率为 `1e-4`。论文实验的膝部/脑部网络配置和 unroll 数并不完全相同，应选择与目标实验一致的 JSON。

### 3.3 依赖、数据与模型

- 数据：fastMRI 多线圈 knee 或 brain 原始 HDF5；仓库不包含数据。
- 框架：PyTorch，`deepinpy/requirements.txt` 声明 `torch>=1.7.0`、PyTorch Lightning 1.0.6、test-tube 0.7.5 等旧依赖。
- 外部工具：BART、SigPy、TensorBoard。BART 的 Python 绑定和系统二进制需要单独正确配置，README 中的 `pip install bart` 不足以保证 BART MRI 工具链可用。
- GPU：示例配置面向 GPU；论文报告使用 TITAN Xp、Titan X Pascal 和 RTX 3090。
- 预训练模型：未提供。
- README 注意事项：仓库 README 仍写着旧地址 `mikgroup/Kband-recon.git`；当前实际仓库和本地代码来自 `mikgroup/K-band.git`。

## 4. SSDU 论文与代码

### 4.1 方法概要

SSDU 将每个样本已经采集到的欠采样 k 空间索引集合拆成两个不相交子集：

```text
Ω = Θ ∪ Λ，且 Θ ∩ Λ = ∅
```

- `Θ` 中的数据进入 unrolled 网络的数据一致性（data consistency，DC）模块。
- 网络输出经编码器变换回各线圈 k 空间后，只在 `Λ` 位置与真实采集值计算训练损失。
- 测试/推理阶段使用全部可用测量 `Ω` 做数据一致性。

这样不需要完整采样参考图像即可训练物理引导的重建网络。论文实验使用 10 个 unroll、每个正则化网络含 15 个 ResNet 残差块，并研究 Gaussian/Uniform 子集选择和默认 `ρ=0.4` 的拆分比例。

### 4.2 数据格式与路径配置

主要入口：

- [`train.py`](./SSDU/train.py)：训练。
- [`test.py`](./SSDU/test.py)：加载模型并重建测试切片。
- [`parser_ops.py`](./SSDU/parser_ops.py)：网络、采样率和数据尺寸参数。
- [`utils.py`](./SSDU/utils.py)：训练/测试数据和模型路径。
- [`masks/ssdu_masks.py`](./SSDU/masks/ssdu_masks.py)：Gaussian 和 Uniform 的 Θ/Λ 拆分。
- [`saved_models`](./SSDU/saved_models/)：仓库自带的 Coronal PD、R=4、100 epochs、10 unrolls、Gaussian 模型。

运行前必须编辑 `utils.py` 中的 `get_train_directory()` 和 `get_test_directory()`，替换 `...` 占位符。预期数据为：

```text
k-space HDF5:
  dataset key: kspace
  shape: [nSlices, nrow, ncol, ncoil]

灵敏度图 HDF5:
  dataset key: sens_maps
  shape: [nSlices, nrow, ncol, ncoil]

采样掩膜 MAT:
  variable name: mask
  shape: [nrow, ncol]
```

默认 Coronal PD 参数是 `320 × 368 × 15 coils`。仓库特别要求把没有信号的零填充外侧列设为数据一致性掩膜中的 1；默认 Coronal PD 代码对应前 17 列和最后 16 列。

在路径配置完成且独立旧环境已建立后，原始入口为：

```powershell
cd SSDU
python train.py
python test.py
```

参数示例：

```powershell
python train.py `
  --data_opt Coronal_PD `
  --acc_rate 4 `
  --epochs 100 `
  --nb_unroll_blocks 10 `
  --nb_res_blocks 15 `
  --mask_type Gaussian
```

### 4.3 依赖与当前源码注意事项

- `enviroment.yml` 是 Linux 环境快照：Python 3.6.10、TensorFlow GPU 1.13.1、CUDA Toolkit 10.0、cuDNN 7.6.5。
- `requirements.txt` 同时列出 `tensorflow==1.13.1` 和 `tensorflow-gpu==1.13.1`；现代 Python/CUDA/Windows 环境通常不能直接安装，应使用隔离的旧版 Linux/容器环境。
- `train.py` 和 `test.py` 将 `CUDA_VISIBLE_DEVICES` 固定为 `0`。
- `parser_ops.py` 暴露了 `--rho`，但当前 `train.py` 使用 `ssdu_masks.ssdu_masks()` 时没有传入该参数，因此实际仍使用构造器默认值 `rho=0.4`。如果未来需要其他比例，应在单独的工作副本中修正并记录变更。
- `utils.py` 内自带模型路径不是当前仓库目录结构下可直接使用的路径；测试前需将其指向 `saved_models/SSDU_Coronal_PD_100Epochs_Rate4_10Unrolls_GaussianSelection`。
- 仓库含预训练权重，但不含论文数据；作者建议公平比较时用匹配的 CNN 重新训练。

## 5. Noisier2Noise_for_recon 相关代码

### 5.1 与两篇本地论文的关系

该仓库实现 Millard 与 Chiew 的“通过 Noisier2Noise 使用欠采样数据进行自监督 MRI 重建”框架，并同时提供两种 partitioned SSDU 配置。它适合作为相关方法或 SSDU 的另一套 PyTorch/VarNet 实现，但不实现 k-band 的随机方向有限分辨率 band 采集与加权损失。

仓库内五个示例配置均以 8 倍 column-wise 欠采样为基础：

| 配置名 | 方法 |
|---|---|
| `full` | 完整参考监督基线 |
| `unweighted_n2n` | 未加权 Noisier2Noise |
| `kweighted_n2n` | 使用 `(1-K)^{-1}` 校正权重的 Noisier2Noise |
| `1D_partitioned_ssdu` | 一维 column-wise partitioned SSDU |
| `2D_partitioned_ssdu` | 二维/Bernoulli partitioned SSDU |

### 5.2 数据与入口

每个 YAML 的 `data.location` 必须指向含下列子目录的数据根目录：

```text
<data.location>/multicoil_train/
<data.location>/multicoil_val/
<data.location>/multicoil_test/
```

每个文件应为 fastMRI 风格 HDF5，并包含 `kspace`。加载器只保留线圈数与配置 `nc` 完全一致的文件；示例脑部配置使用 `nx=640`、`ny=320`、`nc=16`。

训练命令：

```powershell
cd Noisier2Noise_for_recon
python train_network.py 1D_partitioned_ssdu
```

测试命令：

```powershell
python test_network.py <log_location>
```

例如：

```powershell
python test_network.py logs/cuda/1D_partitioned_ssdu/<时间戳>
```

仓库 README 的测试说明第一条误写为 `python train_network.py log_location`，但源码和后续示例均表明正确入口是 `test_network.py`。

### 5.3 依赖与运行限制

- 框架：PyTorch VarNet，并内置一份所需的 fastMRI Python 代码。
- `requirements.txt` 固定 NumPy 1.23.5、scikit-image 0.19.3、torchvision 约 0.11.0 等版本，但没有显式列出 `torch`；建立环境时需选择与 torchvision 0.11.x 匹配的 PyTorch 版本。
- 示例面向 GPU。源码在检测不到 CUDA 时会把训练循环截断为调试模式，因此 CPU 路径不等同于完整训练。
- 训练输出位于 `logs/<cuda或cpu>/<config>/<时间戳>/`，包括配置副本、TensorBoard 指标和 `state_dict`。
- 测试生成 `results.npz`，包含 NMSE 和 SSIM 数组。
- 预训练模型：未提供。

## 6. 环境隔离建议

三套源码不要安装进同一个 Python 环境：

| 代码 | 建议隔离方式 | 主要原因 |
|---|---|---|
| K-band | 单独 PyTorch 环境 | 旧版 PyTorch Lightning/test-tube、BART 和 SigPy 组合 |
| SSDU | 单独旧版 Linux/容器环境 | Python 3.6 + TensorFlow 1.13.1 + CUDA 10.0 |
| Noisier2Noise | 单独 PyTorch 环境 | torchvision 0.11.x 与 NumPy 1.23.5 的版本约束 |

本目录没有创建这些环境。后续配置环境时，建议在仓库外保存环境文件和修改后的实验配置，以保持固定源码目录 `git status` 干净。

## 7. 数据和复现边界

- 三套方法均依赖 fastMRI 或按相同结构准备的多线圈原始 k 空间数据。
- fastMRI 数据需要按数据提供方要求单独申请/下载，本目录不包含任何患者数据。
- K-band 和 Noisier2Noise 没有提供预训练权重；SSDU 提供一套 Coronal PD 模型，但没有对应数据。
- 本次没有验证训练数值、GPU 兼容性或论文指标，因此当前状态是“源码已获取并静态核对”，不是“实验已复现”。

## 8. 许可证

### K-band

仓库根目录未见覆盖整个项目的统一 `LICENSE`。`deepinpy/LICENSE` 是 Jon Tamir 的 MIT License，但不能据此推断 `kband/` 及整个仓库都采用 MIT。使用、修改或再分发前应向仓库作者确认整体许可。

### SSDU

仓库声明代码仅许可非营利机构和美国政府机构用于教育与研究，禁止销售或未经批准的再分发；其他用途需联系 `umotc@umn.edu`。下载或执行代码即被仓库声明视为接受这些条款。本次获取已按“仅科研/教育用途”的前提执行。

### Noisier2Noise_for_recon

仓库随附 GNU GPL v3（或更高版本）许可证，详见 [`GNU_General_Public_License.txt`](./Noisier2Noise_for_recon/GNU_General_Public_License.txt)。

## 9. 引用格式

### k-band 本地论文

```bibtex
@article{wang2024kband,
  title   = {k-band: Self-supervised MRI Reconstruction Trained Only on Limited-resolution Data},
  author  = {Wang, Frederic and Qi, Han and De Goyeneche, Alfredo and Heckel, Reinhard and Lustig, Michael and Shimron, Efrat},
  journal = {arXiv preprint arXiv:2308.02958},
  year    = {2024}
}
```

代码仓库 README 还提供了 2023 年 ISMRM Workshop/Annual Meeting 摘要的 BibTeX；引用具体代码实验时可同时查阅 [`K-band/README.md`](./K-band/README.md)。

### SSDU

```bibtex
@article{yaman2020self_mrm,
  author  = {Yaman, Burhaneddin and Hosseini, Seyed Amir Hossein and Moeller, Steen and Ellermann, Jutta and Ugurbil, Kamil and Akcakaya, Mehmet},
  title   = {Self-Supervised Learning of Physics-Guided Reconstruction Neural Networks without Fully-Sampled Reference Data},
  journal = {Magnetic Resonance in Medicine},
  year    = {2020},
  volume  = {84},
  number  = {6},
  pages   = {3172--3191},
  doi     = {10.1002/mrm.28378}
}
```

### Noisier2Noise_for_recon

```bibtex
@misc{millard2022framework,
  title         = {A Framework for Self-supervised MR Image Reconstruction Using Sub-sampling via Noisier2Noise},
  author        = {Charles Millard and Mark Chiew},
  year          = {2022},
  eprint        = {2205.10278},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV}
}
```

## 10. 后续复现建议顺序

1. 先确定 knee 或 brain 数据集、序列和目标加速率。
2. 为目标代码建立独立环境，并仅做 import/帮助命令检查。
3. 准备 fastMRI k 空间、灵敏度图和掩膜，确认维度、线圈数及零填充区域。
4. 使用少量切片进行最小训练/推理检查。
5. 固定随机种子、采样掩膜、数据划分、GPU/软件版本和配置文件。
6. 再按论文规模训练并计算 NMSE、SSIM；比较时确保网络结构、数据量、输入掩膜和损失一致。
