# 自监督 MRI 重建论文：数据集与实验实施方案

## Material Passport

| 字段 | 内容 |
|---|---|
| Origin Skill | `experiment-agent` |
| Origin Mode | `plan` |
| Origin Date | `2026-08-02` |
| Verification Status | `UNVERIFIED`（尚未下载数据或运行实验） |
| Version Label | `code_plan_v1` |

> 本文档把已经确定的数据集选择转化为可复现的论文实验协议。本次只完成资料核验和方案编写，不下载 fastMRI、不安装环境、不运行训练，也不修改三个上游仓库。

## 1. 最终数据集决策

| 角色 | 数据集 | 使用内容 | 论文中的定位 |
|---|---|---|---|
| 主数据集 | fastMRI Knee | 多线圈、冠状位 PD-FS、完全采样训练/验证 k-space | 主实验、全部主要消融和统计结论 |
| 独立重复实验 | fastMRI Brain | 多线圈、轴位 T2、完全采样训练/验证 k-space | 独立训练和测试，验证结论不局限于膝关节 |
| 补充数据 | T2-weighted Kidney MRI Segmentation | 重建后的 T2 NIfTI 图像及人工肾脏分割 | 仅可做下游分割或明确标注的单线圈模拟实验 |
| 未来外部验证 | CMRx4DFlow2026 renal artery | 肾动脉 4D Flow、多通道 k-space | 等禁运期结束且真值可用后再评估；不进入当前主结论 |

采用 Knee PD-FS + Brain T2 的理由是：SSDU、k-band 和 Noisier2Noise 的论文或代码均直接面向 fastMRI 原始多线圈 k-space，因此可以在同一病例、同一采样掩膜、同一参考真值下做公平比较。肾脏 T2、CHAOS 和 AMOS22 只有重建图像或分割标注，不是真实采集的多线圈 k-space，不能作为 MRI 重建主证据。

Brain T2 必须独立训练、独立调参并独立测试。除非另行开展“只在 Knee 训练、直接在 Brain 测试”的实验，否则不得把两套分别训练的结果描述为零样本跨器官泛化。

## 2. Experiment Overview

### 2.1 研究问题

在完全相同的数据划分、欠采样模式、灵敏度图、评估实现和计算预算下，论文主方法是否能在不使用完全采样训练标签的条件下，稳定优于 SSDU、k-band 和 Noisier2Noise，并在 Knee PD-FS 与 Brain T2 上得到一致结论？

### 2.2 预注册假设

- **主要假设**：在 `R=4、6、8` 下，主方法相对最强自监督基线提高病例级 SSIM，同时降低 NMSE；差值的病例级 95% 置信区间支持同一方向。
- **次要假设**：主要效果在 Knee PD-FS 和 Brain T2 独立实验中方向一致，而不是只在一个小型子集成立。
- **边界假设**：监督模型使用完全采样训练目标，只作为性能上限，不与自监督方法放在同一标签成本类别中。

### 2.3 实验类型

1. **受控主实验**：尽可能使用相同的重建骨干网络，只改变训练监督策略或损失构造。这张表用于支持方法学结论。
2. **原作者实现复现**：分别运行三个固定提交的仓库，保留其原网络和原预处理。这张表用于证明可复现性，但不能单独用于断言损失策略优劣，因为网络、框架和训练细节同时发生了变化。

三个本地仓库固定版本如下：

| 方法 | 本地目录 | 固定提交 |
|---|---|---|
| k-band | `code/K-band` | `5542a0befb65125236b316690a53123ba0dcac63` |
| SSDU | `code/SSDU` | `060ec76d479156560e62f4d95708f7260925e31d` |
| Noisier2Noise | `code/Noisier2Noise_for_recon` | `c94567f46b2102ec9d23a9d5364afc8ed612649e` |

## 3. 数据输入、筛选和划分

### 3.1 只使用有公开参考真值的分区

- 输入使用 fastMRI 的 `multicoil_train` 和 `multicoil_val` HDF5 文件。
- 每个 HDF5 文件代表一次 acquisition；多线圈 `kspace` 通常为 `(slice, coil, height, width)`，训练/验证文件同时提供 `reconstruction_rss`。
- 官方 `multicoil_test` 含欠采样数据但不公开完全采样参考真值，不用于本论文的主要定量测试。
- Knee 只保留 HDF5 元数据明确标记为 **PD-FS** 的 acquisition；Brain 只保留元数据明确标记为 **T2** 的 acquisition。不得只凭文件名猜测序列类型。

### 3.2 固定划分

| fastMRI 原分区 | 本研究用途 | 比例/规则 |
|---|---|---|
| `multicoil_train` | 训练集 | 合格 acquisition 的约 90% |
| `multicoil_train` | 内部验证集 | 合格 acquisition 的约 10%，仅用于调参、早停和传统方法正则参数选择 |
| `multicoil_val` | 最终测试集 | 全部合格 acquisition；在模型和参数冻结前不得查看最终统计结果 |

划分单位至少为 **acquisition/volume**，同一 volume 的切片绝不跨集合。如果官方清单或获批数据元信息提供可用的受试者 ID，则先按受试者分组，再做划分；如果未提供受试者映射，不得在论文中虚称“患者级去重”，应明确写为 volume-level split，并把无法排除重复受试者列为数据限制。

采用确定性哈希划分，而不是依赖运行环境的随机顺序：

```text
group_key = patient_id（若可用），否则为 volume filename
digest = SHA256("fastmri-<anatomy>-<contrast>-20260802|" + group_key)
digest 的整数值 mod 10 == 0 -> internal validation
其余 -> training
```

所有方法共用同一份清单。建议清单字段：

```text
dataset, anatomy, contrast, official_partition, patient_or_volume_id,
filename, n_slices, n_coils, matrix_x, matrix_y, assigned_split,
inclusion_status, exclusion_reason, file_sha256
```

主分析保留每个合格 volume 的全部具有参考图像的切片，并先在 volume 内汇总指标，避免切片较多的检查获得更大统计权重。为对照 k-band 文献，另报告“每个 volume 中央 10 层”的敏感性分析；不得事后选择视觉效果最好的切片。

### 3.3 纳入与排除日志

只允许在读取数据、序列筛选、矩阵/coil 维度检查和参考真值可用性检查阶段排除样本。每个排除项必须写入 `exclusion_reason`。不得依据某种方法的重建质量排除病例。数据获取后应在冻结清单中报告每个分区的 volume 数、切片数、coil 数和矩阵尺寸分布。

## 4. 统一的预处理与欠采样协议

### 4.1 主实验公共协议

- 复数 k-space 的方向、中心化 FFT/IFFT、裁剪和强度恢复由一套公共预处理实现；所有方法读取同一中间数据。
- 灵敏度图统一由完全相同的中心校准数据通过 BART ESPIRiT 估计，并保存后复用。主协议固定为中心 `24×24` 校准区；BART 版本、命令、参数和输出文件哈希写入运行记录。
- 每个切片以其完整多线圈 k-space 的最大绝对值归一化；评估前恢复尺度。若方法必须使用不同的内部归一化，需在输出端恢复到统一尺度并单独披露。
- 参考图像采用 fastMRI 官方 `reconstruction_rss`；预测图像转为幅度图并做与官方评估一致的中心裁剪和动态范围处理。
- 受控主实验中，所有自监督方法尽可能使用同一 physics-guided unrolled/VarNet 骨干、相同初始化、优化器、更新步数和模型选择规则。无法统一的差异必须列入偏差表。

原作者实现复现应保留其论文/仓库预处理，并单列结果，不与上述公共协议结果混表。这样既保留方法忠实度，也避免把预处理差异当成算法增益。

### 4.2 采样掩膜

每个测试切片生成并永久保存同一份采样掩膜，所有方法按文件读取，不得在训练或测试时各自随机生成。

| 维度 | 加速率 | 固定校准区 | 外围采样 |
|---|---:|---|---|
| 1D Cartesian | 4、6、8 | 中心连续 24 条 phase-encoding 线 | 可变密度随机采样至目标采样点数 |
| 2D Cartesian | 4、6、8 | 中心 `24×24` | 二维可变密度随机采样至目标采样点数 |

实际加速率按 `总 k-space 点数 / 已采样点数` 报告，不能只写名义 R。掩膜种子由下式决定：

```text
SHA256(filename | slice_index | mask_type | R | 20260802)
```

采集掩膜记为 Ω；自监督损失使用的 Θ/Λ 或其他子掩膜必须从 Ω 内拆分，且不得接触 Ω 之外的测试信息。SSDU 的损失掩膜比例 `ρ` 以论文/仓库推荐值作为主设置，并在 `ρ=0.1、0.2、0.4` 做消融。采集掩膜和损失掩膜分别保存，避免把二者混淆。

## 5. 比较方法和公平性

主表至少包括：

1. Zero-filled RSS。
2. 并行成像基线：CG-SENSE/ESPIRiT。
3. 压缩感知基线：ESPIRiT + L1-wavelet（正则参数仅在内部验证集选择）。
4. SSDU。
5. k-band。
6. Noisier2Noise，主设置采用 `kweighted_n2n`；未加权版本作为消融。
7. 使用完全采样训练目标的 supervised VarNet/同骨干模型，明确标注为“监督上限”。
8. 论文主方法。

公平性规则：

- 同一 anatomy、R 和 mask_type 下，方法共享病例列表、Ω、灵敏度图、输入尺度和评估代码。
- 每种学习方法训练 3 个固定种子：`20260802、20260803、20260804`。
- 受控比较匹配参数量、训练更新次数、训练数据和超参数搜索预算；不要仅匹配 epoch，因为不同 dataloader 的每个 epoch 更新数可能不同。
- 超参数只能根据内部验证集确定；最终测试集只运行冻结后的配置。
- 官方 SSDU 仓库自带模型仅用于代码完整性检查或文献复现，不能替代在本研究划分上的重新训练。
- 若某仓库因 TensorFlow/PyTorch、输入形状或灵敏度图接口不同而需要适配，适配层必须只做格式变换，并通过“前后数据数值一致”单元测试；不得暗中改变采样、归一化或目标定义。

## 6. 实验矩阵与消融

### 6.1 主实验矩阵

| Anatomy | Contrast | Mask | R | Seeds | 输出 |
|---|---|---|---|---:|---|
| Knee | PD-FS | 1D、2D | 4、6、8 | 3 | 全部基线、主方法、监督上限 |
| Brain | T2 | 1D、2D | 4、6、8 | 3 | 与 Knee 完全相同的方法集合 |

### 6.2 预先规定的消融

- 损失掩膜比例：`ρ=0.1、0.2、0.4`。
- 训练数据规模：冻结同一排序后使用 `25%、50%、100%` 的训练 volumes。
- 采样模式：1D 与 2D；加速率：`R=4、6、8`。
- Noisier2Noise：unweighted 与 k-space weighted loss。
- 评估切片范围：全部切片与中央 10 层。

为控制计算量，完整方法比较在两个 anatomy 的全部 R/mask 条件运行；训练规模、ρ 和 loss weighting 的细粒度消融先固定在 Knee PD-FS、1D、R=4，关键结论再在 Brain T2、1D、R=4 和两个 anatomy 的 R=8 条件复核。

## 7. 指标与统计分析计划

### 7.1 指标

- **主要指标**：SSIM。
- **次要指标**：NMSE、PSNR。
- 可选补充：HFEN 或边缘保持指标，但不得替换预注册的主要指标。

对每个切片计算指标，再在同一 volume 内取均值，最终每个 volume 只贡献一个病例级数值。论文同时给出逐 volume 原始结果文件，不把相关的逐切片观测当作独立样本扩大显著性。

### 7.2 重复训练的处理

每种方法先对同一 volume 的 3 个训练种子结果取均值，再进行病例级配对统计检验；另报告各随机种子的总体均值与标准差，用于展示训练稳定性。随机种子不是独立患者，不能增加统计样本量。

### 7.3 置信区间和检验

- 对病例/volume 做 10,000 次聚类 bootstrap，给出指标及方法间配对差值的 95% CI。
- 同一 anatomy、mask 和 R 下多方法总体比较使用 Friedman 检验。
- 预先指定的两两比较使用 Wilcoxon signed-rank 检验，并在同一比较族内做 Holm 校正。
- 同时报告中位配对差值和 matched rank-biserial effect size，不只报告 p 值。
- Knee 和 Brain 分开检验、分开报告；可提供合并的描述性森林图，但不得用合并结果掩盖某个 anatomy 的失败。

支持主要结论的最低条件是：相对预先指定的最强自监督基线，主要指标在两个 anatomy 上方向一致，配对差值 95% CI 支持该方向，且经多重比较校正后的结果与效应量共同支持实际改善。若只在某个 R 或某个 anatomy 成立，应把结论限定在该条件。

## 8. 图像展示与可选临床评价

- 每个 anatomy 展示预先按病例 ID 固定的低、中、高误差病例，而不是事后挑选最佳案例。
- 每例展示参考图、zero-filled、主要基线、主方法、监督上限、同一窗宽窗位的绝对误差图，以及同一区域局部放大图。
- 图注写明 anatomy、序列、R、mask、实际加速率和病例 ID 的匿名研究编号。
- 若具备影像科资源，至少两名评阅者在方法名隐藏、图像顺序随机的条件下进行 5 分制整体质量/伪影评分，并报告评阅者一致性；该部分在伦理和读片方案确定前属于可选分析。

## 9. 运行、监控和预期产物

本次不创建环境或启动运行。真正执行时，建议把实验产物放在上游仓库之外，例如：

```text
D:/30 医学论文/01/experiments/
  manifests/        # 冻结的数据清单、排除日志和文件哈希
  masks/            # Ω 与自监督损失子掩膜
  sensitivity_maps/ # 统一生成的 ESPIRiT maps
  configs/          # 每次运行的完整配置
  envs/             # conda/pip lock、CUDA/CUDNN 和 BART 版本
  logs/             # stdout、训练曲线和硬件信息
  checkpoints/      # 模型权重
  per_case/         # 每个 volume/seed/method 的指标
  summaries/        # 汇总表、CI、检验和效应量
  figures/          # 重建、误差图、局部放大图和统计图
```

每次运行记录：Git commit、dirty status、数据清单哈希、mask 哈希、配置哈希、随机种子、GPU/驱动、框架版本、开始/结束时间和退出码。每 60 秒检查日志；连续 20 分钟无日志/无 checkpoint 更新、出现 NaN/Inf、GPU OOM 或进程退出时标记失败并保留现场，不自动用新参数重试。单次训练默认硬超时 72 小时，正式大规模运行前用少量 volumes 做显存和耗时预检，再根据硬件在注册表中更新超时。

预期机器可读结果表至少含：

```text
anatomy, contrast, volume_id, slice_scope, mask_type, nominal_R, actual_R,
method, seed, nmse, psnr, ssim, checkpoint_sha256, config_sha256
```

在所有预定运行、失败说明、统计表和图像生成完成前，本文档的实验状态保持 `UNVERIFIED`，不得把预期结果写成已观察结论。

## 10. 肾脏数据的使用边界

- [T2-weighted Kidney MRI Segmentation](https://zenodo.org/records/5153568) 可用于“重建质量对下游肾脏分割的影响”或明确标注的合成实验。对 NIfTI 图像做 FFT 只能得到**回顾性单线圈模拟 k-space**，不能称为真实采集多线圈 k-space。
- [CHAOS](https://zenodo.org/records/3431873) 与 [AMOS22](https://zenodo.org/records/7262581) 可用于腹部/肾脏图像级任务，但不进入重建主表。
- [CMRx4DFlow2026](https://cmrx.chihucloud.com/2026/data.html) 是肾动脉 4D Flow MRI，不等价于常规肾脏结构 MRI；其 renal artery 验证/测试真值受限，且研究复用禁运至 2026 年 12 月，当前只列入未来工作。
- 如果论文题目必须以肾脏重建为核心，应取得自有或合作医院的原始肾脏多线圈 k-space，并完成伦理审批、数据使用协议和跨中心验证；现有公开图像级肾脏数据不足以支撑“真实肾脏并行 MRI 重建”的主结论。

## 11. 数据许可、引用和链接

- fastMRI 数据门户与申请：[https://fastmri.med.nyu.edu/](https://fastmri.med.nyu.edu/)
- fastMRI HDF5 结构说明：[facebookresearch/fastMRI data README](https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md)
- SSDU：[byaman14/SSDU](https://github.com/byaman14/SSDU)
- k-band：[mikgroup/K-band](https://github.com/mikgroup/K-band)
- Noisier2Noise：[charlesmillard/Noisier2Noise_for_recon](https://github.com/charlesmillard/Noisier2Noise_for_recon)

fastMRI 需要申请访问并接受 Data Sharing Agreement。数据仅按获批的内部科研/教育用途使用，不得重新分发；论文应按门户要求引用 fastMRI 论文并使用规定的致谢语。研究开始前保存当时适用的协议版本、批准邮件和数据版本信息，图像展示也须遵守协议。

建议至少引用：

1. Zbontar J, et al. *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI*. arXiv:1811.08839.
2. Knoll F, et al. *fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning*. Radiology: Artificial Intelligence, 2020.
3. Yaman B, et al. SSDU 原论文。
4. Wang S, et al. k-band 原论文。
5. Noisier2Noise 的原始方法论文及所用实现仓库。

本规划文档使用了 AI 辅助的源码与公开资料核验；未生成、修改或选择任何实验结果。投稿时应按期刊政策披露 AI 辅助范围。

## 12. 验收清单

- [ ] Knee PD-FS 和 Brain T2 均由 HDF5 acquisition 元数据筛选，且冻结 volume 清单及哈希。
- [ ] 若无受试者映射，论文明确写 volume-level split，不虚称 patient-level split。
- [ ] 同一 volume 的切片没有跨训练、内部验证和最终测试集。
- [ ] 官方 validation 被保留为最终有参考真值的测试集，官方 test 不用于主要定量结果。
- [ ] 所有方法使用相同 Ω、灵敏度图、预处理、裁剪和指标实现。
- [ ] 采集掩膜与 SSDU/k-band/Noisier2Noise 的损失掩膜分开保存并可追溯。
- [ ] 受控主实验与原作者仓库复现分表，监督上限明确标注其标签优势。
- [ ] 统计单位是病例/volume，95% CI 和配对检验均按病例聚类。
- [ ] 主结论同时核对 Knee 与 Brain，不把独立训练误称为零样本泛化。
- [ ] 所有肾脏图像级或模拟数据在标题、表格和图注中醒目标明“非真实原始多线圈 k-space”。
- [ ] 三个上游仓库保持固定提交且 `git status --short` 为空。
- [ ] 数据访问、引用、致谢和图像展示符合 fastMRI Data Sharing Agreement。

