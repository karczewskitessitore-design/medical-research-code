# 肾脏 MRI 分割论文：数据、实验与统计实施指南

## Material Passport

| 字段 | 内容 |
|---|---|
| Origin Skill | `experiment-agent` |
| Origin Mode | `plan` |
| Origin Date | `2026-08-02` |
| Verification Status | `UNVERIFIED`（链接和公开元数据已核验，数据与实验尚未下载或运行） |
| Version Label | `code_plan_v1` |

> 本文档将肾脏 MRI 分割论文方案固化为可执行协议。本次只编写指南，不下载数据、不安装环境、不运行模型，也不修改 `K-band`、`SSDU` 或 `Noisier2Noise_for_recon` 上游仓库。

## 1. 数据集最终选择

主数据集采用 University of Nottingham 发布的 **T2-weighted Kidney MRI Segmentation**。该数据包含 **60 名受试者的 100 次 T2 加权腹部 MRI 扫描**：30 名健康对照（HC）和 30 名慢性肾病患者（CKD）；其中 5 名 HC 与 5 名 CKD 受试者在同一次检查中各重复扫描 5 次。因此论文中应写“100 次扫描/acquisitions”，不能写成“100 名独立受试者”。

数据提供人工肾脏分割，可用于肾脏分割、总体肾体积（TKV）和重复扫描稳定性研究。它不提供扫描仪原始多线圈 k-space，因此论文任务确定为 **MRI 图像分割**，不能把它用于真实采集的自监督 MRI 重建主实验。

## 2. 官方入口与下载文件

本次只核验链接，不实际下载。三个直链在 2026-08-02 均返回 HTTP 200。

| 文件 | 内容 | 大小 | 官方 MD5 | 下载链接 |
|---|---|---:|---|---|
| `CKD.zip` | CKD 扫描及对应标注 | 83.3 MB | `8fe112954083212cbee80a6087db84c5` | [直接下载](https://zenodo.org/records/5153568/files/CKD.zip?download=1) |
| `Healthy_Control.zip` | 健康对照扫描及对应标注 | 77.0 MB | `561badbc875874c5e69cf725a8545330` | [直接下载](https://zenodo.org/records/5153568/files/Healthy_Control.zip?download=1) |
| `subject_information.csv` | 受试者与扫描信息 | 1.2 kB | `3bc3d595339cb7c1f9208a07f7335ef5` | [直接下载](https://zenodo.org/records/5153568/files/subject_information.csv?download=1) |

- 数据集主页与引用信息：[Zenodo record 5153568](https://zenodo.org/records/5153568)
- 配套论文：[Daniel et al., Magnetic Resonance in Medicine, 2021](https://doi.org/10.1002/mrm.28768)
- PubMed：[PMID 33755256](https://pubmed.ncbi.nlm.nih.gov/33755256/)

公开论文报告的采集条件为 3T Philips Ingenia、冠状位 2D T2-weighted HASTE、约 13 层、约 `1.5×1.5×5 mm³` 体素并带 0.5 mm 层间隔。实际文件格式、方向、spacing、图像/掩膜命名和标签取值必须在下载后从文件与元数据中重新核验，不能仅按论文文字硬编码。

## 3. 下载后的第一道数据审计

数据下载后先建立只读原始数据区，不直接改名、转码或覆盖原文件。审计顺序如下：

1. 对三个下载文件计算 MD5，与上表逐项匹配；不匹配时停止解压和实验。
2. 解压到独立目录，生成病例级清单，保存原始相对路径和文件 SHA-256。
3. 使用 `subject_information.csv` 建立 `subject_id → cohort → acquisition_id` 映射，确认重复扫描属于同一受试者。
4. 核对每个图像是否有且只有一个对应掩膜，检查 shape、affine/orientation、spacing 和空间覆盖完全一致。
5. 统计图像尺寸、层数、体素间距、强度范围、标签唯一值、左右肾是否为一个或两个标签，以及空掩膜/损坏文件。
6. 将异常写入排除日志；不得因某种模型表现差而事后删除病例。

建议冻结的清单字段：

```text
subject_id, acquisition_id, cohort, repeat_index, image_path, mask_path,
image_shape, voxel_spacing, label_values, assigned_fold,
inclusion_status, exclusion_reason, image_sha256, mask_sha256
```

若数据只提供双肾合并的二值掩膜，主任务就固定为 bilateral whole-kidney segmentation；不得未经重新标注而声称区分左肾、右肾、皮质、髓质、肿瘤或囊肿。

## 4. 研究问题与实验目标

### 4.1 主要研究问题

在严格受试者级分组的条件下，2D、3D 和 Transformer/混合分割模型对 HC 与 CKD 肾脏的分割精度、TKV 测量误差及重复扫描稳定性有何差异？

### 4.2 预注册假设

- 主方法相对 2D U-Net 基线提高受试者级 Dice，并降低 TKV 相对误差。
- 改善方向在 HC 与 CKD 两组中一致，而不是由健康组的规则形态单独驱动。
- 在 10 名重复扫描受试者上，自动分割得到的 TKV 具有可重复性；重复 acquisition 不被当作独立患者扩大统计显著性。

### 4.3 模型集合

| 角色 | 模型 | 说明 |
|---|---|---|
| 文献基线 | 2D U-Net | 与原论文的 2D CNN 路线相近；逐层训练和推理 |
| 强基线 | nnU-Net v2 3D full-resolution | 作为主要工程基线，保留其自动配置记录 |
| 混合架构 | SwinUNETR | 不使用未披露的外部医学数据预训练；如使用预训练，必须单列结果 |
| 论文方法 | 用户的主方法 | 使用与基线相同的 folds、输入、增强预算和模型选择规则 |

如果论文尚无新方法，以上模型只能形成基准研究或重复性研究；不能仅凭更换网络名称宣称算法创新。

## 5. 无泄漏的分组五折交叉验证

### 5.1 固定折分配

统计单位为 **60 名受试者**，不是 100 次扫描或约 1299 张二维切片。固定随机种子为 `20260802`。

每一折测试集包含：

- 6 名 HC：5 名单次扫描受试者 + 1 名五次重复扫描受试者；
- 6 名 CKD：5 名单次扫描受试者 + 1 名五次重复扫描受试者；
- 合计 12 名受试者、20 次 acquisitions。

五折恰好覆盖全部 60 名受试者。所有 5 次重复扫描必须跟随其 `subject_id` 进入同一折。优先使用按 `cohort` 和 `repeat_status` 分层的确定性分组算法；生成后人工验证每折结构，再冻结为 `folds.csv`，所有模型共用。

### 5.2 训练与模型选择

- 每折的 48 名非测试受试者中，再按同一分组原则固定一部分作为内部验证；测试折不得用于早停、阈值或超参数选择。
- 重复扫描受试者在训练集中会有 5 次 acquisition。采用受试者均衡采样：每个 epoch 每名受试者权重相同，重复 acquisition 总权重为 1，避免其贡献变成单次扫描受试者的 5 倍。
- 对每个模型运行 3 个种子：`20260802、20260803、20260804`。所有种子使用完全相同的五折清单。
- 最终结果来自 out-of-fold 预测，每名受试者只由未见过该受试者的模型产生预测。

## 6. 公共预处理与训练约束

- 所有模型读取同一病例清单并使用同一图像方向、重采样、归一化和标签定义。
- 下载后以训练折的 spacing 分布确定公共目标 spacing；参数只由训练折计算并写入配置，禁止使用测试折统计量。
- 强度预处理固定为：训练折内确定非零区域百分位裁剪规则，随后逐 volume z-score；测试时只应用冻结规则。
- 2D 模型只做平面内重采样；3D 模型必须考虑原数据明显的层间各向异性。所有预测恢复到参考掩膜的原始空间后再计算指标。
- 主要结果不使用连通域后处理；“保留最大的两个连通区域”只能作为预注册消融，避免后处理掩盖模型差异。
- 损失函数、优化器、最大更新次数、早停规则和超参数搜索预算在模型间尽量匹配。所有版本、配置、commit 和随机种子均保存。

统一数据增强分为三个级别，用于消融：

| 级别 | 内容 |
|---|---|
| None | 仅公共预处理 |
| Spatial | 平面内翻转、`±10°` 旋转、`0.9–1.1` 缩放 |
| Full | Spatial + gamma/强度扰动、轻度高斯噪声和 bias-field |

不做大角度 through-plane 旋转，以免在厚层方向制造不真实插值。图像与掩膜的空间变换必须共享同一随机参数，掩膜只用 nearest-neighbor 插值。

## 7. 评价指标与边界情况

### 7.1 主要与次要指标

- **主要指标**：Dice。
- **次要指标**：HD95、平均表面距离（ASD）、TKV 相对误差。
- TKV 必须由预测体素数乘以原始物理 voxel volume 计算，单位为 mL，不能只比较像素数。

每次 acquisition 先得到指标；普通受试者保留一个值，重复扫描受试者的 5 次结果先在受试者内汇总后再进入总体模型比较。因此总体统计中每名受试者权重相同。

若真值非空但预测为空：Dice 记为 0、TKV 相对误差记为 100%，并单独计为 segmentation failure；HD95/ASD 不得静默删除，应在预注册分析代码中采用图像物理对角线作为保守惩罚值，同时报告失败数量。

### 7.2 统计分析

- 按受试者进行 10,000 次 bootstrap，给出各指标和模型间配对差值的 95% CI。
- 多模型总体比较使用 Friedman 检验；预先指定的两两比较使用 Wilcoxon signed-rank，并做 Holm 校正。
- 同时报告配对差值和效应量，不能只报告 `p < 0.05`。
- HC、CKD 和总体分别报告；总体结果不能替代疾病分层结果。
- 3 个训练种子的同一受试者结果先取均值再做病例级检验；种子不是新增受试者。

### 7.3 重复扫描稳定性

对 5 名 HC 和 5 名 CKD 的五次 acquisition 单独报告：

- 受试者内 TKV 变异系数；
- ICC 及其 95% CI；
- 预测 TKV 与人工 TKV 的 Bland–Altman 分析；
- HC 和 CKD 分层的重复性结果。

这些 50 次重复扫描用于测量精度，不作为 50 个独立病例。

## 8. 预先规定的消融与泛化实验

1. **训练规模**：每折使用 `25%、50%、100%` 训练受试者，采用固定、嵌套的受试者子集并保持 HC/CKD 比例。
2. **数据增强**：None、Spatial、Full。
3. **维度**：2D 与 3D，讨论厚层各向异性对结果的影响。
4. **健康到疾病泛化**：只用 HC 受试者训练/验证，冻结后在全部 CKD 受试者上测试；该结果单列，不能与混合队列五折结果混合。
5. **重复性**：比较不同模型的 TKV scan-rescan CV 和 ICC。
6. **后处理**：无后处理与最大两个连通区域，仅作为补充消融。

## 9. CHAOS 外部验证

CHAOS MRI 训练集包含 20 个有公开真值的腹部 MRI 病例，T1-DUAL 与 T2-SPIR 序列标注肝、脾和左右肾。其人群、序列、标签协议和扫描设备与 Nottingham 数据不同，因此只作为外部域验证。

- 官方记录：[CHAOS Zenodo](https://zenodo.org/records/3431873)
- 训练集直接下载：[CHAOS_Train_Sets.zip，890.8 MB](https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1)
- 官方 MD5：`df21053002a1cc86df918a87da3b2c19`

主外部实验为 zero-shot：仅做预先冻结的方向、重采样和强度归一化，不在 CHAOS 标注上微调。只评价 T2-SPIR 的整肾区域，并先建立标签映射；如果 Nottingham 只提供双肾二值标签，则把 CHAOS 左右肾合并后评价。任何 few-shot fine-tuning 必须单列，不得称为外部零样本验证。

CHAOS 公开测试集不提供分割真值，不能用于本地有参考定量评价。

## 10. 实验产物与验收

后续执行时建议把新实验放在上游仓库之外：

```text
D:/30 医学论文/01/kidney_segmentation_experiments/
  raw/              # 原始压缩包及只读解压内容
  manifests/        # cases.csv、folds.csv、排除日志和哈希
  configs/          # 预处理与模型配置
  predictions/      # out-of-fold 原空间预测
  per_subject/      # acquisition级及subject级指标
  statistics/       # CI、检验、效应量、ICC、Bland–Altman
  figures/          # 代表性结果、误差与统计图
  logs/             # 环境、commit、种子和运行日志
```

每次运行至少记录数据清单哈希、folds 哈希、代码 commit、dirty status、环境锁文件、GPU/驱动、配置、随机种子和退出码。实验完成的最低标准不是达到某个事后设定的 Dice，而是：

- 60 名受试者均有且只有一组有效 out-of-fold 预测；
- 无切片级或重复扫描级数据泄漏；
- 三个种子和全部预注册模型/消融均完成，或有明确失败说明；
- HC、CKD、总体和重复扫描结果均按受试者汇总；
- 指标、95% CI、效应量和校正后的统计检验可由保存的预测重新生成；
- 外部 CHAOS 结果明确标注为跨数据集验证。

## 11. 许可和论文引用

公开可下载不自动等于无限制再许可。开始实验前应保存 Zenodo 当时显示的权利/许可信息；若记录未给出明确标准许可证，应联系数据作者确认论文使用、衍生掩膜或模型发布范围。不要重新分发原始医学影像。

论文至少引用数据集 DOI 与配套论文：

```bibtex
@dataset{daniel2021kidneymri,
  author    = {Daniel, Alexander J. and Buchanan, Charlotte E. and
               Allcock, Thomas and Scerri, Daniel and Cox, Eleanor F. and
               Prestwich, Benjamin L. and Francis, Susan T.},
  title     = {T2-weighted Kidney MRI Segmentation},
  year      = {2021},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.5153568},
  url       = {https://doi.org/10.5281/zenodo.5153568}
}
```

```bibtex
@article{daniel2021automated,
  author  = {Daniel, Alexander J. and others},
  title   = {Automated renal segmentation in healthy and chronic kidney
             disease subjects using a convolutional neural network},
  journal = {Magnetic Resonance in Medicine},
  year    = {2021},
  volume  = {86},
  number  = {2},
  pages   = {1125--1136},
  doi     = {10.1002/mrm.28768}
}
```

本指南使用 AI 辅助核验公开元数据和设计实验协议；没有下载、查看或处理参与者数据，也没有生成实验结果。投稿时应按目标期刊政策披露 AI 辅助范围。

## 12. 最终检查表

- [ ] 三个主数据文件 MD5 与官方值一致。
- [ ] 数据被正确识别为 60 名受试者、100 次 acquisitions。
- [ ] 图像与掩膜空间信息一致，标签含义已从实际文件确认。
- [ ] 五折中同一受试者的所有重复扫描始终位于同一折。
- [ ] 每折同时包含 HC、CKD、单次和重复扫描受试者。
- [ ] 训练采样按受试者均衡，重复扫描不获得五倍权重。
- [ ] 所有模型共用 folds、预处理、增强预算和评价代码。
- [ ] Dice、HD95、ASD 与 TKV 误差均在原始物理空间计算。
- [ ] 统计单位是受试者；切片、acquisition 和随机种子不被伪装成独立病例。
- [ ] HC、CKD、总体、健康→CKD 和重复扫描结果分别报告。
- [ ] CHAOS 仅作外部域验证，未使用公开测试集做无真值定量分析。
- [ ] 论文明确说明该数据不含真实原始多线圈 k-space。

