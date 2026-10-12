# KidneyBand-ReliableSSL v2：候选创新点矩阵

> 检索日期：2026-08-02。三项机制均为待实验验证的候选贡献，不宣称“首次”。若预注册消融不满足保留门槛，将从论文主方法中删除并保留失败结果。

| 既有工作 | 已有机制 | 与 v2 的重叠 | v2 可被检验的限定差异 | 不可使用的表述 |
|---|---|---|---|---|
| [k-band](https://arxiv.org/abs/2308.02958) | 在真实 MRI k-space 中随机抽取方向频带，并用逆采样概率校正随机梯度 | 随机方向频带、逆概率校正 | 对重建后分割概率的左肾、右肾及双肾联合谱进行不确定性自适应抽样；保留 20% 均匀混合并用实际包含概率做 Horvitz–Thompson 校正 | “首次 Fourier 分割损失”；“真实 k-space 方法” |
| [SSDU](https://doi.org/10.1002/mrm.25752) | 将采集 k-space 的 Ω 拆为互斥 Θ/Λ，用于自监督 MRI 重建 | 互补信息视图 | 重建幅度图像的共轭对称 Fourier 扰动视图；中心 8% 共享、外围互斥，仅用于分割一致性 | “SSDU 分割”；“扫描仪 k-space” |
| [Variable-density Noisier2Noise](https://arxiv.org/abs/2205.10278) | 匹配采样分布的 Noisier2Noise 重建训练 | 可变密度频域扰动 | 只借鉴扰动分布，不继承重建无偏性结论 | “理论无偏重建性质同样适用于分割” |
| [FreMIM](https://openaccess.thecvf.com/content/WACV2024/html/Wang_FreMIM_Fourier_Transform_Meets_Masked_Image_Modeling_for_Medical_Image_Segmentation_WACV_2024_paper.html) | Fourier 域掩码图像建模预训练 | 频域自监督医学分割 | v2 不做 MIM 预训练；使用互补输入视图的预测分歧估计扰动不确定性 | “首次把频域用于医学分割” |
| [FRCNet](https://papers.miccai.org/miccai-2024/340-Paper0245.html) | DCT frequency-domain consistency 与多粒度 region similarity consistency | 频域半监督一致性高度重叠 | CSD-MT 的不确定性来自同一样本两幅互补频谱输入的 Jensen–Shannon 分歧；前景、背景和边界分别归一化，边界不被丢弃 | “频域 Mean Teacher 本身具有新颖性” |
| [Semi-URF](https://pubmed.ncbi.nlm.nih.gov/42113656/) | UDAT、BUCE 和 wavelet/cross-attention FEFF，主动利用不确定区域 | 不确定区域、频率边界监督 | CSD-MT 不按模型熵单独筛除像素，而把频谱扰动分歧与 Teacher 熵联合为可靠性；边界使用 Sobel 一致性 | “首次利用不确定边界” |
| [RandDG](https://pubmed.ncbi.nlm.nih.gov/41318360/) | GUNet、ULoFT 与输入/特征双空间随机化用于单源域泛化 | 频域扰动与外部泛化 | v2 的频域视图是互斥信息分解并服务于半监督不确定性；RandDG 作为外部 source-only 强基线 | “任意频率随机化都属于本文创新” |
| [2025 同数据集不确定性研究](https://doi.org/10.1002/mp.17995) | 推理期 MCD/TTA、不确定性、radiomics 与 scan–rescan 评价 | 相同数据及重复测量 | DZ-BiMorph 是训练期、无需配准和标签的五次 acquisition 双肾形态正则；只惩罚超过人工参照容忍区间的 CV，并用单向 PCGrad 保护主分割梯度 | “首次研究该数据集的重复性” |

## 三项候选贡献及保留门槛

1. **UABand-HT**：自适应频带抽样、条件 Horvitz–Thompson 校正和双肾解剖分解。仅当两折平均 Dice 提升至少 `0.005` 且 HD95 恶化不超过 `1 mm` 时保留。
2. **CSD-MT**：以互补频谱视图分歧作为不确定性来源，分区归一化概率监督并保留边界监督。保留门槛同上。
3. **DZ-BiMorph**：五次重复扫描的左肾、右肾和 TKV 死区 CV 约束、EMA 体积锚点与单向 PCGrad。仅当 wCV 至少下降 `0.5` 个百分点且 Dice 下降不超过 `0.005` 时保留。

完整模型相对最强双模块组合下降超过 `0.003 Dice` 时，不进入正式实验。所有被删除模块仍在消融附录中报告。

## 基线复现边界

- nnU-Net v2 3D full-resolution 使用官方 `nnunetv2` 包和默认训练器。
- Daniel-UNet、Damiano-UNet、UNet++、TCSM 与 CMMT 使用论文描述的 clean-room 实现，不复制许可不明确的作者源码。
- FRCNet 官方论文页截至检索日未给出代码仓库；工程中的基线是输出层 FDC/MRSC 近似，不能称为作者实现。
- Semi-URF 和 RandDG 配置明确记录与论文完整架构的偏差。若作者代码许可证允许，应在协议锁定前替换为官方实现；否则论文必须把结果标为 clean-room adaptation。

