# Novelty matrix（实施前快速检索，2026-08-02）

> 结论：三项均为候选贡献，不能据此宣称“首次”。FRCNet 已覆盖频域半监督一致性这一宽泛概念；本项目只能主张更具体的机制组合，并需在投稿前系统检索。

| 工作 | 原任务与已知机制 | 与本项目重叠 | 本项目限定差异 | 声明边界 |
|---|---|---|---|---|
| [SSDU](https://doi.org/10.1002/mrm.25752) | 原始采集 k-space 中 Ω→Θ/Λ，输入/损失点互斥的自监督重建 | 互斥信息拆分思想 | 重建幅度图像的共轭对称 Fourier views；教师在原图、学生在互补视图 | 迁移思想，不称为 SSDU 分割，也不称真实 k-space |
| [k-band](https://arxiv.org/abs/2308.02958) | 对 k-space 随机方向频带做 SGD，并以逆采样概率校正 | 随机中心频带与 Horvitz–Thompson 型加权 | 在分割概率/标签频谱上计算复数 L1 辅助损失 | 候选贡献是“随机方向频带＋逆包含概率”的谱分割监督，不是首次 Fourier loss |
| [Variable-density Noisier2Noise](https://arxiv.org/abs/2205.10278) | 匹配原采样分布的子采样及密度补偿，用于 MRI 重建 | 分布匹配掩膜 | 仅生成重建图像的半监督扰动视图，不做重建校正 | 不声称理论无偏重建性质适用于分割 |
| [FreMIM](https://openaccess.thecvf.com/content/WACV2024/html/Wang_FreMIM_Fourier_Transform_Meets_Masked_Image_Modeling_for_Medical_Image_Segmentation_WACV_2024_paper.html) | 掩码图像建模预训练，重建多阶段 Fourier spectrum | 医学分割与频域自监督 | 本项目不是预训练/频谱重建；使用互补双视图教师一致性与标签谱 band loss | 不能笼统声称首次将频域用于医学分割 |
| [FRCNet](https://papers.miccai.org/miccai-2024/340-Paper0245.html) | 半监督医学分割的 frequency-domain consistency 与区域一致性 | 频域半监督一致性高度相近 | 本项目视图在输入图像频域以共享中心＋外围互斥构造，加入分布匹配和 EMA 原图教师 | “频域 Mean Teacher”本身不构成新颖性；必须消融互补性、分布匹配和中心共享 |
| [2025 kidney uncertainty study/code](https://zenodo.org/records/15188948) | 同一肾脏 T2 数据；MCD/TTA 不确定性、radiomics、scan–rescan repeatability | 相同数据、重复测量评价 | 训练期对无标签重复 acquisition 的软 TKV 做 log-SmoothL1；无需配准、无体素一致性 | 贡献只限“训练期、无配准、无需标签的重复体积正则”，不声称首次研究肾脏重复性 |

## 三条可检验候选贡献

1. **Band-HT loss**：180 个离散方向、固定相对带宽、逆包含概率加权，Monte Carlo 期望逼近完整频谱复数 L1。
2. **DM-Complementary MT**：共享中心 8%，外围由四阶径向可变密度基础掩膜（期望保留率 70%）再互斥拆分；教师只看弱增强原图。
3. **Repeat-TKV regularization**：以真实 voxel spacing 计算 mL，对同一受试者两个 acquisition 的 log 软 TKV 做对称 SmoothL1。

这些差异只有在与普通全频 Fourier loss、随机频域 dropout、非互补双视图、普通 Mean Teacher、无 TKV 正则等强基线做完整消融后，才可能支持论文贡献。

