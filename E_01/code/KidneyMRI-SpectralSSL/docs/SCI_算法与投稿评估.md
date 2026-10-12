# SCI 算法公平比较与投稿评估

## 当前可下的结论

现有 pilot 只能说明流程可运行。fold 0、seed `20260802`、12 名测试受试者上的 Dice 为 B0 `0.7893`、B1 `0.8051`、B2 `0.8279`、B3 `0.8142`；B2 是当前精度模型，B3 单独命名为 `KidneyBand-SSL-TKV`。这些值没有多折、多 seed 置信区间，不能写“已超过 SCI 算法”，也不能据此认定论文分区。

公开背景结果只用于定位：Daniel 等人在原始训练/测试协议下报告 `0.93±0.01`；Damiano 等人的确定性及不确定性方法约为 `0.90–0.94`。由于标注量、预处理、受试者划分和训练预算不同，这些公开分数不会参与本工程的胜负计数。

## 本地公平重跑矩阵

| 家族 | 类型 | 本工程的主要对照目的 | 是否计入六个 SCI 家族 |
|---|---|---|---:|
| Daniel-UNet | 全监督领域基线 | 同数据集经典 2D U-Net | 是 |
| Damiano-UNet/DicePlus | 全监督/不确定性 | Dice++、DP/MCD/TTA 与重复性 | 是，三种推理只计一次 |
| nnU-Net v2 2D | 强监督基线 | 100-epoch 等预算版及官方默认训练版 | 是，以官方默认版判定 |
| UNet++ 2.5D | 强监督结构基线 | 嵌套跳连与深监督 | 是 |
| TCSM | 半监督 | 变换一致性 EMA teacher | 是 |
| CMMT-Net adaptation | 半监督 | 双头互教、CutMix 与 MVAT | 是 |
| B0/B1/B2/B3 | 本文及消融 | 定位三个模块贡献 | 否 |

“SCI 算法”表示原方法发表于 SCI 收录期刊，不表示本 clean-room adaptation 与作者环境完全等价。结构和训练偏差必须与结果一同披露，详见 `THIRD_PARTY_NOTICES.md`。

## 自动判定与报告

`kidney-seg compare` 生成逐受试者表、`paired_sci_comparisons.csv`、JSON 摘要和中文报告。报告中的“超过”同时满足：

1. B2−baseline 的 Dice 配对差值 95% bootstrap CI 全部大于 0；
2. 六基线 Holm 校正后的配对置换 `p<0.05`；
3. B2−baseline 的 TKV 相对误差差值 95% CI 上界不超过绝对 `0.02`；
4. 5 折×3 seed 和 60 名受试者全部完整。

否则一律写“未证明超过”或“缺失/不完整，不能宣称超过”。HC、CKD 分层结果、参数量、显存峰值、训练及推理时间作为附表，不能用较低性能论文筛选制造优势。

## 投稿层级的条件式评估

期刊 JCR Quartile 与中科院分区是不同体系并且按年份变化。截至 2026-08-02，最新 JCR 是 2026 年发布、反映 2025 引文数据的版本；中科院采用 2025 年期刊分区表。投稿前仍必须使用单位订阅的 JCR 和中科院官方平台，以期刊、学科类别和查询日期逐项截图留档；本地代码不抓取付费数据库，也不把第三方网站的分区标签当成已核验事实。

| 确认性结果 | 论文证据强度 | 现实投稿判断 |
|---|---|---|
| 未显著超过 TCSM、CMMT 或 nnU-Net | 缺少方法优势 | 优先继续改进；若转为单中心肾脏应用，目标约为 JCR Q3–Q4 / 中科院 3–4 区 |
| 显著超过至少 3/6，且至少包括一个强半监督方法；HC/CKD 稳定、TKV 非劣 | 有方法学增益但仍是单中心 | 可评估 JCR Q3 并尝试 Q2；中科院约 2–3 区 |
| 超过 nnU-Net 官方训练版和两项强半监督方法，Dice 接近或达到 0.90，并改善重复性 | 较强的精度与生物标志物证据 | JCR Q2 更有说服力；中科院 2–3 区可尝试 |
| 希望 JCR Q1 / 中科院 1 区 | 当前证据不足 | 需要独立多中心 MRI 外部验证、完整标签效率曲线、临床盲评或更强临床终点 |

候选期刊名单必须在确认性结果完成后按论文重心筛选，并当日核验分区。当前阶段不预先承诺某一分区，也不把所对比算法发表期刊的高分区等同于本研究可投同一区。

## 一手来源

- [Daniel et al., Magnetic Resonance in Medicine 2021](https://doi.org/10.1002/mrm.28768)
- [Damiano et al., Medical Physics 2025](https://doi.org/10.1002/mp.17995) 与 [Zenodo 软件记录](https://doi.org/10.5281/zenodo.15188948)
- [nnU-Net, Nature Methods 2021](https://doi.org/10.1038/s41592-020-01008-z) 与 [官方仓库](https://github.com/MIC-DKFZ/nnUNet)
- [UNet++, IEEE TMI](https://doi.org/10.1109/TMI.2019.2959609)
- [TCSM, IEEE TNNLS](https://doi.org/10.1109/TNNLS.2020.2995319)
- [CMMT-Net, Medical Image Analysis](https://doi.org/10.1016/j.media.2024.103302)
- [Clarivate 2026 JCR 发布说明](https://clarivate.com/academia-government/blog/journal-citation-reports-2026-supporting-transparent-responsible-journal-evaluation/)
- [中国科学院文献情报中心期刊分区表入口（深圳大学图书馆说明）](https://www.lib.szu.edu.cn/er/fenqubiao)

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-02
- Verification Status: IMPLEMENTED, CONFIRMATORY RESULTS PENDING
- Version Label: sci_assessment_v1
