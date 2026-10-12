# KidneyBand-ReliableSSL v2：预注册实验协议

## 状态

当前状态为 **工程实现与筛选前验证**。旧 v1 单折结果属于开发性证据，不进入正式统计，也不能用于“超过 SCI 算法”或“具备 Q2 水平”的结论。

## 固定研究问题

在 25% 受试者标注下，UABand-HT、CSD-MT 与 DZ-BiMorph 能否同时改善 T2 肾脏 MRI 的双肾分割精度、CKD 子组表现、TKV 误差、scan–rescan 重复性及 source-only 外部泛化？

## 分阶段规则

1. 工程阶段只验证三分类、增强、BF16/FP32、断点恢复和非有限诊断。
2. 筛选阶段固定 split seed `20260802`、fold 0/1、初始化 seed `20260802`。按 `configs/benchmark/q2_v2.yaml` 的门槛决定模块去留。
3. 筛选后生成协议锁，记录代码、配置、数据清单、依赖和 GPU 哈希。锁定后任何变更必须创建新版本，不能覆盖。
4. 内部正式实验固定 split seed `20260803`，5 折及三个初始化 seed `20260811/12/13`。
5. CHAOS T2-SPIR 与 AMOS22 MRI 在锁定前只允许格式审计；外部指标只在内部方法冻结后读取，不微调、不估计目标域归一化参数。

## 统计单位与门槛

重复 acquisition 先在受试者内合并，再平均随机种子；唯一独立单位是受试者。使用 HC/CKD 分层 10,000 次 bootstrap、100,000 次配对置换和 Holm 校正。

`kidney-seg compare --q2-gate` 只有在所有门槛通过时才允许报告“具备尝试 Q2 投稿的实验条件”。缺失运行、失败运行、外部结果缺失或 100% 标签 sanity check 缺失都会使门槛失败。

## 数据边界

- 主数据的 Fourier 操作针对重建幅度图像，不是真实 k-space。
- CHAOS 训练数据按 CC-BY-NC-SA-4.0 使用；测试真值不公开。
- AMOS 只保留官方 MRI 标识范围内、公开且带标签的病例；CT 不进入 MRI 外部清单。
- 外部验证分别报告 CHAOS 与 AMOS，不合并平均值掩盖域差异。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED — SCREENING AND CONFIRMATORY RESULTS PENDING
- Version Label: q2_experiment_plan_v2

