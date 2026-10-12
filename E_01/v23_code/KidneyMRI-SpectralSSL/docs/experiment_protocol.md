# 固定开发协议

## 研究问题

在仅使用 25% 受试者标注时，随机频带监督、分布匹配的频域双视图自监督和 scan–rescan 体积一致性，能否同时提高 T2 肾脏 MRI 分割精度、CKD 泛化能力及 TKV 重复测量稳定性？

## 开发实验

- 数据：60 subjects / 100 acquisitions；30 HC、30 CKD；每组 5 名受试者各 5 次扫描。
- 固定开发划分：seed `20260802`，subject-level 5-fold；fold 0 仅作 pilot。
- 25/50/100% 标签集合在每个外折内严格嵌套，并按 cohort/repeat status 分层。
- B0：2.5D U-Net + DiceCE。
- B1：B0 + Band-HT loss。
- B2：B1 + complementary Fourier-view Mean Teacher。
- B3：B2 + registration-free Repeat-TKV loss。
- 先导验收只要求流程结束且指标有限；不得用其作论文确认性结论。

## 统计单位

Dice、HD95、ASD 和 TKV 误差先按 acquisition 计算，再按 subject 汇总。置信区间与显著性检验以 subject 为单位。重复的 50 acquisitions 始终属于 10 名受试者；不得当作 50 名患者。重复性报告 within-subject CV、ICC(2,1) 和受试者内两两 Bland–Altman。

## 正式研究冻结点

开发阶段只允许学习率 `{3e-4, 1e-3}`。B3 额外比较 `λ_tkv={0.002,0.005,0.01,0.02}`，先取最高 Dice，再在 Dice 距最高值不超过 0.005 的配置中选择 TKV 误差最低者。官方 nnU-Net 默认日程不使用开发折选择，直接作为预先指定日程写入协议锁。

开发门禁成功后生成 `protocol_lock.json`，其中包括 suite、开发选择、正式 seed/fold、每个选择的配置哈希，以及 `src/`、全部 YAML、`pyproject.toml`、`uv.lock` 的联合 SHA-256。任何代码或配置变化都会阻止正式运行。正式实验使用 split seed `20260803`、初始化 seed `20260811/20260812/20260813`、五折。确认性矩阵是 9 个配置 × 5 折 × 3 seed = 135 次训练；B1 和 nnU-Net 100-epoch 等预算版仅用于开发说明，不进入主统计。

自定义方法统一为 100 epochs 和同一采样更新定义；nnU-Net 同时保留 100-epoch 开发版和官方默认训练版。所有方法使用相同受试者折、25%标注子集、原始空间评价和无后处理规则。失败、OOM、NaN 与超时日志保留，不静默删除。

CHAOS T2-SPIR 只保留适配器；外部数据尚未下载，因此当前单中心实验不能支持跨中心普适性声明。

## 预注册比较门槛

- 先在 acquisition 内汇总重复扫描，再在三个初始化 seed 间平均，subject 是唯一推断单位。
- 使用 10,000 次按 HC/CKD 与 repeat status 分层的 subject bootstrap，报告配对差值 95% CI。
- 使用双侧配对符号置换检验；六个 SCI 算法家族使用 Holm 校正。
- 仅当 Dice 差值 CI 下界大于 0 且 Holm `p<0.05`，并且 proposed−baseline 的 TKV 相对误差差值 CI 上界不超过 0.02，才写“超过”。
- 所有预期折、seed 和 60 名受试者不完整时，统计器强制输出“缺失/不完整，不能宣称超过”。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-02
- Verification Status: UNVERIFIED
- Version Label: protocol_v1
