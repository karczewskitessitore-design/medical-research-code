# KidneyBand-ReliableSSL v2：少标注 T2 肾脏 MRI 分割与可靠 TKV

当前主开发版本为 `schema_version: 2`：三分类输出（背景、左肾、右肾），包含 UABand-HT、CSD-MT、DZ-BiMorph、稳定 BF16/FP32 损失、完整断点恢复、外部 source-only 评价和严格 Q2 门槛报告。旧 v1 配置、命令和结果继续保留，但不会进入 v2 确认性统计。

```powershell
# 审计左右肾语义与连通分量
.\.venv\Scripts\kidney-seg.exe audit-bilateral `
  --manifest "D:\30 医学论文\01\data\T2KidneyMRI\audit\manifest.csv" `
  --output-dir "D:\30 医学论文\01\data\T2KidneyMRI\audit\bilateral_v2"

# 两折单种子筛选；只有筛选完成后才允许锁定正式协议
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\q2_v2.yaml --phase screen --resume
.\.venv\Scripts\kidney-seg.exe lock-protocol --suite configs\benchmark\q2_v2.yaml
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\q2_v2.yaml --phase internal --resume

# 官方外部数据；AMOS 约 24.2 GB，下载器支持保留断点
.\.venv\Scripts\kidney-seg.exe prepare-external --dataset chaos --data-root "D:\30 医学论文\01\data\CHAOS"
.\.venv\Scripts\kidney-seg.exe prepare-external --dataset amos --data-root "D:\30 医学论文\01\data\AMOS22"
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\q2_v2.yaml --phase external --resume

# 缺失任何预注册运行都会使 Q2 gate 失败
.\.venv\Scripts\kidney-seg.exe compare `
  --benchmark-dir "D:\30 医学论文\01\experiments\KidneyMRI-SpectralSSL-v2\benchmarks\kidneyband_reliable_ssl_q2_v2" `
  --suite configs\benchmark\q2_v2.yaml --q2-gate
```

方法边界和筛选规则见 [v2 novelty matrix](docs/novelty_matrix_v2.md) 与 [v2 预注册协议](docs/q2_protocol_v2.md)。频域操作仅针对重建后的幅度图像，不是真实扫描仪 k-space。

## v1 背景与兼容说明

这是一个独立、clean-room 的 PyTorch/MONAI 科研工程。它针对 [T2-weighted Kidney MRI Segmentation](https://zenodo.org/records/5153568) 的 60 名受试者、100 次扫描，实现 2.5D U-Net 基线和三项可独立开关的候选方法：

1. 随机方向 k-band 与逆包含概率加权的谱监督；
2. 分布匹配、互补图像频域视图的 EMA Mean Teacher；
3. 训练期、无配准、无标签的 scan–rescan TKV 一致性。

频域操作只作用于重建后的幅度图像，不能称为扫描仪真实 k-space。三项方法均是待完整文献检索和正式实验确认的候选贡献，不宣称“首次”。

## 快速开始

```powershell
./scripts/bootstrap.ps1
.\.venv\Scripts\kidney-seg.exe download --data-root "D:\30 医学论文\01\data\T2KidneyMRI"
.\.venv\Scripts\kidney-seg.exe audit --data-root "D:\30 医学论文\01\data\T2KidneyMRI"
.\.venv\Scripts\kidney-seg.exe make-folds --manifest "D:\30 医学论文\01\data\T2KidneyMRI\audit\manifest.csv" --seed 20260802
.\.venv\Scripts\kidney-seg.exe train --config configs\pilot\B0.yaml --fold 0 --seed 20260802
.\.venv\Scripts\kidney-seg.exe evaluate --run-dir "D:\30 医学论文\01\experiments\KidneyMRI-SpectralSSL\B0\fold0_seed20260802"
.\.venv\Scripts\kidney-seg.exe pilot
```

引导脚本先按 `uv.lock` 同步第三方依赖，再把本项目安装为普通 wheel。这样可避免 Windows 在含中文的项目路径下读取 editable `.pth` 时出现编码错误。

所有 YAML 都固定包含 `data、split、model、spectral_band、mean_teacher、repeat_tkv、optimizer、runtime` 七个区块。默认数据不进入 Git；实验输出位于项目外的 `D:\30 医学论文\01\experiments\KidneyMRI-SpectralSSL`。

## 数据与泄漏防护

- 拆分和标注比例均以受试者为单位；同一人的全部 acquisitions 位于同一折。
- 未标注样本的 Dataset 返回值没有 `mask`、`mask_path` 或任何标签派生字段。
- 默认预处理：RAS、平面内 256×256、非零区域 0.5–99.5 百分位裁剪、逐 volume z-score；mask 最近邻插值。
- RAS 后依据 voxel spacing 识别厚层轴，并仅在模型张量中把它移到最后一维；保存预测时逆变换回 RAS 和原始 affine。
- 2.5D 输入由相邻三层组成，边缘层复制；仅预测中心层。
- 主结果不应用连通域后处理。

## 训练与评价

默认目标为 `DiceCE + 0.05 * band + ramp * consistency + 0.02 * ramp * TKV`。100 epochs、AdamW、学习率 `1e-3`、weight decay `1e-5`、batch size 8、AMP、梯度裁剪 1.0。Student 与 EMA Teacher 的最终 checkpoint 都会保存；正式评价默认使用 Teacher。

评价输出按 acquisition 与 subject 汇总 Dice、HD95、ASD、TKV 误差；重复扫描额外报告受试者内 TKV CV、ICC(2,1) 和受试者内两两 Bland–Altman。逐切片不会作为独立统计样本。

## 先导边界

`pilot` 仅运行 fold 0、seed 20260802、25% 标注的 B0–B3。每项硬超时 12 小时、每 60 秒 heartbeat、20 分钟无日志或 NaN/OOM 即失败且不自动重试。先导只验收流程与有限数值，不作为论文确认性结论。

## SCI 公平对比

基准套件含 Daniel-UNet、Damiano-UNet/DicePlus（同一模型同时评价确定性、MCD、TTA）、nnU-Net v2 2D、UNet++ 2.5D、TCSM 和 CMMT-Net clean-room adaptation。开发阶段只使用旧 seed 的 fold 0；开发门禁完成后会写入带 suite、方法配置及源码树联合 SHA-256 的 `protocol_lock.json`，确认性阶段才允许使用新划分 seed `20260803`。

```powershell
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\sci_suite.yaml --phase dev --resume
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\sci_suite.yaml --phase confirmatory --resume
.\.venv\Scripts\kidney-seg.exe compare --benchmark-dir "D:\30 医学论文\01\experiments\KidneyMRI-SpectralSSL\benchmarks\sci_fair_comparison_v1\confirmatory"
```

确认性矩阵固定为 9 个配置 × 5 折 × 3 初始化种子，共 135 次训练。统计器要求所有预注册运行及 60 名受试者完整，先对 acquisition、再对 seed 汇总，以 10,000 次分层 subject bootstrap、配对置换检验和六基线 Holm 校正判定。只有 Dice 差值 95% CI 全部大于 0、Holm `p<0.05`，且 TKV 相对误差差值的 95% CI 上界不超过 `0.02`，报告才会输出“超过”；缺失或失败运行不能形成优势结论。

进一步内容见 [英文说明](README_EN.md)、[实验协议](docs/experiment_protocol.md)、[SCI算法与投稿评估](docs/SCI_算法与投稿评估.md)、[novelty matrix](docs/novelty_matrix.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。
