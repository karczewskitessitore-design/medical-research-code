# v2.1 对比算法 AutoDL 运行指南

## 可执行方法

| ID | 方法 | 输入 | 实现性质 |
|---|---|---:|---|
| `V21_B0plus_5slice` | 匹配B0+ | 5层 | 统一clean-room基线 |
| `KidneyBand_ReliableSSL_v2_1` | 本文方法 | 5层 | 只有v2.1筛选通过才允许运行 |
| `daniel_unet` | Daniel-UNet | 1层 | clean-room |
| `damiano_mcd_plus` | Damiano/DicePlus | 1层 | clean-room；MCD/TTA只是评价模式 |
| `nnunet_3d_default` | nnU-Net v2 3D fullres | 3D | 官方`nnunetv2==2.8.1` |
| `unetpp_5slice` | UNet++ | 5层 | clean-room匹配输入 |
| `tcsm_5slice` | TCSM | 5层 | clean-room匹配输入 |
| `cmmt_5slice` | CMMT-Net | 5层 | clean-room核心适配 |
| `frcnet_5slice` | FRCNet | 5层 | 代理实现，非官方复现 |
| `semi_urf_5slice` | Semi-URF | 5层 | 代理实现，非官方完整方法 |
| `randdg_5slice` | RandDG | 5层 | 输入级代理，非官方GUNet |

正式统计会同时输出`paired_method_comparisons.csv`和`paired_sci_comparisons.csv`。FRCNet作为MICCAI会议方法保留在前者，但不会被计入“超过SCI期刊算法”的数量。

## 阶段顺序

1. 等待v2.1的8次筛选完成并下载审计。
2. 若`formal_method_ready=false`，`07_prepare_v21_comparison.sh`会拒绝正式对比。
3. 若通过，`07` 会运行10种非 nnU-Net 方法的1 epoch/2 step GPU冒烟测试，然后锁定正式协议。
4. `08`、`09`和`10`分别运行主要方法、半监督/频域方法和官方 nnU-Net。
5. 只有11种方法各15次评价全部存在时，`11`才生成统一比较报告。
6. 失败后不得自动重试；审核日志后，使用`12`明确确认续跑。

## 命令

```bash
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL

bash scripts/autodl/07_prepare_v21_comparison.sh
bash scripts/autodl/08_run_v21_comparison_primary.sh
bash scripts/autodl/09_run_v21_comparison_ssl.sh
bash scripts/autodl/10_run_v21_comparison_nnunet.sh
bash scripts/autodl/11_finalize_v21_comparison.sh
bash scripts/autodl/13_collect_v21_comparison_results.sh
```

上述长时任务均应在tmux中分阶段执行，不要把`08`、`09`和`10`在同一GPU上并行。

审核后续跑示例：

```bash
CONFIRM_AUDITED_RESUME=YES \
  bash scripts/autodl/12_resume_v21_comparison_after_audit.sh cmmt_5slice
```

## 正式规模

- 11种方法；
- 5折；
- 3个初始化种子；
- 共165次内部训练/评价；
- 另有10次工程冒烟运行；
- 标签效率和外部验证不会被内部对比脚本静默启动。

正式内部比较结束后，`comparison_summary.json`只评估内部公平对比。在标签效率和CHAOS/AMOS外部验证完成之前，不得输出“通过Q2门槛”。
