# v2.2 CSD-only正式对比AutoDL指南

## 前提

- v2.1的8次五层筛选已经完成并备份；
- `CSD.keep=true`、`UABand.keep=false`、`formal_method_ready=false`；
- v2.2只选择CSD，不改变开发阶段已观察到的超参数；
- 不再运行旧的`07_prepare_v21_comparison.sh`。

## 运行顺序

在工程目录中执行：

```bash
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL

bash scripts/autodl/14_prepare_v22_csd_comparison.sh
bash scripts/autodl/15_run_v22_csd_comparison.sh primary
bash scripts/autodl/15_run_v22_csd_comparison.sh ssl
bash scripts/autodl/15_run_v22_csd_comparison.sh nnunet
bash scripts/autodl/16_finalize_v22_csd_comparison.sh
bash scripts/autodl/18_collect_v22_csd_results.sh
```

`14`只执行10种非nnU-Net方法的1 epoch/2 step冒烟测试并锁定协议，不启动正式训练。`15`的三个分片必须在单GPU上依次执行，不能并行。

## 中断恢复

先检查失败方法日志。确认属于实例关机等外部中断后执行：

```bash
CONFIRM_AUDITED_RESUME=YES \
  bash scripts/autodl/17_resume_v22_csd_comparison.sh \
  KidneyBand_ReliableSSL_v2_2_CSD
```

可同时给出多个方法ID。已存在`evaluation_summary.json`的运行会跳过；其他运行从`last_checkpoint.pt`恢复。

## 规模和输出

- 11种方法，5折，3种子，共165次内部训练/评价；
- 主输出目录：`/root/autodl-tmp/kidney/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_2_csd_comparison_autodl_v1/internal`；
- `paired_method_comparisons.csv`包含所有方法学对比；
- `paired_sci_comparisons.csv`只包含SCI期刊算法；
- FRCNet是MICCAI方法，不计入超过SCI算法的数量；
- 外部验证和标签效率曲线不会被内部脚本自动启动。
