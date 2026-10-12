# KidneyBand-ReliableSSL v2：AutoDL 从零运行指南

本流程的“从零”含义如下：

- 不上传或恢复本地实验 checkpoint；
- 不上传本地 `.venv`；
- 在 Linux 上从官方 ZIP 重新校验、解压、审计并生成清单和划分；
- AutoDL screening 使用新套件名和新输出目录，18个运行全部从 epoch 0 开始；
- 本地已有13/18项开发结果只保留为工程历史，不进入新 screening 的模块选择。

## 1. 租用实例

推荐先按量租用单张 RTX 4090 24GB，至少8核CPU、32GB内存和100GB数据盘。若计划同时保存AMOS、nnU-Net工作区及大量checkpoint，建议150–200GB数据盘。当前代码为单GPU串行调度，直接租双卡不会自动缩短时间。

系统建议使用 Ubuntu 22.04 和较新的 NVIDIA 驱动。项目会自行安装锁定的 Python 3.11、PyTorch 2.11.0 cu128和MONAI 1.6.0，不要在服务器上静默更换PyTorch版本。

## 2. 在本地生成上传包

在PowerShell中执行：

```powershell
cd "D:\30 医学论文\01\code\KidneyMRI-SpectralSSL"
powershell -ExecutionPolicy Bypass -File .\scripts\autodl\package_fresh.ps1
```

默认生成：

```text
D:\30 医学论文\01\autodl_upload\
├── KidneyMRI-SpectralSSL_code_fresh.zip
├── T2KidneyMRI_verified_downloads.zip
└── bundle_manifest.txt
```

数据包只包含三个经MD5核对的Zenodo原始下载文件，不包含本地解压结果、manifest、fold或实验输出。代码包排除`.git`、`.venv`、缓存和构建目录。

## 3. 上传并解压

把两个ZIP和`bundle_manifest.txt`上传到：

```text
/root/autodl-tmp/kidney/upload/
```

在AutoDL终端执行：

```bash
mkdir -p /root/autodl-tmp/kidney/{code,data,experiments,logs,upload}
cd /root/autodl-tmp/kidney/upload

unzip KidneyMRI-SpectralSSL_code_fresh.zip -d /root/autodl-tmp/kidney/code
unzip T2KidneyMRI_verified_downloads.zip -d /root/autodl-tmp/kidney/data
```

解压后应为：

```text
/root/autodl-tmp/kidney/
├── code/KidneyMRI-SpectralSSL/
├── data/T2KidneyMRI/downloads/
├── experiments/
├── logs/
└── upload/
```

## 4. 安装锁定环境

```bash
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL
bash scripts/autodl/00_bootstrap.sh
```

脚本会安装Python 3.11和Linux `.venv`，使用`uv.lock`冻结安装依赖，检查CUDA/BF16，并把环境记录写入`/root/autodl-tmp/kidney/logs/setup/`。任一步失败时停止，不要删除报错或改用另一个torch版本继续跑。

## 5. 从官方压缩包重新生成数据状态

```bash
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL
bash scripts/autodl/01_prepare_data_fresh.sh
```

脚本会校验三个下载文件MD5、安全解压、重新生成Linux绝对路径manifest、核对60名受试者和100次扫描、用`20260802`与`20260803`生成两套受试者级划分，并进行左右肾审计。

如果`raw`、`audit`或`splits`已经非空，脚本会拒绝运行，不会自动删除或覆盖旧状态。真正重跑时应使用新的`KIDNEY_WORKSPACE`。

## 6. GPU预检

```bash
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL
bash scripts/autodl/02_preflight_gpu.sh
```

必须同时通过全部pytest测试、CUDA、BF16 AMP、DZ-BiMorph的`repeat_update_applied=true`、单次训练和原空间评价，并确认smoke运行没有恢复旧checkpoint。通过标记为：

```text
/root/autodl-tmp/kidney/logs/preflight/preflight_passed.json
```

## 7. 全新运行screening

建议使用tmux：

```bash
tmux new -s kidney-screen
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL
bash scripts/autodl/03_run_screen_fresh.sh
```

按`Ctrl+B`后按`D`退出tmux，实验继续运行。重新进入：

```bash
tmux attach -t kidney-screen
```

另开终端监控：

```bash
nvidia-smi -l 1
tail -f /root/autodl-tmp/kidney/logs/screen_fresh/benchmark_console.log
```

screen输出目录固定为：

```text
/root/autodl-tmp/kidney/experiments/benchmarks/
  kidneyband_reliable_ssl_q2_v2_autodl_fresh_v1/screen/
```

启动脚本明确不传`--resume`，且只要screen目录已存在就拒绝启动，因此不会把旧运行混入新实验。

## 8. 中断后的处理

不要自动重试。先检查失败任务的`benchmark_status.json`、`benchmark_train.log`、`failure_diagnostics.json`和`metrics.jsonl`。

确认是服务器断连、实例回收等外部中断，而不是NaN、OOM、数据错误或DZ故障后，才手动执行：

```bash
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL
source .venv/bin/activate
kidney-seg benchmark \
  --suite configs/autodl/q2_v2_fresh.yaml \
  --phase screen \
  --resume
```

该命令只用于同一AutoDL fresh suite的人工审核后续跑，不能用来导入本地RTX 5060 checkpoint。

## 9. screen结束后的验收

要求九种配置各完成2折，共18个`evaluation_summary.json`。重点检查`screening_summary.csv`、`screening_decision.json`、`screen_runs.csv`和`resolved_configs/`。

随后生成不含大checkpoint的紧凑备份：

```bash
bash scripts/autodl/04_collect_screen_results.sh
```

下载`backups`中的`.tar.gz`及`.sha256`到本地。原checkpoint在确认单独备份前不要删除。

## 10. 不要立即启动正式五折实验

screen完成后先依据预注册门槛决定保留UABand、CSD和DZ模块。只有18/18运行完整、异常记录已检查、最终模块已确定后，才执行：

```bash
kidney-seg lock-protocol --suite configs/autodl/q2_v2_fresh.yaml
```

随后才能启动五折×三种子的internal矩阵。不能因为screen结果不理想而修改门槛、折或种子后继续沿用同一个suite名称；任何方法级修改都必须创建新的版本化suite并重新screen。
