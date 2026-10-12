# KidneyBand-ReliableSSL v2.1：AutoDL 五层输入定向筛选

## 目的

v2 开发筛选选择了5层 2.5D 输入，但 UABand 和 CSD 的模块实验使用的是3层输入。v2.1 只修正这一配置不匹配，不读取正式划分，不改动数据、优化器、训练轮数、折或随机种子。

DZ-BiMorph 在 v2 中只将 wCV 改善了 `0.0795` 个百分点，未达到预注册的 `0.5` 个百分点门槛，因此不进入 v2.1 正式候选。

## 运行矩阵

| 方法 | 输入 | UABand | CSD | DZ |
|---|---:|---:|---:|---:|
| `V21_B0plus_5slice` | 5层 | 否 | 否 | 否 |
| `V21_UABand_5slice` | 5层 | 是 | 否 | 否 |
| `V21_CSD_5slice` | 5层 | 否 | 是 | 否 |
| `KidneyBand_ReliableSSL_v2_1` | 5层 | 是 | 是 | 否 |

每种方法使用开发划分 `20260802`、fold 0/1、初始化种子 `20260802`、25%受试者标注和150 epochs，共8次新训练。

## 选择规则

- UABand 和 CSD 分别相对匹配的5层基线要求 Dice 提高至少 `0.005`，且 HD95 恶化不超过 `1 mm`。
- 组合模型的 Dice 不得比最佳单模块低超过 `0.003`。
- 只有两个单模块和组合模型都通过时，`formal_method_ready` 才为 `true`。
- 这仍是开发筛选，不能作为论文确证性结果。

## AutoDL 命令

将 v2.1 增量包覆盖解压到项目根目录后，先检查不存在旧的 v2.1 输出，然后执行：

```bash
cd /root/autodl-tmp/kidney/code/KidneyMRI-SpectralSSL
tmux new -s kidney-v21
bash scripts/autodl/05_run_v21_5slice_screen.sh
```

日志：

```text
/root/autodl-tmp/kidney/logs/screen_v21_5slice/benchmark_console.log
```

结果：

```text
/root/autodl-tmp/kidney/experiments/benchmarks/
  kidneyband_reliable_ssl_q2_v2_1_5slice_screen_autodl_fresh_v1/screen/
```

完成后备份：

```bash
bash scripts/autodl/06_collect_v21_5slice_results.sh
```

在 `screening_decision.json` 经人工复核前，不得启动正式五折实验或执行 `lock-protocol`。
