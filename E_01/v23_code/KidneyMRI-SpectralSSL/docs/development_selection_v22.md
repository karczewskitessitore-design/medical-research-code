# v2.2开发筛选决策记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: ANALYZED — values transcribed from the user-provided AutoDL output; raw archive not independently rerun locally
- Version Label: v22_csd_development_selection

## 决策

v2.1五层输入筛选已完成4种方法、2个开发折、单种子，共8次运行。正式候选冻结为：

`KidneyBand-ReliableSSL v2.2 = five-slice B0+ + CSD-MT`

UABand-HT和UABand+CSD完整v2.1不进入正式主方法。它们保留为开发消融和负结果，不删除、不重新解释。

| 方法 | 运行数 | Dice | HD95 (mm) | repeat wCV (%) |
|---|---:|---:|---:|---:|
| V21_B0plus_5slice | 2 | 0.880337 | 8.242528 | 2.336449 |
| V21_UABand_5slice | 2 | 0.875448 | 7.358593 | 2.947166 |
| V21_CSD_5slice | 2 | 0.888556 | 6.513101 | 1.896445 |
| KidneyBand_ReliableSSL_v2_1 | 2 | 0.887067 | 6.390171 | 2.325067 |

CSD相对匹配B0+的Dice变化为`+0.008219`，HD95变化为`-1.729427 mm`，满足预注册的分割模块保留规则。UABand的Dice变化为`-0.004889`，未满足规则。完整v2.1虽满足相对参考模型的容差，但其必需模块UABand失败，因此`formal_method_ready=false`。

## 研究边界

- 这是开发阶段两折、单种子筛选，不是确认性统计证据。
- 不使用这8次运行宣称统计显著性、SCI算法优越性或期刊分区。
- 正式内部比较使用新划分种子`20260803`和初始化种子`20260811–20260813`。
- 正式结果不得静默恢复UABand或DZ-BiMorph；任何再次修改必须创建新版本和新开发套件。

