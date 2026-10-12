# -*- coding: utf-8 -*-
"""Generate paper-ready forest plot + three-line tables from Q2 gate summary."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RESULTS = Path(r"D:\30 医学论文\01\autodl_results")
OUT = RESULTS / "paper_tables"
OUT.mkdir(exist_ok=True)

for font_path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"):
    try:
        font_manager.fontManager.addfont(font_path)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

gate = json.loads((RESULTS / "q2_gate_summary.json").read_text(encoding="utf-8"))
proposed = gate["proposed_method"]
proposed_dice = gate["proposed_internal_mean_dice"]

internal = []
for row in gate["internal_comparisons"]:
    internal.append({
        "id": row["baseline"], "family": row["family"],
        "dice": row["baseline_mean_dice"], "diff": row["dice_difference"],
        "lo": row["dice_ci95_low"], "hi": row["dice_ci95_high"],
        "p_holm": row["p_holm"],
    })
internal.sort(key=lambda r: r["diff"])

tkv = {}
with open(RESULTS / "paired_method_comparisons.csv", encoding="utf-8") as handle:
    for record in csv.DictReader(handle):
        tkv[record["baseline"]] = (
            float(record["tkv_error_difference"]),
            float(record["tkv_error_ci95_low"]),
            float(record["tkv_error_ci95_high"]),
        )

external = []
for item in gate["external_evaluations"]:
    for comp in item["comparisons"]:
        external.append({
            "dataset": item["dataset"], "id": comp["baseline"],
            "diff": comp["difference"], "lo": comp["ci95_low"], "hi": comp["ci95_high"],
            "p_holm": comp["p_holm"], "noninferior": comp["noninferior"],
        })

SHORT = {
    "V22_B0plus_5slice": "B0+ (匹配基线)",
    "nnunet_3d_default": "nnU-Net v2 3D",
    "daniel_unet": "Daniel-UNet",
    "damiano_mcd_plus": "Damiano-UNet",
    "unetpp_5slice": "UNet++",
    "tcsm_5slice": "TCSM",
    "cmmt_5slice": "CMMT-Net",
    "frcnet_5slice": "FRCNet",
    "semi_urf_5slice": "Semi-URF",
    "randdg_5slice": "RandDG",
}

def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def star(p):
    return " *" if p < 0.05 else ""

fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.2))
fig.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.86, bottom=0.10)
colors = {"win": "#1e8449", "loss": "#b03a2e", "tie": "#566573"}

ax = axes[0]
labels = [SHORT.get(r["id"], r["id"]) for r in internal]
for idx, row in enumerate(internal):
    color = colors["loss"] if row["diff"] < 0 else (colors["win"] if row["p_holm"] < 0.05 else colors["tie"])
    ax.errorbar(row["diff"], idx, xerr=[[row["diff"]-row["lo"]], [row["hi"]-row["diff"]]],
                fmt="o", color=color, capsize=3, markersize=5.5, linewidth=1.4)
    ax.text(row["hi"]+0.0009, idx, f"{row['diff']:+.4f}{star(row['p_holm'])}",
            va="center", fontsize=8.5, color=color)
ax.axvline(0, color="black", linewidth=0.9)
ax.axvline(0.01, color="gray", linewidth=0.8, linestyle="--")
ax.text(0.0103, len(internal)-0.35, "预注册最小效应 0.01", fontsize=8, color="gray")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-0.024, 0.0245)
ax.set_xlabel("内部 5 折 Dice 差(本文方法 − 基线)", fontsize=10)
ax.set_title("A. 内部对比(60 受试者, 5 折×3 种子)", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.25)

ax = axes[1]
names, values = [], []
for item in gate["external_evaluations"]:
    dname = "CHAOS T2-SPIR" if item["dataset"] == "chaos_t2_spir" else "AMOS22 MRI"
    names.append(dname); values.append(None)
    for comp in item["comparisons"]:
        names.append("  " + SHORT.get(comp["baseline"], comp["baseline"]))
        values.append(comp)
for idx, value in enumerate(values):
    if value is None:
        ax.text(-0.34, idx, names[idx], fontsize=10, fontweight="bold", va="center")
        continue
    color = colors["loss"] if value["difference"] < 0 else (colors["win"] if value["p_holm"] < 0.05 else colors["tie"])
    ax.errorbar(value["difference"], idx, xerr=[[value["difference"]-value["ci95_low"]], [value["ci95_high"]-value["difference"]]],
                fmt="o", color=color, capsize=3, markersize=5.5, linewidth=1.4)
    ax.text(value["ci95_high"]+0.008, idx, f"{value['difference']:+.4f}{star(value['p_holm'])}",
            va="center", fontsize=8.5, color=color)
ax.axvline(0, color="black", linewidth=0.9)
ax.axvline(-0.01, color="gray", linewidth=0.8, linestyle="--")
ax.text(-0.013, len(names)-0.4, "非劣效边距 −0.01", fontsize=8, color="gray")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-0.36, 0.24)
ax.set_xlabel("外部 Dice 差(本文方法 − 基线)", fontsize=10)
ax.set_title("B. 外部验证(source-only)", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.25)

fig.suptitle("KidneyBand-ReliableSSL v2.2 (CSD) 相对基线的 Dice 差异(点=均值, 线=95% CI; * Holm 校正 p<0.05)",
             fontsize=12, fontweight="bold")
fig.text(0.06, 0.015,
         "绿=显著优于基线, 灰=无显著差异, 红=显著劣于基线; 置信区间来自 10,000 次受试者分层 bootstrap",
         fontsize=8.5)
fig.savefig(OUT / "森林图_内部与外部Dice差异.png", dpi=200, facecolor="white")
plt.close(fig)

lines = ["# 表 1  内部正式对比(25% 受试者标注; 受试者级统计)", ""]
lines.append("| 方法 | 家族 | 平均 Dice | ΔDice vs 本文 | 95% CI | Holm p | TKV 相对误差差 | 结论 |")
lines.append("|---|---|---|---|---|---|---|---|")
matched = next(r for r in internal if r["id"] == "V22_B0plus_5slice")
for row in internal:
    if row["id"] == "V22_B0plus_5slice":
        continue
    t = tkv.get(row["id"])
    tkv_text = f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]" if t else "—"
    verdict = "显著落后" if (row["diff"] < 0 and row["p_holm"] < 0.05) else ("显著优于" if row["p_holm"] < 0.05 else "无显著差异")
    lines.append(f"| {SHORT.get(row['id'], row['id'])} | {row['family']} | {row['dice']:.4f} | "
                 f"{row['diff']:+.4f} | [{row['lo']:+.4f}, {row['hi']:+.4f}] | {fmt_p(row['p_holm'])} | "
                 f"{tkv_text} | {verdict} |")
lines.append(f"| B0+ (匹配基线) | Matched five-slice 2.5D U-Net | {matched['dice']:.4f} | {matched['diff']:+.4f} | "
             f"[{matched['lo']:+.4f}, {matched['hi']:+.4f}] | {fmt_p(matched['p_holm'])} | — | 显著优于 |")
lines += [
    "",
    f"注: 本文方法 = KidneyBand-ReliableSSL v2.2 (CSD-only); 内部平均 Dice = {proposed_dice:.4f}, "
    f"TKV 相对误差 = {gate['proposed_tkv_relative_error']:.4f}, 重复扫描 wCV = {gate['proposed_repeat_wcv_percent']:.2f}%。",
    "统计: 受试者内合并重复扫描后按受试者聚类, 10,000 次分层 bootstrap 计算 95% CI, 100,000 次配对置换检验, "
    "六基线 Holm 校正。ΔDice>0 表示本文方法更优; TKV 差>0 表示本文方法误差更高。",
    "*, 论文表中可按期刊要求以粗体标出 Holm p<0.05 的行。",
]
(OUT / "表1_内部对比.md").write_text("\n".join(lines), encoding="utf-8")

lines = ["# 表 2  外部验证(source-only, 不微调; 5 折集成 × 3 种子)", ""]
lines.append("| 外部数据集 | 对比基线 | ΔDice vs 本文 | 95% CI | Holm p | 结论 |")
lines.append("|---|---|---|---|---|---|")
dataset_names = {"chaos_t2_spir": "CHAOS T2-SPIR", "amos22_mri": "AMOS22 MRI"}
for item in gate["external_evaluations"]:
    dname = dataset_names.get(item["dataset"], item["dataset"])
    for comp in item["comparisons"]:
        if comp["difference"] < 0 and comp["p_holm"] < 0.05:
            verdict = "显著落后"
        elif comp["p_holm"] < 0.05:
            verdict = "显著优于"
        elif comp["noninferior"]:
            verdict = "无差异(非劣效)"
        else:
            verdict = "无差异(未达非劣效边距)"
        lines.append(f"| {dname} | {SHORT.get(comp['baseline'], comp['baseline'])} | {comp['difference']:+.4f} | "
                     f"[{comp['ci95_low']:+.4f}, {comp['ci95_high']:+.4f}] | {fmt_p(comp['p_holm'])} | {verdict} |")
lines += [
    "",
    f"注: 本文方法在 CHAOS T2-SPIR 平均 Dice = {gate['external_evaluations'][0]['proposed_mean_dice']:.4f}, "
    f"在 AMOS22 MRI 平均 Dice = {gate['external_evaluations'][1]['proposed_mean_dice']:.4f}。",
    "统计: 各数据集分别报告; 与内部同一套 bootstrap/置换/Holm 流程; 非劣效边距 ΔDice ≥ −0.01。",
]
(OUT / "表2_外部验证.md").write_text("\n".join(lines), encoding="utf-8")

print("OK ->", OUT)