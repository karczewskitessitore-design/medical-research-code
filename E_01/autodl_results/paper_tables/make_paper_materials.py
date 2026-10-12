# -*- coding: utf-8 -*-
"""Paper materials: forest plot + three-line tables (CN/EN, MD/LaTeX/Word).
Data sources: q2_gate_summary.json (Dice stats, Holm family=10) +
paired_method_comparisons.csv (TKV error diff, Cohen's dz)."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

RESULTS = Path(r"D:\30 医学论文\01\autodl_results")
OUT = RESULTS / "paper_tables"
OUT.mkdir(exist_ok=True)

for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"):
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

gate = json.loads((RESULTS / "q2_gate_summary.json").read_text(encoding="utf-8"))
proposed = gate["proposed_method"]
proposed_dice = gate["proposed_internal_mean_dice"]
proposed_tkv = gate["proposed_tkv_relative_error"]
proposed_wcv = gate["proposed_repeat_wcv_percent"]
OURS_CN = "KidneyBand-ReliableSSL v2.2 (CSD)"
OURS_EN = "KidneyBand-ReliableSSL v2.2 (CSD)"

# ---------- internal rows (single source: q2_gate_summary.json, Holm family=10) ----------
internal = []
for row in gate["internal_comparisons"]:
    internal.append({
        "id": row["baseline"], "family": row["family"],
        "dice": row["baseline_mean_dice"], "diff": row["dice_difference"],
        "lo": row["dice_ci95_low"], "hi": row["dice_ci95_high"],
        "p_raw": row["p_raw"], "p_holm": row["p_holm"], "superior": row["superior"],
    })
internal.sort(key=lambda r: r["diff"])

# CSV extras: TKV error diff (CI) + Cohen's dz
csvx = {}
with open(RESULTS / "paired_method_comparisons.csv", encoding="utf-8") as fh:
    for rec in csv.DictReader(fh):
        csvx[rec["baseline"]] = rec

def tkv_of(mid):
    rec = csvx.get(mid)
    if not rec:
        return None
    return (float(rec["tkv_error_difference"]), float(rec["tkv_error_ci95_low"]),
            float(rec["tkv_error_ci95_high"]))

def dz_of(mid):
    rec = csvx.get(mid)
    if not rec or rec["dice_effect_dz"] == "":
        return None
    return float(rec["dice_effect_dz"])

# ---------- external rows ----------
external = []
for item in gate["external_evaluations"]:
    for comp in item["comparisons"]:
        external.append({
            "dataset": item["dataset"], "id": comp["baseline"],
            "diff": comp["difference"], "lo": comp["ci95_low"], "hi": comp["ci95_high"],
            "p_holm": comp["p_holm"], "noninferior": comp["noninferior"],
        })

SHORT_CN = {
    "V22_B0plus_5slice": "B0+ (匹配基线)", "nnunet_3d_default": "nnU-Net v2 3D",
    "daniel_unet": "Daniel-UNet", "damiano_mcd_plus": "Damiano-UNet",
    "unetpp_5slice": "UNet++", "tcsm_5slice": "TCSM", "cmmt_5slice": "CMMT-Net",
    "frcnet_5slice": "FRCNet", "semi_urf_5slice": "Semi-URF", "randdg_5slice": "RandDG",
}
SHORT_EN = {
    "V22_B0plus_5slice": "B0+ (matched baseline)", "nnunet_3d_default": "nnU-Net v2 3D",
    "daniel_unet": "Daniel-UNet", "damiano_mcd_plus": "Damiano-UNet",
    "unetpp_5slice": "UNet++", "tcsm_5slice": "TCSM", "cmmt_5slice": "CMMT-Net",
    "frcnet_5slice": "FRCNet", "semi_urf_5slice": "Semi-URF", "randdg_5slice": "RandDG",
}
DNAME_CN = {"chaos_t2_spir": "CHAOS T2-SPIR", "amos22_mri": "AMOS22 MRI"}
DNAME_EN = {"chaos_t2_spir": "CHAOS T2-SPIR", "amos22_mri": "AMOS22 MRI"}

def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def star(p):
    return " *" if p < 0.05 else ""

def v_int_cn(row):
    if row["diff"] < 0 and row["p_holm"] < 0.05:
        return "显著落后"
    if row["p_holm"] < 0.05:
        return "显著优于"
    return "无显著差异"

def v_int_en(row):
    if row["diff"] < 0 and row["p_holm"] < 0.05:
        return "Significantly worse"
    if row["p_holm"] < 0.05:
        return "Significantly better"
    return "No significant difference"

def v_ext_cn(c):
    if c["diff"] < 0 and c["p_holm"] < 0.05:
        return "显著落后"
    if c["p_holm"] < 0.05:
        return "显著优于"
    if c["noninferior"]:
        return "无差异(非劣效)"
    return "无差异(未达非劣效)"

def v_ext_en(c):
    if c["diff"] < 0 and c["p_holm"] < 0.05:
        return "Significantly worse"
    if c["p_holm"] < 0.05:
        return "Significantly better"
    if c["noninferior"]:
        return "Non-inferior"
    return "No difference (NI margin not met)"

# =========================================================== forest plots
colors = {"win": "#1e8449", "loss": "#b03a2e", "tie": "#566573"}

def draw_forest(path, en):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.2))
    fig.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.86, bottom=0.12)
    ax = axes[0]
    labels = [SHORT_EN.get(r["id"], r["id"]) if en else SHORT_CN.get(r["id"], r["id"]) for r in internal]
    for idx, row in enumerate(internal):
        color = colors["loss"] if row["diff"] < 0 else (colors["win"] if row["p_holm"] < 0.05 else colors["tie"])
        ax.errorbar(row["diff"], idx, xerr=[[row["diff"]-row["lo"]], [row["hi"]-row["diff"]]],
                    fmt="o", color=color, capsize=3, markersize=5.5, linewidth=1.4)
        ax.text(row["hi"]+0.0009, idx, f"{row['diff']:+.4f}{star(row['p_holm'])}",
                va="center", fontsize=8.5, color=color)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.axvline(0.01, color="gray", linewidth=0.8, linestyle="--")
    if en:
        ax.text(0.0103, len(internal)-0.35, "pre-registered minimal effect 0.01", fontsize=8, color="gray")
        ax.set_xlabel("Internal Dice difference (Ours $-$ Baseline)", fontsize=10)
        ax.set_title("A. Internal comparison (60 subjects, 5-fold $\\times$ 3 seeds)", fontsize=11, fontweight="bold")
    else:
        ax.text(0.0103, len(internal)-0.35, "预注册最小效应 0.01", fontsize=8, color="gray")
        ax.set_xlabel("内部 5 折 Dice 差(本文方法 − 基线)", fontsize=10)
        ax.set_title("A. 内部对比(60 受试者, 5 折×3 种子)", fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.024, 0.0245)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    names, values = [], []
    for item in gate["external_evaluations"]:
        dname = DNAME_EN.get(item["dataset"], item["dataset"]) if en else DNAME_CN.get(item["dataset"], item["dataset"])
        names.append(dname); values.append(None)
        for comp in item["comparisons"]:
            names.append("  " + (SHORT_EN.get(comp["baseline"], comp["baseline"]) if en else SHORT_CN.get(comp["baseline"], comp["baseline"])))
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
    if en:
        ax.text(-0.013, len(names)-0.4, "non-inferiority margin $-0.01$", fontsize=8, color="gray")
        ax.set_xlabel("External Dice difference (Ours $-$ Baseline)", fontsize=10)
        ax.set_title("B. External validation (source-only)", fontsize=11, fontweight="bold")
        fig.suptitle("Dice differences versus baselines (dots = mean, lines = 95% CI; * Holm-adjusted p < 0.05)",
                     fontsize=12, fontweight="bold")
        fig.text(0.06, 0.015,
                 "Green = significantly better, gray = no significant difference, red = significantly worse; "
                 "CIs from 10,000 subject-stratified bootstrap draws", fontsize=8.5)
    else:
        ax.text(-0.013, len(names)-0.4, "非劣效边距 −0.01", fontsize=8, color="gray")
        ax.set_xlabel("外部 Dice 差(本文方法 − 基线)", fontsize=10)
        ax.set_title("B. 外部验证(source-only)", fontsize=11, fontweight="bold")
        fig.suptitle("KidneyBand-ReliableSSL v2.2 (CSD) 相对基线的 Dice 差异(点=均值, 线=95% CI; * Holm 校正 p<0.05)",
                     fontsize=12, fontweight="bold")
        fig.text(0.06, 0.015,
                 "绿=显著优于基线, 灰=无显著差异, 红=显著劣于基线; 置信区间来自 10,000 次受试者分层 bootstrap",
                 fontsize=8.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.36, 0.24)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)

draw_forest(OUT / "森林图_内部与外部Dice差异.png", en=False)
draw_forest(OUT / "forest_plot_internal_external_EN.png", en=True)

# =========================================================== Markdown tables
def fmt_ci(lo, hi):
    return f"[{lo:+.4f}, {hi:+.4f}]"

def fmt_tkv(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]" if t else "—"

def fmt_dz(dz):
    return f"{dz:+.3f}" if dz is not None else "—"

# --- Chinese Table 1 ---
lines = ["# 表 1  内部正式对比(25% 受试者标注; 受试者级统计)", ""]
lines.append("| 方法 | 家族 | 平均 Dice | ΔDice vs 本文 | 95% CI | Cohen's dz | TKV 相对误差差 | Holm p | 结论 |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for row in internal:
    t = tkv_of(row["id"])
    lines.append(f"| {SHORT_CN.get(row['id'], row['id'])} | {row['family']} | {row['dice']:.4f} | "
                 f"{row['diff']:+.4f} | {fmt_ci(row['lo'], row['hi'])} | {fmt_dz(dz_of(row['id']))} | "
                 f"{fmt_tkv(t)} | {fmt_p(row['p_holm'])} | {v_int_cn(row)} |")
lines += [
    "",
    f"注: 本文方法 = {OURS_CN}; 内部平均 Dice = {proposed_dice:.4f}, TKV 相对误差 = {proposed_tkv:.4f}, "
    f"重复扫描 wCV = {proposed_wcv:.2f}%。",
    "统计: Dice 的 95% CI 来自 10,000 次受试者分层 bootstrap, p 值来自 100,000 次配对置换检验; "
    "Holm 校正家族 = 10(全部内部比较, 含匹配基线 B0+), 与 Q2 门槛报告一致。"
    "TKV 相对误差差与 Cohen's dz 来自配对比较流水线。",
    "ΔDice > 0 表示本文方法更优; TKV 差 > 0 表示本文方法误差更高。",
    "*, 论文表中可按期刊要求以粗体标出 Holm p < 0.05 的行。",
]
(OUT / "表1_内部对比.md").write_text("\n".join(lines), encoding="utf-8")

# --- Chinese Table 2 ---
lines = ["# 表 2  外部验证(source-only, 不微调; 5 折集成 × 3 种子)", ""]
lines.append("| 外部数据集 | 对比基线 | ΔDice vs 本文 | 95% CI | Holm p | 结论 |")
lines.append("|---|---|---|---|---|---|")
for c in external:
    lines.append(f"| {DNAME_CN.get(c['dataset'], c['dataset'])} | {SHORT_CN.get(c['id'], c['id'])} | "
                 f"{c['diff']:+.4f} | {fmt_ci(c['lo'], c['hi'])} | {fmt_p(c['p_holm'])} | {v_ext_cn(c)} |")
lines += [
    "",
    f"注: 本文方法在 CHAOS T2-SPIR 平均 Dice = {gate['external_evaluations'][0]['proposed_mean_dice']:.4f}, "
    f"在 AMOS22 MRI 平均 Dice = {gate['external_evaluations'][1]['proposed_mean_dice']:.4f}。",
    "统计: 各数据集分别报告; 与内部同一套 bootstrap/置换/Holm 流程; 非劣效边距 ΔDice ≥ −0.01。",
]
(OUT / "表2_外部验证.md").write_text("\n".join(lines), encoding="utf-8")

# --- English Table 1 ---
lines = ["# Table 1. Internal formal comparison (25% subject labels; subject-level statistics)", ""]
lines.append("| Method | Family | Mean Dice | $\\Delta$Dice vs Ours | 95% CI | Cohen's $d_z$ | TKV rel. err. diff. | Holm $p$ | Conclusion |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for row in internal:
    t = tkv_of(row["id"])
    lines.append(f"| {SHORT_EN.get(row['id'], row['id'])} | {row['family']} | {row['dice']:.4f} | "
                 f"{row['diff']:+.4f} | {fmt_ci(row['lo'], row['hi'])} | {fmt_dz(dz_of(row['id']))} | "
                 f"{fmt_tkv(t)} | {fmt_p(row['p_holm'])} | {v_int_en(row)} |")
lines += [
    "",
    f"Notes: Ours = {OURS_EN}; internal mean Dice = {proposed_dice:.4f}, TKV relative error = {proposed_tkv:.4f}, "
    f"repeat-scan wCV = {proposed_wcv:.2f}%.",
    "Statistics: Dice 95% CIs from 10,000 subject-stratified bootstrap draws, p-values from 100,000 paired "
    "permutation tests; Holm adjustment across the family of 10 internal comparisons (including the matched "
    "baseline B0+), consistent with the Q2 gate report. TKV relative-error differences and Cohen's $d_z$ are "
    "from the paired-comparison pipeline.",
    "$\\Delta$Dice > 0 favors ours; TKV difference > 0 indicates higher error of ours.",
    "*, mark rows with Holm $p$ < 0.05 in bold as required by the target journal.",
]
(OUT / "Table1_Internal_EN.md").write_text("\n".join(lines), encoding="utf-8")

# --- English Table 2 ---
lines = ["# Table 2. External validation (source-only, no fine-tuning; 5-fold ensemble $\\times$ 3 seeds)", ""]
lines.append("| External dataset | Baseline | $\\Delta$Dice vs Ours | 95% CI | Holm $p$ | Conclusion |")
lines.append("|---|---|---|---|---|---|")
for c in external:
    lines.append(f"| {DNAME_EN.get(c['dataset'], c['dataset'])} | {SHORT_EN.get(c['id'], c['id'])} | "
                 f"{c['diff']:+.4f} | {fmt_ci(c['lo'], c['hi'])} | {fmt_p(c['p_holm'])} | {v_ext_en(c)} |")
lines += [
    "",
    f"Notes: Ours achieves mean Dice = {gate['external_evaluations'][0]['proposed_mean_dice']:.4f} on "
    f"CHAOS T2-SPIR and {gate['external_evaluations'][1]['proposed_mean_dice']:.4f} on AMOS22 MRI.",
    "Statistics: reported per dataset with the same bootstrap/permutation/Holm pipeline as the internal "
    "analysis; non-inferiority margin $\\Delta$Dice $\\geq -0.01$.",
]
(OUT / "Table2_External_EN.md").write_text("\n".join(lines), encoding="utf-8")

# =========================================================== LaTeX (booktabs)
def tex_num(v, digits=4):
    return f"${v:+.{digits}f}$"

def tex_ci(lo, hi):
    return f"[{tex_num(lo)}, {tex_num(hi)}]"

def tex_p(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"

def esc(s):
    return s.replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("_", "\\_")

tex1 = ["% Table 1 - Internal comparison (requires booktabs; compile with a Unicode/LaTeX engine or map glyphs)",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Internal comparison with 25\\% subject labels (subject-level statistics). Ours = "
        f"KidneyBand-ReliableSSL v2.2 (CSD); mean Dice = {proposed_dice:.4f}, TKV relative error = {proposed_tkv:.4f}, "
        f"repeat-scan wCV = {proposed_wcv:.2f}\\%. $\\Delta$Dice $>$ 0 favors ours; positive TKV error difference "
        "indicates higher error of ours. Dice 95\\% CIs from 10{,}000 subject-stratified bootstrap draws, "
        "p-values from 100{,}000 paired permutation tests; Holm adjustment across all 10 internal comparisons.}",
        "\\label{tab:internal_comparison}",
        "\\begin{tabular}{lcccccccc}", "\\toprule",
        "Method & Family & Mean Dice & $\\Delta$Dice & 95\\% CI & Cohen's $d_z$ & TKV err. diff. & Holm $p$ & Conclusion \\\\",
        "\\midrule"]
for row in internal:
    t = tkv_of(row["id"])
    tkv_cell = tex_num(t[0]) + " " + tex_ci(t[1], t[2]) if t else "---"
    dz_cell = f"${dz_of(row['id']):+.3f}$" if dz_of(row['id']) is not None else "---"
    tex1.append(f"{esc(SHORT_EN.get(row['id'], row['id']))} & {esc(row['family'])} & {row['dice']:.4f} & "
                f"{tex_num(row['diff'])} & {tex_ci(row['lo'], row['hi'])} & {dz_cell} & {tkv_cell} & "
                f"{tex_p(row['p_holm'])} & {v_int_en(row)} \\\\")
tex1 += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
(OUT / "Table1_Internal_EN.tex").write_text("\n".join(tex1), encoding="utf-8")

tex2 = ["% Table 2 - External validation (requires booktabs)",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{External validation (source-only, no fine-tuning; 5-fold ensemble $\\times$ 3 seeds). "
        f"Ours achieves mean Dice = {gate['external_evaluations'][0]['proposed_mean_dice']:.4f} on CHAOS T2-SPIR "
        f"and {gate['external_evaluations'][1]['proposed_mean_dice']:.4f} on AMOS22 MRI. "
        "Statistics reported per dataset with the same bootstrap/permutation/Holm pipeline as the internal "
        "analysis; non-inferiority margin $\\Delta$Dice $\\geq -0.01$.}",
        "\\label{tab:external_validation}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "External dataset & Baseline & $\\Delta$Dice & 95\\% CI & Holm $p$ & Conclusion \\\\",
        "\\midrule"]
for c in external:
    tex2.append(f"{esc(DNAME_EN.get(c['dataset'], c['dataset']))} & {esc(SHORT_EN.get(c['id'], c['id']))} & "
                f"{tex_num(c['diff'])} & {tex_ci(c['lo'], c['hi'])} & {tex_p(c['p_holm'])} & {v_ext_en(c)} \\\\")
tex2 += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
(OUT / "Table2_External_EN.tex").write_text("\n".join(tex2), encoding="utf-8")

# =========================================================== Word (python-docx, three-line tables)
def cell_border(cell, specs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcB = tcPr.find(qn("w:tcBorders"))
    if tcB is None:
        tcB = OxmlElement("w:tcBorders")
        tcPr.append(tcB)
    for edge in ("top", "bottom", "left", "right"):
        el = tcB.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            tcB.append(el)
        if edge in specs:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), specs[edge]); el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")

def make_docx(path, caption, headers, rows, notes, cn=False):
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = Cm(2.0); sec.right_margin = Cm(2.0)
    p = doc.add_paragraph()
    run = p.add_run(caption)
    run.bold = True
    run.font.size = Pt(10.5)
    run.font.name = "Times New Roman"
    if cn:
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    nrows, ncols = len(rows) + 1, len(headers)
    for i in range(nrows):
        for j in range(ncols):
            cell = table.cell(i, j)
            para = cell.paragraphs[0]
            run = para.add_run((headers if i == 0 else rows[i - 1])[j])
            run.font.size = Pt(9)
            run.font.name = "Times New Roman"
            if cn:
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            if i == 0:
                run.bold = True
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            specs = {}
            if i == 0:
                specs["top"] = "12"; specs["bottom"] = "6"
            if i == nrows - 1:
                specs["bottom"] = "12"
            cell_border(cell, specs)
    for note in notes:
        p = doc.add_paragraph()
        run = p.add_run(note)
        run.italic = True
        run.font.size = Pt(8)
        run.font.name = "Times New Roman"
        if cn:
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.save(path)

# Chinese Word
rows1 = []
for row in internal:
    t = tkv_of(row["id"])
    rows1.append([SHORT_CN.get(row["id"], row["id"]), row["family"], f"{row['dice']:.4f}",
                  f"{row['diff']:+.4f}", fmt_ci(row["lo"], row["hi"]), fmt_dz(dz_of(row["id"])),
                  fmt_tkv(t), fmt_p(row["p_holm"]), v_int_cn(row)])
make_docx(OUT / "表1_内部对比.docx",
          "表 1  内部正式对比(25% 受试者标注; 受试者级统计)",
          ["方法", "家族", "平均 Dice", "ΔDice vs 本文", "95% CI", "Cohen's dz", "TKV 相对误差差", "Holm p", "结论"],
          rows1,
          [f"注: 本文方法 = {OURS_CN}; 内部平均 Dice = {proposed_dice:.4f}, TKV 相对误差 = {proposed_tkv:.4f}, "
           f"重复扫描 wCV = {proposed_wcv:.2f}%。",
           "统计: Dice 的 95% CI 来自 10,000 次受试者分层 bootstrap, p 值来自 100,000 次配对置换检验; "
           "Holm 校正家族 = 10(全部内部比较, 含匹配基线 B0+)。TKV 相对误差差与 Cohen's dz 来自配对比较流水线。",
           "ΔDice > 0 表示本文方法更优; TKV 差 > 0 表示本文方法误差更高。"],
          cn=True)

rows2 = []
for c in external:
    rows2.append([DNAME_CN.get(c["dataset"], c["dataset"]), SHORT_CN.get(c["id"], c["id"]),
                  f"{c['diff']:+.4f}", fmt_ci(c["lo"], c["hi"]), fmt_p(c["p_holm"]), v_ext_cn(c)])
make_docx(OUT / "表2_外部验证.docx",
          "表 2  外部验证(source-only, 不微调; 5 折集成 × 3 种子)",
          ["外部数据集", "对比基线", "ΔDice vs 本文", "95% CI", "Holm p", "结论"],
          rows2,
          [f"注: 本文方法在 CHAOS T2-SPIR 平均 Dice = {gate['external_evaluations'][0]['proposed_mean_dice']:.4f}, "
           f"在 AMOS22 MRI 平均 Dice = {gate['external_evaluations'][1]['proposed_mean_dice']:.4f}。",
           "统计: 各数据集分别报告; 与内部同一套 bootstrap/置换/Holm 流程; 非劣效边距 ΔDice ≥ −0.01。"],
          cn=True)

# English Word
rows1e = []
for row in internal:
    t = tkv_of(row["id"])
    rows1e.append([SHORT_EN.get(row["id"], row["id"]), row["family"], f"{row['dice']:.4f}",
                   f"{row['diff']:+.4f}", fmt_ci(row["lo"], row["hi"]), fmt_dz(dz_of(row["id"])),
                   fmt_tkv(t), fmt_p(row["p_holm"]), v_int_en(row)])
make_docx(OUT / "Table1_Internal_EN.docx",
          "Table 1. Internal formal comparison (25% subject labels; subject-level statistics)",
          ["Method", "Family", "Mean Dice", "$\\Delta$Dice vs Ours", "95% CI", "Cohen's $d_z$",
           "TKV rel. err. diff.", "Holm $p$", "Conclusion"],
          rows1e,
          [f"Notes: Ours = {OURS_EN}; internal mean Dice = {proposed_dice:.4f}, TKV relative error = "
           f"{proposed_tkv:.4f}, repeat-scan wCV = {proposed_wcv:.2f}%.",
           "Statistics: Dice 95% CIs from 10,000 subject-stratified bootstrap draws, p-values from 100,000 "
           "paired permutation tests; Holm adjustment across the family of 10 internal comparisons "
           "(including the matched baseline B0+). TKV relative-error differences and Cohen's dz are from "
           "the paired-comparison pipeline.",
           "$\\Delta$Dice > 0 favors ours; TKV difference > 0 indicates higher error of ours."])

rows2e = []
for c in external:
    rows2e.append([DNAME_EN.get(c["dataset"], c["dataset"]), SHORT_EN.get(c["id"], c["id"]),
                   f"{c['diff']:+.4f}", fmt_ci(c["lo"], c["hi"]), fmt_p(c["p_holm"]), v_ext_en(c)])
make_docx(OUT / "Table2_External_EN.docx",
          "Table 2. External validation (source-only, no fine-tuning; 5-fold ensemble x 3 seeds)",
          ["External dataset", "Baseline", "$\\Delta$Dice vs Ours", "95% CI", "Holm $p$", "Conclusion"],
          rows2e,
          [f"Notes: Ours achieves mean Dice = {gate['external_evaluations'][0]['proposed_mean_dice']:.4f} on "
           f"CHAOS T2-SPIR and {gate['external_evaluations'][1]['proposed_mean_dice']:.4f} on AMOS22 MRI.",
           "Statistics: reported per dataset with the same bootstrap/permutation/Holm pipeline as the "
           "internal analysis; non-inferiority margin $\\Delta$Dice >= -0.01."])

print("OK ->", OUT)