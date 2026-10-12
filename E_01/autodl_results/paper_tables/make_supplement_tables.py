# -*- coding: utf-8 -*-
"""Generate supplement tables 3/4/5 (CN/EN; MD/LaTeX/Word), consistent with Tables 1/2.
Sources: amos_label_summary_full.csv (AMOS 10/20%), external_supplement_summary.csv
(FRCNet/CMMT external), q2_gate_summary.json (CSD external + 25% internal),
and server label_efficiency summaries (12.5/50/100%, hard-coded with source comment).
Optional: autodl_results/tkv_robust_stats.json -> adds median/IQR footnote to Table 4."""
import csv
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

RESULTS = Path(r"D:\30 医学论文\01\autodl_results")
OUT = RESULTS / "paper_tables"
OUT.mkdir(exist_ok=True)

OURS_CN = "KidneyBand-ReliableSSL v2.2 (CSD)"
OURS_EN = "KidneyBand-ReliableSSL v2.2 (CSD)"

gate = json.loads((RESULTS / "q2_gate_summary.json").read_text(encoding="utf-8"))

# ---------- Table 3: internal label efficiency ----------
# 12.5/50/100%: 1 seed x 5 folds from server label_efficiency summaries;
# 25%: formal 15 runs (q2_gate_summary.json).
b0plus_25 = next(r["baseline_mean_dice"] for r in gate["internal_comparisons"]
                 if r["baseline"] == "V22_B0plus_5slice")
T3 = [
    (0.125, 0.8958, 0.8802, "1 seed×5 折", "1 seed x 5 folds"),
    (0.250, gate["proposed_internal_mean_dice"], b0plus_25, "正式 15 组", "formal, 15 runs"),
    (0.500, 0.9138, 0.9093, "1 seed×5 折", "1 seed x 5 folds"),
    (1.000, 0.9216, 0.9187, "全标注 Sanity", "full-label sanity"),
]

# ---------- Table 4: AMOS22 public benchmark ----------
T4 = []
with open(RESULTS / "amos_label_summary_full.csv", encoding="utf-8") as fh:
    for rec in csv.DictReader(fh):
        T4.append((float(rec["labeled_fraction"]), rec["method"],
                   float(rec["mean_dice"]), float(rec["mean_hd95_mm"]),
                   float(rec["mean_tkv_err"])))
T4.sort(key=lambda r: (r[0], r[1]))
M4 = {"KidneyBand_ReliableSSL_v2_2_CSD": "KidneyBand-ReliableSSL v2.2 (CSD)",
      "V22_B0plus_5slice": "V22 B0+ (匹配基线)", "frcnet_5slice": "FRCNet 复现*"}
M4_EN = {"KidneyBand_ReliableSSL_v2_2_CSD": "KidneyBand-ReliableSSL v2.2 (CSD)",
         "V22_B0plus_5slice": "V22 B0+ (matched baseline)", "frcnet_5slice": "FRCNet proxy*"}

# ---------- Table 5: external SSL baselines ----------
T5 = []
ext_supp = {}
with open(RESULTS / "external_supplement_summary.csv", encoding="utf-8") as fh:
    for rec in csv.DictReader(fh):
        ext_supp[(rec["dataset"], rec["method"])] = float(rec["dice"])
for item in gate["external_evaluations"]:
    ds = item["dataset"]
    T5.append((ds, item["proposed_mean_dice"],
               ext_supp[(ds, "frcnet_5slice")], ext_supp[(ds, "cmmt_5slice")]))
D5 = {"chaos_t2_spir": "CHAOS T2-SPIR", "amos22_mri": "AMOS22 MRI"}

# ---------- optional robust TKV stats (server output) ----------
robust = {}
rpath = RESULTS / "tkv_robust_stats.json"
if rpath.exists():
    robust = json.loads(rpath.read_text(encoding="utf-8"))

def tkv_footnote_cn():
    lines = [
        "注：*FRCNet 为按 MICCAI 2024 论文在本协议下复现的输出级 FDC/MRSC 代理（clean-room proxy）；指标为逐 subject 均值（n=40）。",
        "10% 标签下 CSD 的 TKV 均值（2.054）由 2/40 例（5%）CSD 特有的超分割失败主导（AMOS_0508: 39.3、AMOS_0514: 33.3，"
        "均 Dice≈0.03、HD95>175 mm），排除后均值 0.25；8 例 Dice<0.05 的困难病例为三种方法共有（B0+ 与 FRCNet 同样各 8 例），属 10% 标签协议的困难样本而非方法缺陷。"
        "20% 标签后无超分割失败，中位数 0.046。",
        "文献对照（BoCLIS TMI 2025、VerSemi TMI 2025、FRCNet 原文等）均在 ACDC/LA 等基准上报告 10%/20% 标签结果，"
        "与 AMOS22 MRI 肾脏任务不可直接跨数据集比较；本表为同协议、同数据上的公平对比。",
    ]
    if robust:
        r = robust.get("0.1", {}).get("KidneyBand_ReliableSSL_v2_2_CSD", {})
        if r:
            lines.insert(2, f"10% 标签下 CSD 的 TKV 误差中位数 {r['median']:.3f}（IQR [{r['iqr_lo']:.3f}, {r['iqr_hi']:.3f}]），"
                            f"误差<30% 的病例占比 {r['pct_lt_30']:.1f}%，误差<10% 占比 {r['pct_lt_10']:.1f}%（n={r['n']}）。")
    return lines

def tkv_footnote_en():
    lines = [
        "Notes: *FRCNet is a clean-room, output-level FDC/MRSC proxy reproduced under our protocol following the MICCAI 2024 paper; metrics are per-subject means (n=40).",
        "At 10% labels, the CSD mean TKV relative error (2.054) is driven by 2/40 cases (5%) with CSD-specific over-segmentation failure "
        "(AMOS_0508: 39.3, AMOS_0514: 33.3; both Dice≈0.03, HD95>175 mm), dropping to 0.25 when excluded; the 8 hard cases with "
        "Dice<0.05 are shared by all three methods (B0+ and FRCNet also 8 each), i.e., hard samples of the 10%-label "
        "protocol rather than a method-specific defect. At 20% labels there are no over-segmentation failures (median 0.046).",
        "Literature comparisons (BoCLIS TMI 2025, VerSemi TMI 2025, FRCNet original) report 10%/20%-label results on ACDC/LA "
        "and are not directly comparable to this AMOS22 MRI kidney task; this table is a same-protocol, same-data comparison.",
    ]
    if robust:
        r = robust.get("0.1", {}).get("KidneyBand_ReliableSSL_v2_2_CSD", {})
        if r:
            lines.insert(2, f"At 10% labels, the CSD TKV-error median is {r['median']:.3f} (IQR [{r['iqr_lo']:.3f}, {r['iqr_hi']:.3f}]), "
                            f"with {r['pct_lt_30']:.1f}% of cases within 30% error and {r['pct_lt_10']:.1f}% within 10% (n={r['n']}).")
    return lines

# =========================================================== Markdown
# --- Table 3 CN ---
lines = ["**表3  内部数据集标签效率（T2KidneyMRI，Dice，5 折；12.5/50/100% 为 1 seed×5 折，25% 为正式 15 组）**", ""]
lines.append("| 标注比例 | KidneyBand-ReliableSSL v2.2 (CSD) | V22 B0+ (匹配基线) | Δ Dice |")
lines.append("|---|---|---|---|")
for frac, csd, b0, note_cn, _ in T3:
    lines.append(f"| {int(frac*100):g}% | {csd:.4f} | {b0:.4f} | {csd-b0:+.4f} |")
lines += ["", "注：Δ = CSD − B0+；100% 档同时满足 Q2 全标注 sanity（Dice ≥ 0.90）。"]
(OUT / "表3_内部标签效率.md").write_text("\n".join(lines), encoding="utf-8")

# --- Table 4 CN ---
lines = ["**表4  AMOS22 MRI 公共基准（5 折 × 1 seed，batch 16，同协议对比；指标为逐 subject 均值，n=40）**", ""]
lines.append("| 标注比例 | 方法 | Dice | HD95 (mm) | TKV 相对误差 |")
lines.append("|---|---|---|---|---|")
for frac, method, dice, hd95, tkv in T4:
    lines.append(f"| {int(frac*100):g}% | {M4[method]} | {dice:.4f} | {hd95:.2f} | {tkv:.3f} |")
lines += [""] + tkv_footnote_cn()
(OUT / "表4_AMOS公共基准.md").write_text("\n".join(lines), encoding="utf-8")

# --- Table 5 CN ---
lines = ["**表5  近两年 SSL 基线外部 source-only 泛化（Dice，内部训练后直接推理，零训练成本）**", ""]
lines.append("| 外部数据集 | KidneyBand-ReliableSSL v2.2 (CSD) | FRCNet 复现 | CMMT-Net 复现 |")
lines.append("|---|---|---|---|")
for ds, csd, frc, cmmt in T5:
    lines.append(f"| {D5[ds]} | {csd:.4f} | {frc:.4f} | {cmmt:.4f} |")
lines += ["", "注：FRCNet/CMMT 均为本协议下的 clean-room 复现；CSD 为正式外部结果（5 折集成 × 3 seeds 平均）。",
          "补充指标（HD95/TKV）见 external_supplement 明细 CSV。"]
(OUT / "表5_外部SSL基线.md").write_text("\n".join(lines), encoding="utf-8")

# --- Table 3 EN ---
lines = ["# Table 3. Label efficiency on the internal dataset (T2KidneyMRI, Dice, 5-fold; "
         "12.5/50/100% = 1 seed x 5 folds, 25% = formal 15 runs)", ""]
lines.append("| Label fraction | KidneyBand-ReliableSSL v2.2 (CSD) | V22 B0+ (matched baseline) | $\\Delta$Dice |")
lines.append("|---|---|---|---|")
for frac, csd, b0, _, _ in T3:
    lines.append(f"| {int(frac*100):g}% | {csd:.4f} | {b0:.4f} | {csd-b0:+.4f} |")
lines += ["", "Notes: $\\Delta$Dice = CSD $-$ B0+; the 100% row also satisfies the Q2 full-label sanity check (Dice $\\geq$ 0.90)."]
(OUT / "Table3_LabelEfficiency_EN.md").write_text("\n".join(lines), encoding="utf-8")

# --- Table 4 EN ---
lines = ["# Table 4. AMOS22 MRI public benchmark (5-fold x 1 seed, batch size 16, same protocol; "
         "per-subject means, n = 40)", ""]
lines.append("| Label fraction | Method | Dice | HD95 (mm) | TKV rel. error |")
lines.append("|---|---|---|---|---|")
for frac, method, dice, hd95, tkv in T4:
    lines.append(f"| {int(frac*100):g}% | {M4_EN[method]} | {dice:.4f} | {hd95:.2f} | {tkv:.3f} |")
lines += [""] + tkv_footnote_en()
(OUT / "Table4_AMOS_Benchmark_EN.md").write_text("\n".join(lines), encoding="utf-8")

# --- Table 5 EN ---
lines = ["# Table 5. External source-only generalization of recent SSL baselines (Dice; trained internally, "
         "zero fine-tuning cost)", ""]
lines.append("| External dataset | KidneyBand-ReliableSSL v2.2 (CSD) | FRCNet proxy | CMMT-Net proxy |")
lines.append("|---|---|---|---|")
for ds, csd, frc, cmmt in T5:
    lines.append(f"| {D5[ds]} | {csd:.4f} | {frc:.4f} | {cmmt:.4f} |")
lines += ["", "Notes: FRCNet/CMMT are clean-room proxies under our protocol; CSD is the formal external result "
          "(5-fold ensemble x 3 seeds mean). HD95/TKV details are in the external-supplement CSV."]
(OUT / "Table5_ExternalSSL_EN.md").write_text("\n".join(lines), encoding="utf-8")

# =========================================================== LaTeX (booktabs)
def esc(s):
    return s.replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("_", "\\_")

def tex_table(path, caption, label, header, rows, notes):
    out = ["% " + label + " (requires booktabs; compile with a Unicode/LaTeX engine or map glyphs)",
           "\\begin{table}[htbp]", "\\centering",
           "\\caption{" + caption + "}", "\\label{" + label + "}",
           "\\begin{tabular}{" + "l" * len(header) + "}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        out.append("\\begin{minipage}{\\linewidth}\\footnotesize " + " ".join(notes) + "\\end{minipage}")
    out += ["\\end{table}", ""]
    Path(path).write_text("\n".join(out), encoding="utf-8")

tex_table(OUT / "Table3_LabelEfficiency_EN.tex", "Label efficiency on the internal dataset (T2KidneyMRI, Dice, 5-fold). "
          "$\\Delta$Dice $>$ 0 favors ours.", "tab:label_efficiency",
          ["Label fraction", "CSD", "B0+ (matched)", "$\\Delta$Dice"],
          [[f"{int(frac*100):g}\\%", f"{csd:.4f}", f"{b0:.4f}", f"${csd-b0:+.4f}$"] for frac, csd, b0, _, _ in T3],
          ["12.5/50/100\\% = 1 seed $\\times$ 5 folds; 25\\% = formal 15 runs. The 100\\% row satisfies the "
           "Q2 full-label sanity check (Dice $\\geq$ 0.90)."])

tex_table(OUT / "Table4_AMOS_Benchmark_EN.tex",
          "AMOS22 MRI public benchmark (5-fold $\\times$ 1 seed, batch size 16, same protocol; per-subject means, $n=40$).",
          "tab:amos_benchmark",
          ["Label fraction", "Method", "Dice", "HD95 (mm)", "TKV rel. error"],
          [[f"{int(frac*100):g}\\%", esc(M4_EN[m]), f"{d:.4f}", f"{h:.2f}", f"{t:.3f}"] for frac, m, d, h, t in T4],
          [esc(n) for n in tkv_footnote_en()])

tex_table(OUT / "Table5_ExternalSSL_EN.tex",
          "External source-only generalization of recent SSL baselines (Dice; zero fine-tuning cost).",
          "tab:external_ssl",
          ["External dataset", "CSD", "FRCNet proxy", "CMMT-Net proxy"],
          [[D5[ds], f"{csd:.4f}", f"{frc:.4f}", f"{cmmt:.4f}"] for ds, csd, frc, cmmt in T5],
          ["FRCNet/CMMT are clean-room proxies under our protocol; CSD is the formal external result "
           "(5-fold ensemble $\\times$ 3 seeds mean)."])

# =========================================================== Word (three-line tables)
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

def docx_caption_cn(n, cap):
    return f"表 {n}  {cap}"

# CN docs
make_docx(OUT / "表3_内部标签效率.docx", docx_caption_cn(3, "内部数据集标签效率（T2KidneyMRI，Dice，5 折）"),
          ["标注比例", "CSD", "V22 B0+ (匹配基线)", "Δ Dice"],
          [[f"{int(f*100):g}%", f"{csd:.4f}", f"{b0:.4f}", f"{csd-b0:+.4f}"] for f, csd, b0, _, _ in T3],
          ["注: Δ = CSD − B0+；12.5/50/100% 为 1 seed×5 折，25% 为正式 15 组；100% 档满足 Q2 全标注 sanity（Dice ≥ 0.90）。"],
          cn=True)
make_docx(OUT / "表4_AMOS公共基准.docx", docx_caption_cn(4, "AMOS22 MRI 公共基准（5 折 × 1 seed，batch 16，逐 subject 均值，n=40）"),
          ["标注比例", "方法", "Dice", "HD95 (mm)", "TKV 相对误差"],
          [[f"{int(f*100):g}%", M4[m], f"{d:.4f}", f"{h:.2f}", f"{t:.3f}"] for f, m, d, h, t in T4],
          tkv_footnote_cn(), cn=True)
make_docx(OUT / "表5_外部SSL基线.docx", docx_caption_cn(5, "近两年 SSL 基线外部 source-only 泛化（Dice）"),
          ["外部数据集", "CSD", "FRCNet 复现", "CMMT-Net 复现"],
          [[D5[ds], f"{csd:.4f}", f"{frc:.4f}", f"{cmmt:.4f}"] for ds, csd, frc, cmmt in T5],
          ["注: FRCNet/CMMT 均为本协议下的 clean-room 复现；CSD 为正式外部结果（5 折集成 × 3 seeds 平均）。"], cn=True)

# EN docs
make_docx(OUT / "Table3_LabelEfficiency_EN.docx",
          "Table 3. Label efficiency on the internal dataset (T2KidneyMRI, Dice, 5-fold)",
          ["Label fraction", "CSD", "V22 B0+ (matched)", "$\\Delta$Dice"],
          [[f"{int(f*100):g}%", f"{csd:.4f}", f"{b0:.4f}", f"{csd-b0:+.4f}"] for f, csd, b0, _, _ in T3],
          ["Notes: DeltaDice = CSD - B0+; 12.5/50/100% = 1 seed x 5 folds, 25% = formal 15 runs; "
           "the 100% row satisfies the Q2 full-label sanity check (Dice >= 0.90)."])
make_docx(OUT / "Table4_AMOS_Benchmark_EN.docx",
          "Table 4. AMOS22 MRI public benchmark (5-fold x 1 seed, batch size 16, per-subject means, n = 40)",
          ["Label fraction", "Method", "Dice", "HD95 (mm)", "TKV rel. error"],
          [[f"{int(f*100):g}%", M4_EN[m], f"{d:.4f}", f"{h:.2f}", f"{t:.3f}"] for f, m, d, h, t in T4],
          tkv_footnote_en())
make_docx(OUT / "Table5_ExternalSSL_EN.docx",
          "Table 5. External source-only generalization of recent SSL baselines (Dice)",
          ["External dataset", "CSD", "FRCNet proxy", "CMMT-Net proxy"],
          [[D5[ds], f"{csd:.4f}", f"{frc:.4f}", f"{cmmt:.4f}"] for ds, csd, frc, cmmt in T5],
          ["Notes: FRCNet/CMMT are clean-room proxies under our protocol; CSD is the formal external result "
           "(5-fold ensemble x 3 seeds mean)."])


# =========================================================== Table 6: efficiency
E6 = {}
with open(RESULTS / "efficiency_summary.csv", encoding="utf-8") as fh:
    for rec in csv.DictReader(fh):
        E6[rec["method"]] = (int(rec["params"]), float(rec["batch8_latency_ms"]), float(rec["est_30slice_volume_s"]))
EFF_HEAD = ["KidneyBand_ReliableSSL_v2_2_CSD", "V22_B0plus_5slice"]
EFF_ORDER = EFF_HEAD + sorted((m for m in E6 if m not in EFF_HEAD), key=lambda m: E6[m][0])

M6_CN = dict(M4)
M6_CN.update({"daniel_unet": "Daniel-UNet", "damiano_mcd_plus": "Damiano-UNet", "unetpp_5slice": "UNet++",
              "tcsm_5slice": "TCSM", "cmmt_5slice": "CMMT-Net", "frcnet_5slice": "FRCNet",
              "semi_urf_5slice": "Semi-URF", "randdg_5slice": "RandDG"})
M6_EN = dict(M4_EN)
M6_EN.update({"daniel_unet": "Daniel-UNet", "damiano_mcd_plus": "Damiano-UNet", "unetpp_5slice": "UNet++",
              "tcsm_5slice": "TCSM", "cmmt_5slice": "CMMT-Net", "frcnet_5slice": "FRCNet",
              "semi_urf_5slice": "Semi-URF", "randdg_5slice": "RandDG"})

nnunet = {}
npath = RESULTS / "nnunet_efficiency.json"
if npath.exists():
    nnunet = json.loads(npath.read_text(encoding="utf-8"))

def eff_rows_cn():
    rows = [[M6_CN[m], f"{E6[m][0]/1e6:.2f} M", f"{E6[m][1]:.2f}", f"{E6[m][2]:.2f}"] for m in EFF_ORDER]
    if nnunet:
        rows.append(["nnU-Net v2 3D (3D fullres)", f"{nnunet['params']/1e6:.2f} M",
                     f"{nnunet['latency_ms']:.2f}*", f"{nnunet['inference_s']:.2f}*"])
    else:
        rows.append(["nnU-Net v2 3D (3D fullres)", "待补", "待补", "待补"])
    return rows

def eff_rows_en():
    rows = [[M6_EN[m], f"{E6[m][0]/1e6:.2f} M", f"{E6[m][1]:.2f}", f"{E6[m][2]:.2f}"] for m in EFF_ORDER]
    if nnunet:
        rows.append(["nnU-Net v2 3D (3D fullres)", f"{nnunet['params']/1e6:.2f} M",
                     f"{nnunet['latency_ms']:.2f}*", f"{nnunet['inference_s']:.2f}*"])
    else:
        rows.append(["nnU-Net v2 3D (3D fullres)", "TBD", "TBD", "TBD"])
    return rows

def eff_notes_cn():
    lines = ["注：测量条件为 RTX 4090、batch=8×256×256 输入、20 次前向平均；参数量由 build_model 实测。",
             "CSD 与匹配基线 B0+ 及多数 2.5D 基线共享同一主干（6.50 M），即全部正则项不增加任何推理成本；"
             "估计单卷 = 单批延迟 × 30 切片（保守上界，实际批量推理更快）。"]
    if nnunet:
        lines.append(f"*nnU-Net v2 3D 为 sliding-window 推理（batch1×{nnunet['patch']} patch，30 层参考卷估约 ×{nnunet.get('n_patches', 9)} patch），"
                     "条件与 2.5D 不同，仅作量级参考。")
    else:
        lines.append("*nnU-Net v2 3D 行待补测（sliding-window，batch1×3D patch）。")
    return lines

def eff_notes_en():
    lines = ["Notes: measured on RTX 4090 with batch=8 x 256 x 256 inputs, averaged over 20 forward passes; params counted from build_model.",
             "CSD shares the same backbone as the matched baseline B0+ and most 2.5D baselines (6.50 M), so all regularization terms add zero inference cost; "
             "per-volume estimate = batch-8 latency x 30 slices (conservative upper bound; batched inference is faster in practice)."]
    if nnunet:
        lines.append(f"*nnU-Net v2 3D uses sliding-window inference (batch1 x {nnunet['patch']} patch, ~x{nnunet.get('n_patches', 9)} patches per 30-slice volume); "
                     "conditions differ from the 2.5D rows and are indicative only.")
    else:
        lines.append("*nnU-Net v2 3D row pending (sliding-window, batch1 x 3D patch).")
    return lines

# --- Table 6 CN ---
lines = ["**表6  效率对比（RTX 4090；参数量与推理延迟，2.5D 方法同协议实测）**", ""]
lines.append("| 方法 | 参数量 | batch8 延迟 (ms) | 估计单卷推理 (s) |")
lines.append("|---|---|---|---|")
for r in eff_rows_cn():
    lines.append("| " + " | ".join(r) + " |")
lines += [""] + eff_notes_cn()
(OUT / "表6_效率对比.md").write_text("\n".join(lines), encoding="utf-8")

# --- Table 6 EN ---
lines = ["# Table 6. Efficiency comparison (RTX 4090; params and inference latency, 2.5D methods measured under the same protocol)", ""]
lines.append("| Method | Params | Batch-8 latency (ms) | Est. per-volume (s) |")
lines.append("|---|---|---|---|")
for r in eff_rows_en():
    lines.append("| " + " | ".join(r) + " |")
lines += [""] + eff_notes_en()
(OUT / "Table6_Efficiency_EN.md").write_text("\n".join(lines), encoding="utf-8")

tex_table(OUT / "Table6_Efficiency_EN.tex",
          "Efficiency comparison (RTX 4090; parameters and inference latency). CSD shares the backbone with the matched baseline B0+ and most 2.5D baselines (6.50 M), i.e., all regularization terms add zero inference cost.",
          "tab:efficiency",
          ["Method", "Params", "Batch-8 latency (ms)", "Est. per-volume (s)"],
          eff_rows_en(),
          [esc(n) for n in eff_notes_en()])

make_docx(OUT / "表6_效率对比.docx", docx_caption_cn(6, "效率对比（RTX 4090；参数量与推理延迟）"),
          ["方法", "参数量", "batch8 延迟 (ms)", "估计单卷推理 (s)"], eff_rows_cn(), eff_notes_cn(), cn=True)
make_docx(OUT / "Table6_Efficiency_EN.docx",
          "Table 6. Efficiency comparison (RTX 4090; params and inference latency)",
          ["Method", "Params", "Batch-8 latency (ms)", "Est. per-volume (s)"], eff_rows_en(), eff_notes_en())

print("OK ->", OUT)


