# -*- coding: utf-8 -*-
"""Figure 2 (candidate): subject-level scatter of Dice vs TKV relative error.
AMOS22 MRI, 10%/20% labels; CSD vs B0+ vs FRCNet proxy. 300 dpi, EN + CN."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

RESULTS = Path(r"D:\30 医学论文\01\autodl_results")
OUT = RESULTS / "paper_tables"

for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"):
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rows = list(csv.DictReader((RESULTS / "amos_subject_stats.csv").open(encoding="utf-8")))
data = {}
for r in rows:
    key = (r["fraction"], r["method"])
    data.setdefault(key, []).append({
        "sid": r["subject_id"], "dice": float(r["dice"]),
        "tkv": max(float(r["tkv_relative_error"]), 0.005)})

METHODS = ["KidneyBand_ReliableSSL_v2_2_CSD", "V22_B0plus_5slice", "frcnet_5slice"]
STYLE_EN = {"KidneyBand_ReliableSSL_v2_2_CSD": ("#1f77b4", "KidneyBand-ReliableSSL v2.2 (CSD)"),
            "V22_B0plus_5slice": ("#ff7f0e", "V22 B0+ (matched baseline)"),
            "frcnet_5slice": ("#2ca02c", "FRCNet proxy")}
STYLE_CN = {"KidneyBand_ReliableSSL_v2_2_CSD": ("#1f77b4", "KidneyBand-ReliableSSL v2.2 (CSD)"),
            "V22_B0plus_5slice": ("#ff7f0e", "V22 B0+ (匹配基线)"),
            "frcnet_5slice": ("#2ca02c", "FRCNet 复现")}
SHORT = {"KidneyBand_ReliableSSL_v2_2_CSD": "CSD", "V22_B0plus_5slice": "B0+", "frcnet_5slice": "FRCNet"}

def draw(path, cn):
    style = STYLE_CN if cn else STYLE_EN
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.125, wspace=0.17)
    for ax, frac, title in ((axes[0], "0p1", "10% 标签" if cn else "10% labels"),
                            (axes[1], "0p2", "20% 标签" if cn else "20% labels")):
        for method in METHODS:
            color, label = style[method]
            pts = data[(frac, method)]
            ax.scatter([p["tkv"] for p in pts], [p["dice"] for p in pts],
                       s=26, alpha=0.72, color=color, label=label, edgecolors="none", zorder=3)
        ax.axhspan(0, 0.05, color="#b03a2e", alpha=0.10, zorder=0)
        ax.axhline(0.05, color="#b03a2e", linewidth=0.7, linestyle=":", alpha=0.6)
        ax.text(0.011, 0.012, "困难例区域 (Dice<0.05)" if cn else "hard-case region (Dice<0.05)",
                fontsize=7.5, color="#8b2c22")
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.text(1.15, 0.985, "TKV 误差 100%" if cn else "100% TKV error",
                fontsize=7.5, color="gray", va="top")
        if frac == "0p1":
            for sid, tx, ty in (("AMOS_0508", 60, 0.20), ("AMOS_0514", 18, 0.36)):
                p = next(p for p in data[(frac, "KidneyBand_ReliableSSL_v2_2_CSD")] if p["sid"] == sid)
                ax.annotate(f"{sid}\n误差 {p['tkv']:.0f}×" if cn else f"{sid}\nerr {p['tkv']:.0f}x",
                            xy=(p["tkv"], p["dice"]), xytext=(tx, ty), fontsize=8, color="#1f77b4",
                            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0), zorder=5)
        counts = "  |  ".join(f"{SHORT[m]} {sum(1 for p in data[(frac, m)] if p['dice'] < 0.05)}"
                              for m in METHODS)
        ax.text(0.99, 0.985, ("Dice<0.05: " if cn else "Dice<0.05: ") + counts,
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85), zorder=6)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 100)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("TKV 相对误差（对数轴）" if cn else "TKV relative error (log scale)", fontsize=10)
        ax.set_ylabel("Dice", fontsize=10)
        ax.grid(axis="x", alpha=0.25, which="both")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)

draw(OUT / "Figure2_subject_scatter_EN.png", cn=False)
draw(OUT / "图2_subject散点图_CN.png", cn=True)
print("OK ->", OUT)
