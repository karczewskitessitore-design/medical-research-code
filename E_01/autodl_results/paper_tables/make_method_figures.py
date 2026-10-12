# -*- coding: utf-8 -*-
"""Generate method figures mimicking k-band (Wang et al. 2023) Fig.1-3."""
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

OUT = r"D:\30 医学论文\01\autodl_results\paper_tables"
IMG = r"D:\30 医学论文\01\data\T2KidneyMRI\raw\CKD_03_T2W.nii.gz"
MSK = r"D:\30 医学论文\01\data\T2KidneyMRI\raw\CKD_03_T2W_mask.nii.gz"
PRB = r"D:\30 医学论文\01\experiments\KidneyMRI-SpectralSSL\B0\fold0_seed20260802\predictions\CKD_03_1_prob.nii.gz"
LOG = r"D:\30 医学论文\01\autodl_results\audit_v22_csd_primary_20260811T144011Z\experiments\benchmarks\kidneyband_reliable_ssl_q2_v2_2_csd_comparison_autodl_v1\internal\KidneyBand_ReliableSSL_v2_2_CSD\fold0_seed20260811\benchmark_train.log"

def variable_density_probability(H, W, center_fraction=0.08, base_keep_rate=0.70, radial_power=4):
    fy = np.abs(np.fft.fftfreq(H))[:, None]
    fx = np.abs(np.fft.fftfreq(W))[None, :]
    radius = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2).clip(0, 1)
    center = radius <= center_fraction
    score = np.clip((1.0 - radius) ** radial_power, 1e-6, None)
    lo, hi = 0.0, 1e6
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if np.clip(score[~center] * mid, None, 1.0).mean() < base_keep_rate:
            lo = mid
        else:
            hi = mid
    q = np.clip(score * ((lo + hi) / 2.0), None, 1.0)
    q[center] = 1.0
    return q, center

def complementary_masks(H, W, center_fraction=0.08, base_keep_rate=0.70, radial_power=4, seed=7):
    rng = np.random.default_rng(seed)
    q, center = variable_density_probability(H, W, center_fraction, base_keep_rate, radial_power)
    row = np.arange(H)[:, None] + np.zeros((H, W), dtype=int)
    col = np.zeros((H, W), dtype=int) + np.arange(W)[None, :]
    flat = row * W + col
    partner = ((-row) % H) * W + ((-col) % W)
    source = np.minimum(flat, partner).ravel()
    u = rng.random(H * W)
    v = rng.random(H * W)
    base = (u[source] < q.ravel()).reshape(H, W)
    choose = (v[source] < 0.5).reshape(H, W)
    theta = base & choose
    lam = base & ~choose
    theta[center] = True
    lam[center] = True
    return theta.astype(float), lam.astype(float), center, q

def fourier_view(p, mask):
    spec = np.fft.fftshift(np.fft.fft2(p, norm="ortho"))
    return np.abs(np.fft.ifft2(np.fft.ifftshift(spec * mask), norm="ortho"))

def js_divergence(a, b, eps=1e-6):
    a = np.clip(a, eps, None); b = np.clip(b, eps, None)
    m = 0.5 * (a + b)
    js = 0.5 * (a * (np.log(a) - np.log(m)) + b * (np.log(b) - np.log(m)))
    return np.clip(js / np.log(2.0), 0, 1)

def sobel_mag(v):
    gx = ndimage.sobel(v, axis=1)
    gy = ndimage.sobel(v, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2 + 1e-8)

def boundary_band(p, width=5):
    hard = (p >= 0.5).astype(float)
    it = width // 2
    dil = ndimage.binary_dilation(hard, iterations=it)
    ero = ndimage.binary_erosion(hard, iterations=it)
    return dil & ~ero

def rgb_mask(base, m, color):
    out = np.stack([base] * 3, axis=-1)
    out[m.astype(bool)] = color
    return out

# ---------------- data ----------------
img = np.asarray(nib.load(IMG).get_fdata())
msk = np.asarray(nib.load(MSK).get_fdata())
prb = np.asarray(nib.load(PRB).get_fdata())
slice_idx = int(np.argmax(prb.sum(axis=(0, 1))))
x = img[:, :, slice_idx].T
x = (x - np.percentile(x, 0.5)) / (np.percentile(x, 99.5) - np.percentile(x, 0.5) + 1e-6)
x = np.clip(x, 0, 1)
y = msk[:, :, slice_idx].T
p = prb[:, :, slice_idx].T
p = np.clip((p - 0.05) / 0.9, 0, 1)

H, W = p.shape
theta, lam, center, q = complementary_masks(H, W, seed=7)
v_t = fourier_view(p, theta)
v_l = fourier_view(p, lam)
diff = np.abs(v_t - v_l)
js_map = js_divergence(v_t, v_l)
ent = -(np.clip(p, 1e-6, None) * np.log(np.clip(p, 1e-6, None)) +
         np.clip(1 - p, 1e-6, None) * np.log(np.clip(1 - p, 1e-6, None))) / np.log(2)
rel = (1 - js_map) * (1 - ent)
bd = boundary_band(p)
fg = (p >= 0.5) & ~bd
bg = (p < 0.5) & ~bd

# ---------------- training curves ----------------
epochs, c_cons, c_bnd, c_dis, c_sup = [], [], [], [], []
with open(LOG, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            epochs.append(d["epoch"]); c_cons.append(d["loss_consistency"])
            c_bnd.append(d["loss_boundary"]); c_dis.append(d["loss_disagreement"])
            c_sup.append(d["loss_supervised"])
epochs = np.array(epochs); c_cons = np.array(c_cons); c_bnd = np.array(c_bnd)
c_dis = np.array(c_dis); c_sup = np.array(c_sup)

# ---------------- Figure 1 ----------------
fig = plt.figure(figsize=(13.5, 8.2), dpi=300)
gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1.0], hspace=0.45, wspace=0.35,
                      left=0.05, right=0.98, top=0.90, bottom=0.08)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(x, cmap="gray")
ax.imshow(np.ma.masked_where(~(y > 0.5), y), cmap="Reds", alpha=0.55)
ax.set_title("(a) 5-slice input + bilateral labels", fontsize=9); ax.axis("off")

ax = fig.add_subplot(gs[0, 1])
ax.imshow(p, cmap="gray"); ax.set_title("(b) prob. map $p$", fontsize=9); ax.axis("off")
ax = fig.add_subplot(gs[0, 2])
ax.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(p, norm="ortho"))) + 1e-6), cmap="inferno")
ax.set_title("spectrum $|\\mathcal{F}(p)|$", fontsize=9); ax.axis("off")
ax = fig.add_subplot(gs[0, 3])
ax.imshow(theta, cmap="gray"); ax.set_title("mask $\\Theta$", fontsize=9); ax.axis("off")
ax = fig.add_subplot(gs[0, 4])
ax.imshow(lam, cmap="gray"); ax.set_title("mask $\\Lambda$", fontsize=9); ax.axis("off")
ax = fig.add_subplot(gs[0, 5])
ax.imshow(v_t, cmap="gray"); ax.set_title("view $V_\\Theta$", fontsize=9); ax.axis("off")

ax = fig.add_subplot(gs[1, 0])
ax.imshow(js_map, cmap="inferno"); ax.set_title("(c) JS disagreement", fontsize=9); ax.axis("off")
ax = fig.add_subplot(gs[1, 1])
ax.imshow(ent, cmap="inferno"); ax.set_title("teacher entropy", fontsize=9); ax.axis("off")
ax = fig.add_subplot(gs[1, 2])
ax.imshow(rel, cmap="viridis"); ax.set_title("reliability $r$", fontsize=9); ax.axis("off")

overlay = rgb_mask(x, np.ones_like(x, dtype=bool), np.array([0.35, 0.35, 0.35]))
overlay = rgb_mask(x, fg, (0.0, 0.8, 0.2))
overlay = rgb_mask(x, bd, (1.0, 0.2, 0.1))
ax = fig.add_subplot(gs[1, 3])
ax.imshow(overlay); ax.set_title("FG / BG / BD regions", fontsize=9); ax.axis("off")

ax = fig.add_subplot(gs[1, 4])
ax.imshow(sobel_mag(p), cmap="magma"); ax.set_title("(d) Sobel boundary", fontsize=9); ax.axis("off")

ax = fig.add_subplot(gs[1, 5])
ax.plot(epochs, c_sup, label="supervised", lw=1.4)
ax.plot(epochs, c_cons, label="CSD prob.", lw=1.4)
ax.plot(epochs, c_bnd, label="boundary", lw=1.4)
ax.plot(epochs, c_dis, label="disagreement", lw=1.4)
ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("loss", fontsize=8)
ax.set_title("(e) training curves (fold 0)", fontsize=9)
ax.legend(fontsize=6.5, frameon=False); ax.tick_params(labelsize=7)
ax.set_xlim(0, 150)

fig.suptitle("Figure 1  |  The proposed CSD-MT framework (illustrated on real data, CKD-03)", fontsize=12, y=0.97)
fig.savefig(f"{OUT}\\Figure1_MethodOverview.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------- Figure 2 ----------------
fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), dpi=300,
                         gridspec_kw=dict(hspace=0.45, wspace=0.3, left=0.05, right=0.98, top=0.88, bottom=0.1))
for ax, cf in zip(axes[0], [0.04, 0.08, 0.16, 0.08]):
    qq, _ = variable_density_probability(H, W, center_fraction=cf)
    ax.imshow(qq, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(f"center fraction $c$ = {cf}", fontsize=8)
    ax.axis("off")
axes[0, 3].imshow(theta + 2 * lam, cmap="bwr")
axes[0, 3].set_title("$\\Theta$ + 2$\\Lambda$ (disjoint)", fontsize=8); axes[0, 3].axis("off")

qq, _ = variable_density_probability(H, W, base_keep_rate=0.5)
axes[1, 0].imshow(qq, cmap="gray_r", vmin=0, vmax=1)
axes[1, 0].set_title("base keep $\\gamma$ = 0.5", fontsize=8); axes[1, 0].axis("off")
qq, _ = variable_density_probability(H, W, base_keep_rate=0.9)
axes[1, 1].imshow(qq, cmap="gray_r", vmin=0, vmax=1)
axes[1, 1].set_title("base keep $\\gamma$ = 0.9", fontsize=8); axes[1, 1].axis("off")

ax = axes[1, 2]
rr = np.linspace(0, 1, 200)
for k, col in zip([1, 2, 4, 8], ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]):
    fy = np.abs(np.fft.fftfreq(400))[:200]
    radius = fy
    score = np.clip((1 - radius) ** k, 1e-6, None)
    lo, hi = 0.0, 1e6
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.clip(score * mid, None, 1.0).mean() < 0.7:
            lo = mid
        else:
            hi = mid
    qq = np.clip(score * ((lo + hi) / 2), None, 1.0)
    qq[radius <= 0.08] = 1.0
    ax.plot(rr, qq, label=f"power {k}", color=col, lw=1.6)
ax.axvline(0.08, color="k", ls="--", lw=1)
ax.text(0.085, 0.92, "$c$=0.08", fontsize=7)
ax.set_xlabel("radial frequency $r$", fontsize=8); ax.set_ylabel("$q(r)$", fontsize=8)
ax.set_title("radial profile of $q(r)$", fontsize=8)
ax.legend(fontsize=7, frameon=False); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=7)

ax = axes[1, 3]
ax.imshow(rel, cmap="viridis"); ax.set_title("reliability map (real)", fontsize=8); ax.axis("off")
fig.suptitle("Figure 2  |  Variable-density keep probability $q(r)$ and complementary masks", fontsize=12, y=0.95)
fig.savefig(f"{OUT}\\Figure2_Masks_Profiles.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------- Figure 3 ----------------
fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.0), dpi=300,
                         gridspec_kw=dict(hspace=0.35, wspace=0.3, left=0.05, right=0.98, top=0.88, bottom=0.1))
axes[0, 0].imshow(theta, cmap="gray"); axes[0, 0].set_title("mask $\\Theta$", fontsize=8); axes[0, 0].axis("off")
axes[0, 1].imshow(lam, cmap="gray"); axes[0, 1].set_title("mask $\\Lambda$", fontsize=8); axes[0, 1].axis("off")
axes[0, 2].imshow(np.minimum(theta, lam), cmap="gray"); axes[0, 2].set_title("$\\Theta \\cap \\Lambda$ (shared center)", fontsize=8); axes[0, 2].axis("off")
axes[0, 3].imshow(theta + 2 * lam, cmap="bwr"); axes[0, 3].set_title("$\\Theta$+2$\\Lambda$ (disjoint)", fontsize=8); axes[0, 3].axis("off")
axes[1, 0].imshow(v_t, cmap="gray"); axes[1, 0].set_title("view $V_\\Theta$", fontsize=8); axes[1, 0].axis("off")
axes[1, 1].imshow(v_l, cmap="gray"); axes[1, 1].set_title("view $V_\\Lambda$", fontsize=8); axes[1, 1].axis("off")
axes[1, 2].imshow(diff, cmap="inferno"); axes[1, 2].set_title("$|V_\\Theta - V_\\Lambda|$", fontsize=8); axes[1, 2].axis("off")
axes[1, 3].imshow(rel, cmap="viridis"); axes[1, 3].set_title("reliability $r$", fontsize=8); axes[1, 3].axis("off")
fig.suptitle("Figure 3  |  Complementary views and per-pixel reliability (real data, CKD-03)", fontsize=12, y=0.95)
fig.savefig(f"{OUT}\\Figure3_Views_Reliability.png", dpi=300, facecolor="white")
plt.close(fig)

print("OK slice_idx =", slice_idx)


