# -*- coding: utf-8 -*-
"""
三标签口径对比（一）：AHP 标签 vs Bekker-Wong 机理标签的空间一致性。
- 三级标签 3x3 混淆、总体一致率、Cohen's Kappa
- 连续量：AHP 分数 S 与 RCI 的 Pearson / Spearman（含对数 RCI）
输出 label_consistency.txt 与 fig_label_confusion.png、fig_S_RCI_scatter.png。
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

plt.rcParams["axes.unicode_minus"] = False

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
RCI_TIF = os.path.join(DATA_DIR, "mech_RCI_30m.tif")
OUTDIR = OUT_DIR
P = 128
os.makedirs(OUTDIR, exist_ok=True)

d = np.load(NPZ)
Yahp = d["Y_ahp"].reshape(-1).astype(np.int64)
Ymech = d["Y_mech"].reshape(-1).astype(np.int64)
S = d["S"].reshape(-1).astype(np.float64)
GR, GC = d["GR"], d["GC"]
N = len(Yahp)

# ---- 三级标签一致性 ----
cm = confusion_matrix(Ymech, Yahp, labels=[0, 1, 2])   # 行=机理，列=AHP
agree = np.trace(cm) / cm.sum()
kappa = cohen_kappa_score(Ymech, Yahp)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# ---- 连续 RCI（按公共域网格窗口读取）----
rci_parts = []
with rasterio.open(RCI_TIF) as rf:
    for i in range(len(GR)):
        r, c = int(GR[i]) * P, int(GC[i]) * P
        v = rf.read(1, window=Window(c, r, P, P)).reshape(-1)
        rci_parts.append(v)
RCI = np.concatenate(rci_parts).astype(np.float64)
assert len(RCI) == N and np.isfinite(RCI).all()

pr = pearsonr(S, RCI)
sp = spearmanr(S, RCI)
pr_log = pearsonr(S, np.log(RCI))

# ---- 文本输出 ----
lines = []
lines.append("Spatial consistency of the AHP label vs the Bekker-Wong mechanistic label "
             "(common domain, %d pixels)" % N)
lines.append("")
lines.append("Three-class confusion matrix (rows = mechanistic, columns = AHP; counts):")
lines.append("Mech\\AHP          Low         Medium         High")
for i, nm in enumerate(["Low", "Medium", "High"]):
    lines.append("%-6s %12d %13d %13d" % (nm, cm[i, 0], cm[i, 1], cm[i, 2]))
lines.append("")
lines.append("Row-normalized (share of each mechanistic class assigned to AHP levels):")
for i, nm in enumerate(["Low", "Medium", "High"]):
    lines.append("%-6s %12.4f %13.4f %13.4f" % (nm, cm_norm[i, 0], cm_norm[i, 1], cm_norm[i, 2]))
lines.append("")
lines.append("Overall agreement = %.4f" % agree)
lines.append("Cohen's Kappa = %.4f" % kappa)
lines.append("")
lines.append("Continuous measures: AHP score S vs RCI")
lines.append("Pearson  r = %.4f (p=%.2e)" % (pr.statistic, pr.pvalue))
lines.append("Spearman rho = %.4f (p=%.2e)" % (sp.statistic, sp.pvalue))
lines.append("Pearson(S, lnRCI) = %.4f" % pr_log.statistic)
report = "\n".join(lines)
with open(os.path.join(OUTDIR, "label_consistency.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print(report)

# ---- 图1：标签混淆热力图 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, mat, fmt, title in [
    (axes[0], cm, "d", "Counts (rows = mechanistic, columns = AHP)"),
    (axes[1], cm_norm, ".2f", "Row-normalized proportions")]:
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["Low", "Medium", "High"]); ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("AHP label"); ax.set_ylabel("Mechanistic label")
    ax.set_title(title)
    th = mat.max() * 0.6
    for i in range(3):
        for j in range(3):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                    color="white" if mat[i, j] > th else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Spatial consistency of the three-class AHP and mechanistic labels "
             "(agreement %.2f%%, Kappa %.3f)" % (100 * agree, kappa), fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, "fig_label_confusion.png"), dpi=200)
plt.close(fig)

# ---- 图2：S vs RCI 散点（抽样，避免过密）----
rng = np.random.default_rng(0)
idx = rng.choice(N, size=min(20000, N), replace=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(S[idx], RCI[idx], s=4, alpha=0.25, color="#2E75B6")
axes[0].set_xlabel("AHP continuous score S"); axes[0].set_ylabel("RCI / psi")
axes[0].set_title("S vs RCI (Pearson %.3f)" % pr.statistic)
axes[1].scatter(S[idx], np.log(RCI[idx]), s=4, alpha=0.25, color="#C55A11")
axes[1].set_xlabel("AHP continuous score S"); axes[1].set_ylabel("ln(RCI)")
axes[1].set_title("S vs ln(RCI) (Pearson %.3f)" % pr_log.statistic)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig_S_RCI_scatter.png"), dpi=200)
plt.close(fig)
print("\nFigures saved:", OUTDIR)
