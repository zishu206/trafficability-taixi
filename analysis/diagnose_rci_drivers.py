# -*- coding: utf-8 -*-
"""
诊断 RCI 与 13 通道、AHP 综合分 S 的关系，定位 S-RCI 弱负相关来源。
输出 rci_driver_diagnosis.txt。
"""
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
OUT = os.path.join(OUT_DIR, "rci_driver_diagnosis.txt")

d = np.load(NPZ)
X = d["X"]
S = d["S"].reshape(-1)
names = ["NLI", "NBD", "NCLAY", "NSAL", "NAI", "NSI", "NCUR", "NDW",
         "NWDI", "NSAR", "NLST", "NVFI", "NLULC"]

# Re-read the RCI raster
import rasterio
from rasterio.windows import Window
RCI_TIF = os.path.join(DATA_DIR, "mech_RCI_30m.tif")
GR, GC = d["GR"], d["GC"]
parts = []
with rasterio.open(RCI_TIF) as rf:
    for i in range(len(GR)):
        r, c = int(GR[i]) * 128, int(GC[i]) * 128
        parts.append(rf.read(1, window=Window(c, r, 128, 128)).reshape(-1))
RCI = np.concatenate(parts)
lnRCI = np.log(RCI)

lines = ["Correlation of RCI with each channel (common domain, %d pixels)" % len(RCI),
         "Channel   Pearson(RCI)  Spearman(RCI)  Pearson(lnRCI)"]
for k, nm in enumerate(names):
    xk = X[:, k].reshape(-1)
    a = pearsonr(xk, RCI).statistic
    b = spearmanr(xk, RCI).statistic
    c = pearsonr(xk, lnRCI).statistic
    lines.append("%-8s %12.4f %14.4f %14.4f" % (nm, a, b, c))
lines.append("")
lines.append("S(composite)  Pearson(RCI)=%.4f  Spearman=%.4f  Pearson(lnRCI)=%.4f"
             % (pearsonr(S, RCI).statistic, spearmanr(S, RCI).statistic,
                pearsonr(S, lnRCI).statistic))

# The material dimension (NLI) against RCI; also inspect after removing the material term.
lines.append("")
lines.append("Group means by mechanistic class (direction of AHP indicators across RCI levels)")
Ymech = d["Y_mech"].reshape(-1)
for k, nm in enumerate(names):
    xk = X[:, k].reshape(-1)
    m = [xk[Ymech == g].mean() for g in range(3)]
    lines.append("%-8s mech low=%.3f medium=%.3f high=%.3f" % (nm, m[0], m[1], m[2]))
mS = [S[Ymech == g].mean() for g in range(3)]
lines.append("S        mech low=%.3f medium=%.3f high=%.3f" % (mS[0], mS[1], mS[2]))

report = "\n".join(lines)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
