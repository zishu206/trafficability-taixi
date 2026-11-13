# -*- coding: utf-8 -*-
"""
对照原始土壤参数验证 RCI 内部合理性，并分析 NLI 与原始土壤参数关系，
定位 NLI-RCI 负相关的来源。输出 mech_vs_nli_diagnosis.txt。
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.stats import pearsonr, spearmanr
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
SOIL = os.path.join(DATA_DIR, "soil_physical_raw_30m.tif")
RCI_TIF = os.path.join(DATA_DIR, "mech_RCI_30m.tif")
OUT = os.path.join(OUT_DIR, "mech_vs_nli_diagnosis.txt")

d = np.load(NPZ)
GR, GC = d["GR"], d["GC"]
NLI = d["X"][:, 0].reshape(-1)
S = d["S"].reshape(-1)

soil_names = ["SAND", "SILT", "CLAY", "BD", "SOC", "W33", "TEX"]
soil_acc = [[] for _ in range(7)]
rci_acc = []
with rasterio.open(SOIL) as sf, rasterio.open(RCI_TIF) as rf:
    for i in range(len(GR)):
        r, c = int(GR[i]) * 128, int(GC[i]) * 128
        win = Window(c, r, 128, 128)
        sb = sf.read(window=win)
        for b in range(7):
            soil_acc[b].append(sb[b].reshape(-1))
        rci_acc.append(rf.read(1, window=win).reshape(-1))
soil = [np.concatenate(a) for a in soil_acc]
RCI = np.concatenate(rci_acc)
lnRCI = np.log(RCI)

lines = ["Common domain, %d pixels: correlation of raw soil parameters with RCI and NLI" % len(RCI), ""]
lines.append("Variable  with RCI(Pearson)  with RCI(Spearman)  with NLI(Pearson)  with S(Pearson)")
for b, nm in enumerate(soil_names):
    xb = soil[b]
    lines.append("%-8s %14.4f %16.4f %15.4f %13.4f" % (
        nm, pearsonr(xb, RCI).statistic, spearmanr(xb, RCI).statistic,
        pearsonr(xb, NLI).statistic, pearsonr(xb, S).statistic))
lines.append("")
lines.append("NLI with RCI Pearson=%.4f; NLI with lnRCI=%.4f"
             % (pearsonr(NLI, RCI).statistic, pearsonr(NLI, lnRCI).statistic))

# Coarse texture grouping; inspect RCI and NLI means.
lines.append("")
lines.append("Means grouped by SAND terciles:")
qs = np.quantile(soil[0], [0.333, 0.667])
grp = np.digitize(soil[0], qs)
for g in range(3):
    m = grp == g
    lines.append("Sand group %d  mean SAND=%.1f  mean RCI=%.1f  mean NLI=%.3f  mean W33=%.1f"
                 % (g, soil[0][m].mean(), RCI[m].mean(), NLI[m].mean(), soil[5][m].mean()))

lines.append("")
lines.append("Means grouped by W33 (moisture) terciles:")
qw = np.quantile(soil[5], [0.333, 0.667])
grpw = np.digitize(soil[5], qw)
for g in range(3):
    m = grpw == g
    lines.append("Water group %d  mean W33=%.1f  mean RCI=%.1f  mean NLI=%.3f  mean SAND=%.1f"
                 % (g, soil[5][m].mean(), RCI[m].mean(), NLI[m].mean(), soil[0][m].mean()))

report = "\n".join(lines)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
