# -*- coding: utf-8 -*-
"""
标签空间复杂度对比：解释机理标签为何更难学。
- 三级标签边界密度（相邻像元类间差异比例）
- 连续量变异系数、平均局部梯度
输出 label_complexity.txt。
"""
import os
import numpy as np

# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
OUT = os.path.join(OUT_DIR, "label_complexity.txt")

d = np.load(NPZ)
Yahp, Ymech = d["Y_ahp"], d["Y_mech"]
S = d["S"]

import rasterio
from rasterio.windows import Window
GR, GC = d["GR"], d["GC"]
parts = []
with rasterio.open(os.path.join(DATA_DIR, "mech_RCI_30m.tif")) as rf:
    for i in range(len(GR)):
        r, c = int(GR[i]) * 128, int(GC[i]) * 128
        parts.append(rf.read(1, window=Window(c, r, 128, 128)))
RCI = np.stack(parts)


def edge_density(y):
    h = (y[:, :, 1:] != y[:, :, :-1]).mean()
    v = (y[:, 1:, :] != y[:, :-1, :]).mean()
    return float((h + v) / 2)


def cv(x):
    x = x.reshape(-1)
    return float(x.std() / x.mean())


def grad(x):
    gx = np.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
    gy = np.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
    return float((gx + gy) / 2)


lines = ["Label spatial complexity comparison (common domain, 771 tiles)", ""]
lines.append("Three-class edge density (share of between-class adjacent pixels; higher = more fragmented):")
lines.append("  AHP label = %.4f" % edge_density(Yahp))
lines.append("  Mechanistic label = %.4f" % edge_density(Ymech))
lines.append("")
lines.append("Coefficient of variation CV of continuous measures:")
lines.append("  S(AHP) = %.4f" % cv(S))
lines.append("  RCI = %.4f" % cv(RCI))
lines.append("")
lines.append("Mean local gradient of continuous measures (absolute change between adjacent pixels; higher = less smooth):")
lines.append("  S(AHP) = %.5f" % grad(S))
lines.append("  RCI = %.4f" % grad(RCI))
lines.append("")
lines.append("Global spatial autocorrelation (Moran's I, approximated on the tile-grid sampling):")
# Simplified Moran's I: approximate with the dispersion of per-tile means and report within-tile variance.
for nm, x in [("S", S), ("RCI", RCI)]:
    pm = x.reshape(len(x), -1).mean(axis=1)
    lines.append("  %s std of tile means = %.4f (dispersion of tile means)" % (nm, pm.std()))

report = "\n".join(lines)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
