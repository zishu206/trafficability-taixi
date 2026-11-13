# -*- coding: utf-8 -*-
"""
三标签公共域切片准备：
在同一 128 网格上，要求 13 通道 X 与机理标签同时全有效（公共有效像元），
每个切片同时携带 AHP 三级标签（公共域 Jenks）与机理三级标签，
保证同 X、同空间划分、仅标签构建逻辑不同。
输出 common_domain.npz。
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import mapclassify as mc
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

X_TIF = os.path.join(DATA_DIR, "features_13ch.tif")
MECH_TIF = os.path.join(DATA_DIR, "mech_label_3class.tif")
OUT = os.path.join(DATA_DIR, "common_domain.npz")
WEIGHTS = np.array([
    0.2335, 0.1313, 0.0758, 0.0426,
    0.0529, 0.0630, 0.0153, 0.0257,
    0.1811, 0.0906,
    0.0262, 0.0144, 0.0476], dtype=np.float32)
P = 128

os.makedirs(os.path.dirname(OUT), exist_ok=True)

with rasterio.open(X_TIF) as xf, rasterio.open(MECH_TIF) as mf:
    H, W = xf.height, xf.width
    coords = []
    for r in range(0, H - P + 1, P):
        for c in range(0, W - P + 1, P):
            win = Window(c, r, P, P)
            x = xf.read(window=win)
            m = mf.read(1, window=win)
            if (x.shape == (13, P, P) and not np.isnan(x).any()
                    and m.shape == (P, P) and not np.isnan(m).any()):
                coords.append((r, c))
    print("Fully valid common-domain grid count:", len(coords))

    X, Ymech, S, GR, GC = [], [], [], [], []
    for r, c in coords:
        win = Window(c, r, P, P)
        x = xf.read(window=win).astype(np.float32)
        m = mf.read(1, window=win).astype(np.int64)
        X.append(x)
        Ymech.append(m)
        S.append(np.tensordot(WEIGHTS, x, axes=(0, 0)))
        GR.append(r // P); GC.append(c // P)

X = np.stack(X); Ymech = np.stack(Ymech); S = np.stack(S)
GR = np.array(GR); GC = np.array(GC)

# AHP 公共域 Jenks 三级
jahp = mc.JenksCaspall(S.reshape(-1), k=3)
ahp_breaks = np.asarray(jahp.bins, dtype=np.float64)
Yahp = np.digitize(S, ahp_breaks[:2]).astype(np.int64)
print("AHP common-domain Jenks upper bounds:", np.round(ahp_breaks, 4))
for i, nm in enumerate(["Low", "Medium", "High"]):
    print("  AHP %s share: %.2f%%" % (nm, 100 * (Yahp == i).mean()))
print("Mechanistic label three-class shares:")
for i, nm in enumerate(["Low", "Medium", "High"]):
    print("  mech %s share: %.2f%%" % (nm, 100 * (Ymech == i).mean()))

np.savez_compressed(OUT, X=X, Y_ahp=Yahp, Y_mech=Ymech, S=S,
                    GR=GR, GC=GC, ahp_breaks=ahp_breaks)
print("SAVED:", OUT)
