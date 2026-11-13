# -*- coding: utf-8 -*-
"""
Bekker-Wong 机理标签构建（任务14核心）：
输入 soil_physical_raw_30m.tif（7波段：SAND,SILT,CLAY,BD,SOC,W33,TEX）
流程：
  1) 质地反距离加权插值 c1,c2,c,phi,n,kc,kth（参数取自 Wong/Yong 与圆锥指数系数表）
  2) 重量含水率 MC = W33(体积%) × 干密度(BD/1000)
  3) CI = exp(c1 + c2·ln MC)（psi）
  4) 有机碳软化 + 重塑折减 -> RCI
  5) 全量 Jenks 自然断点分三级（0/1/2），并报告工程区间占比
输出：mech_RCI_30m.tif（连续）、mech_label_3class.tif（三级）
分块读取，两遍法控制内存。
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

SOIL = os.path.join(DATA_DIR, "soil_physical_raw_30m.tif")
OUT_RCI = os.path.join(DATA_DIR, "mech_RCI_30m.tif")
OUT_LBL = os.path.join(DATA_DIR, "mech_label_3class.tif")

# 6 参考类型质心（SAND,SILT,CLAY）
REF = np.array([
    [92, 3, 5], [65, 25, 10], [20, 70, 10],
    [40, 40, 20], [30, 35, 35], [20, 20, 60]], dtype=np.float64)
P = {
 "c1":  np.array([3.987, 8.749, 10.225, 10.998, 12.321, 13.816]),
 "c2":  np.array([0.815, -1.195, -1.565, -1.848, -2.044, -5.583]),
 "c":   np.array([1.0, 1.7, 3.0, 3.0, 3.5, 4.14]),
 "phi": np.array([33.0, 29.0, 27.0, 26.0, 20.0, 13.0]),
 "n":   np.array([1.1, 0.7, 0.8, 1.01, 0.73, 0.5]),
 "kc":  np.array([0.9, 5.27, 3.0, 0.06, 41.6, 13.19]),
 "kth": np.array([1523.4, 1515.04, 1200.0, 5880.0, 2471.0, 692.15]),
}
BLOCK = 512

def compute_block(d):
    """d: (7,bh,bw)，返回 RCI 2D（无效像元 nan）"""
    shape2 = d.shape[1:]
    valid = ~np.isnan(d[0])
    rci = np.full(shape2, np.nan, dtype=np.float32)
    if valid.sum() == 0:
        return rci
    sand = d[0][valid]; silt = d[1][valid]; clay = d[2][valid]
    bd = d[3][valid]; soc = d[4][valid]; w33 = d[5][valid]
    tex = np.stack([sand, silt, clay], axis=1)
    dist = np.sqrt(((tex[:, None, :] - REF[None, :, :]) ** 2).sum(axis=2))
    w = 1.0 / (dist + 1e-6) ** 2
    w = w / w.sum(axis=1, keepdims=True)
    c1 = (w * P["c1"][None, :]).sum(axis=1)
    c2 = (w * P["c2"][None, :]).sum(axis=1)
    # 重量含水率（%），保护下界
    mc_pct = np.clip(w33 * (bd / 1000.0), 1.0, None)
    ci = np.exp(np.clip(c1 + c2 * np.log(mc_pct), -20, 20))
    soc_f = np.clip(1.0 - 0.0025 * soc, 0.75, 1.0)
    rem_f = np.clip(1.0 - 0.4 * (clay / 100.0), 0.6, 1.0)
    rci[valid] = (ci * soc_f * rem_f).astype(np.float32)
    return rci

with rasterio.open(SOIL) as src:
    H, Wd = src.height, src.width
    trans, crs = src.transform, src.crs

# ---- 第一遍：分块计算，收集有效 RCI ----
print("First pass: computing RCI ...")
rci_vals = []
for r0 in range(0, H, BLOCK):
    r1 = min(r0 + BLOCK, H); bh = r1 - r0
    with rasterio.open(SOIL) as src:
        d = src.read(window=Window(0, r0, Wd, bh))
    rci = compute_block(d)
    v = rci[~np.isnan(rci)]
    if len(v):
        rci_vals.append(v)
rci_all = np.concatenate(rci_vals).astype(np.float64)
print("Valid pixels:", len(rci_all))
print("RCI min=%.2f max=%.2f mean=%.2f median=%.2f"
      % (rci_all.min(), rci_all.max(), rci_all.mean(), np.median(rci_all)))

# Jenks three-class breaks
j = mc.JenksCaspall(rci_all, k=3)
breaks = np.asarray(j.bins, dtype=np.float64)
internal = breaks[:2]
print("Jenks three-class upper bounds:", np.round(breaks, 3))

# Engineering intervals (30-75 / 75-125 / >125) as a reference for shares.
eng = np.digitize(rci_all, [75.0, 125.0])
for i, nm in enumerate(["<75 (soft)", "75-125 (medium)", ">125 (hard)"]):
    print("  Engineering interval %s: %.2f%%" % (nm, 100 * (eng == i).mean()))

# ---- Second pass: recompute and write RCI and the label ----
print("\nSecond pass: writing RCI and the three-class label ...")
prof_rci = dict(driver="GTiff", height=H, width=Wd, count=1, dtype="float32",
                crs=crs, transform=trans, nodata=None, compress="lzw", tiled=True)
prof_lbl = dict(prof_rci)
counts = np.zeros(3, dtype=np.int64)
with rasterio.open(OUT_RCI, "w", **prof_rci) as fo, rasterio.open(OUT_LBL, "w", **prof_lbl) as fl:
    fo.set_band_description(1, "RCI_psi")
    fl.set_band_description(1, "mech_label_012")
    for r0 in range(0, H, BLOCK):
        r1 = min(r0 + BLOCK, H); bh = r1 - r0
        with rasterio.open(SOIL) as src:
            d = src.read(window=Window(0, r0, Wd, bh))
        rci = compute_block(d)
        valid = ~np.isnan(rci)
        lbl = np.full(rci.shape, np.nan, dtype=np.float32)
        y = np.digitize(rci[valid], internal)
        lbl[valid] = y
        for i in range(3):
            counts[i] += int((y == i).sum())
        fo.write(rci, 1, window=Window(0, r0, Wd, bh))
        fl.write(lbl, 1, window=Window(0, r0, Wd, bh))

print("\nMechanistic label three-class shares (valid pixels %d):" % counts.sum())
for i, nm in enumerate(["Low", "Medium", "High"]):
    print("  %s: %d  %.2f%%" % (nm, counts[i], 100 * counts[i] / counts.sum()))
print("\nWritten:\n ", OUT_RCI, "\n ", OUT_LBL)
