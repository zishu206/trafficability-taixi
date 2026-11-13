# -*- coding: utf-8 -*-
"""
验证台湾本岛（经度>120.05，排除西半全null海域块）内土壤缺失像元是否对应水体/湿地：
对比土壤有效 vs 缺失像元的 NDW(索引7)、NWDI(索引8)、NLULC(索引12) 分布。
分块读取。
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

TIF = os.path.join(DATA_DIR, "features_13ch.tif")
SOIL = os.path.join(DATA_DIR, "soil_physical_raw_30m.tif")

with rasterio.open(TIF) as src:
    H, W = src.height, src.width
    trans = src.transform
    # 本岛经度>120.05 对应的列
    lon_cut = 120.05
    col_cut = int(np.ceil((lon_cut - trans.c) / trans.a))
    print("Island boundary column:", col_cut, " corresponding longitude:", trans.c + col_cut * trans.a)

BLOCK = 512
# 收集有效/缺失像元的 NDW、NWDI（用分位数聚合，避免存全量）
val_ndw, val_nwdi, miss_ndw, miss_nwdi = [], [], [], []
val_lulc, miss_lulc = [], []
n_val = 0; n_miss = 0
for r0 in range(0, H, BLOCK):
    r1 = min(r0 + BLOCK, H); bh = r1 - r0
    with rasterio.open(TIF) as src:
        d = src.read(window=Window(0, r0, W, bh))
    with rasterio.open(SOIL) as sl:
        sv = ~np.isnan(sl.read(1, window=Window(0, r0, W, bh)))
    refv = ~np.isnan(d[0])
    miss = refv & ~sv
    # 只看本岛列
    isl = np.zeros_like(refv); isl[:, col_cut:] = True
    vm = sv & isl
    mm = miss & isl
    n_val += int(vm.sum()); n_miss += int(mm.sum())
    val_ndw.append(d[7][vm]); val_nwdi.append(d[8][vm])
    miss_ndw.append(d[7][mm]); miss_nwdi.append(d[8][mm])
    val_lulc.append(d[12][vm]); miss_lulc.append(d[12][mm])

val_ndw = np.concatenate(val_ndw); val_nwdi = np.concatenate(val_nwdi)
miss_ndw = np.concatenate(miss_ndw); miss_nwdi = np.concatenate(miss_nwdi)
val_lulc = np.concatenate(val_lulc); miss_lulc = np.concatenate(miss_lulc)

print("\nIsland soil valid pixels: %d, missing pixels: %d" % (n_val, n_miss))
def qstat(name, a):
    print("  %-12s mean=%.3f  P50=%.3f  P90=%.3f" % (name, a.mean(), np.percentile(a, 50), np.percentile(a, 90)))
print("\n[Soil valid pixels]")
qstat("NDW", val_ndw); qstat("NWDI", val_nwdi)
print("[Soil missing pixels]")
qstat("NDW", miss_ndw); qstat("NWDI", miss_nwdi)

print("\nNLULC class distribution (share %):")
def lulc_dist(a):
    u, c = np.unique(a, return_counts=True)
    return {int(k): round(100 * v / c.sum(), 2) for k, v in zip(u, c)}
print("  Valid pixels:", lulc_dist(val_lulc))
print("  Missing pixels:", lulc_dist(miss_lulc))
