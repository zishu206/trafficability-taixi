# -*- coding: utf-8 -*-
"""
诊断土壤掩膜与现有tif掩膜的差异：
统计 ref 有效但土壤 nan 的缺失像元（missing）的空间分布，判断是海域还是漏导陆地。
分块读取，避免全量内存；输出按经度分桶统计 + 降采样缺失分布图。
"""
import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

REF_TIF = os.path.join(DATA_DIR, "features_13ch.tif")
SOIL_TIF = os.path.join(DATA_DIR, "soil_physical_raw_30m.tif")
OUT_PNG = os.path.join(OUT_DIR, "mask_diagnosis.png")

with rasterio.open(REF_TIF) as ref:
    H, W = ref.height, ref.width
    trans = ref.transform
    with rasterio.open(SOIL_TIF) as sl:
        assert sl.height == H and sl.width == W

# 经度分桶（每 0.1 度）
lon_edges = np.arange(119.3, 121.05, 0.1)
ref_bucket = np.zeros(len(lon_edges) - 1, dtype=np.int64)
miss_bucket = np.zeros(len(lon_edges) - 1, dtype=np.int64)

# 降采样概览（南北约 500 像素）
out_h = 500
out_w = max(1, int(round(W * out_h / H)))
ref_small = np.zeros((out_h, out_w), dtype=np.float32)
miss_small = np.zeros((out_h, out_w), dtype=np.float32)
cnt_small = np.zeros((out_h, out_w), dtype=np.float32)

BLOCK = 512
total_ref = 0
total_miss = 0
for r0 in range(0, H, BLOCK):
    r1 = min(r0 + BLOCK, H)
    bh = r1 - r0
    with rasterio.open(REF_TIF) as ref:
        ref_v = ~np.isnan(ref.read(1, window=rasterio.windows.Window(0, r0, W, bh)))
    with rasterio.open(SOIL_TIF) as sl:
        soil_v = ~np.isnan(sl.read(1, window=rasterio.windows.Window(0, r0, W, bh)))
    missing = ref_v & ~soil_v
    total_ref += int(ref_v.sum())
    total_miss += int(missing.sum())

    # 每列经度
    cols = np.arange(W)
    lons = trans.c + (cols + 0.5) * trans.a
    bi = np.digitize(lons, lon_edges) - 1
    for k in range(len(lon_edges) - 1):
        m = bi == k
        ref_bucket[k] += int(ref_v[:, m].sum())
        miss_bucket[k] += int(missing[:, m].sum())

    # 降采样累加
    rs0 = int(round(r0 / H * out_h))
    rs1 = max(rs0 + 1, int(round(r1 / H * out_h)))
    for rr in range(rs0, min(rs1, out_h)):
        a = int(round((rr / out_h) * H)) - r0
        b = int(round(((rr + 1) / out_h) * H)) - r0
        a = max(0, min(a, bh)); b = max(a + 1, min(b, bh))
        for cc in range(out_w):
            ca = int(round((cc / out_w) * W)); cb = max(ca + 1, int(round(((cc + 1) / out_w) * W)))
            cb = min(cb, W)
            ref_small[rr, cc] += ref_v[a:b, ca:cb].sum()
            miss_small[rr, cc] += missing[a:b, ca:cb].sum()
            cnt_small[rr, cc] += (b - a) * (cb - ca)

print("Valid in current tif:", total_ref, " missing (soil nan):", total_miss,
      " missing share: %.2f%%" % (100 * total_miss / total_ref))
print("\nBuckets by longitude (center lon / valid ref / missing / missing rate %):")
for k in range(len(lon_edges) - 1):
    ctr = 0.5 * (lon_edges[k] + lon_edges[k + 1])
    rate = 100 * miss_bucket[k] / ref_bucket[k] if ref_bucket[k] else 0
    print("  %5.2f E  %10d  %10d  %6.2f%%" % (ctr, ref_bucket[k], miss_bucket[k], rate))

# 缺失像元的纬度范围（分块再扫一次，按纬度分桶）
lat_edges = np.arange(21.9, 25.0, 0.25)
miss_lat = np.zeros(len(lat_edges) - 1, dtype=np.int64)
for r0 in range(0, H, BLOCK):
    r1 = min(r0 + BLOCK, H); bh = r1 - r0
    with rasterio.open(REF_TIF) as ref:
        ref_v = ~np.isnan(ref.read(1, window=rasterio.windows.Window(0, r0, W, bh)))
    with rasterio.open(SOIL_TIF) as sl:
        soil_v = ~np.isnan(sl.read(1, window=rasterio.windows.Window(0, r0, W, bh)))
    missing = ref_v & ~soil_v
    rows = np.arange(r0, r1)
    lats = trans.f + (rows + 0.5) * trans.e
    bi = np.digitize(lats, lat_edges) - 1
    for k in range(len(lat_edges) - 1):
        miss_lat[k] += int(missing[bi == k, :].sum())
print("\nMissing pixels bucketed by latitude (center lat / missing count):")
for k in range(len(lat_edges) - 1):
    ctr = 0.5 * (lat_edges[k] + lat_edges[k + 1])
    print("  %5.2f N  %10d" % (ctr, miss_lat[k]))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 7))
with np.errstate(invalid="ignore", divide="ignore"):
    ref_density = ref_small / cnt_small
    miss_density = miss_small / cnt_small
ext = [trans.c, trans.c + W * trans.a, trans.f + H * trans.e, trans.f]
im0 = axes[0].imshow(ref_density, extent=ext, cmap="Greys", aspect="auto", vmin=0, vmax=1)
axes[0].set_title("Valid pixel density in current tif")
im1 = axes[1].imshow(miss_density, extent=ext, cmap="Reds", aspect="auto", vmin=0, vmax=1)
axes[1].set_title("Soil-missing pixel density (red = missing)")
for ax in axes:
    ax.set_xlabel("Longitude E"); ax.set_ylabel("Latitude N")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print("\nMissing-distribution figure saved:", OUT_PNG)
