# -*- coding: utf-8 -*-
"""
绘制 NLI、W33、SAND、RCI、AHP综合S 的空间分布（降采样），厘清空间格局。
输出 fig_spatial_maps.png。
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

plt.rcParams["axes.unicode_minus"] = False

X_TIF = os.path.join(DATA_DIR, "features_13ch.tif")
SOIL = os.path.join(DATA_DIR, "soil_physical_raw_30m.tif")
RCI_TIF = os.path.join(DATA_DIR, "mech_RCI_30m.tif")
OUT = os.path.join(OUT_DIR, "fig_spatial_maps.png")

SCALE = 24  # 降采样倍数

def read_small(path, band, scale=SCALE):
    with rasterio.open(path) as src:
        h = max(1, src.height // scale); w = max(1, src.width // scale)
        if band is None:
            arr = src.read(out_shape=(src.count, h, w),
                           resampling=rasterio.enums.Resampling.average)
        else:
            arr = src.read(band, out_shape=(h, w),
                           resampling=rasterio.enums.Resampling.average)
        bounds = src.bounds
    return arr, bounds

nli, b = read_small(X_TIF, 1)
sand, _ = read_small(SOIL, 1)
w33, _ = read_small(SOIL, 6)
rci, _ = read_small(RCI_TIF, 1)
extent = [b.left, b.right, b.bottom, b.top]

# AHP 综合 S（降采样 13 通道后加权）
with rasterio.open(X_TIF) as src:
    h = max(1, src.height // SCALE); w = max(1, src.width // SCALE)
    xs = src.read(out_shape=(13, h, w), resampling=rasterio.enums.Resampling.average)
WEIGHTS = np.array([0.2335,0.1313,0.0758,0.0426,0.0529,0.0630,0.0153,0.0257,
                    0.1811,0.0906,0.0262,0.0144,0.0476])
S = np.tensordot(WEIGHTS, xs, axes=(0, 0))

panels = [
    (nli, "NLI lithology membership (AHP channel 0)", "viridis"),
    (w33, "W33 soil moisture / %", "Blues"),
    (sand, "SAND sand content / %", "YlOrBr"),
    (rci, "RCI mechanistic resistance / psi", "magma"),
    (S, "S composite AHP score", "viridis"),
]
fig, axes = plt.subplots(1, 5, figsize=(18, 6.2))
for ax, (arr, title, cmap) in zip(axes, panels):
    a = np.where(np.isfinite(arr), arr, np.nan)
    if "RCI" in title:
        a = np.log10(np.where(a > 0, a, np.nan))  # RCI is right-skewed; show on log scale
        title = title + " (log10)"
    im = ax.imshow(a, extent=extent, origin="upper", cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude / °E"); ax.set_ylabel("Latitude / °N")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Spatial distribution of NLI, soil parameters, RCI and the composite AHP score "
             "over the Taixi Plain", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=180)
print("saved:", OUT)
