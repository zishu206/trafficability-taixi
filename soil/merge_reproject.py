# -*- coding: utf-8 -*-
"""
合并 6 个 OpenLandMap 分块（merge 输出已在目标网格上），用现有 tif 掩膜规范区外
输出：soil_physical_raw_30m.tif（7 波段 float32，与现有 tif 像元级配准）
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

SRC_DIR = os.path.join(DATA_DIR, "gee_raw")
REF_TIF = os.path.join(DATA_DIR, "features_13ch.tif")
OUT_TIF = os.path.join(DATA_DIR, "soil_physical_raw_30m.tif")
BAND_NAMES = ["SAND", "SILT", "CLAY", "BD", "SOC", "W33", "TEX"]

files = sorted(glob.glob(SRC_DIR + "/*.tif"))
print("Tile count:", len(files))
handles = [rasterio.open(f) for f in files]
mosaic, mos_trans = merge(handles, resampling=Resampling.nearest)
for h in handles:
    h.close()
print("Merged shape:", mosaic.shape)

with rasterio.open(REF_TIF) as ref:
    H, W = ref.height, ref.width
    dst_trans = ref.transform
    ref_mask = ~np.isnan(ref.read(1))

assert mosaic.shape == (len(BAND_NAMES), H, W), "shape mismatch"
assert tuple(round(x, 6) for x in mos_trans[:6]) == tuple(round(x, 6) for x in dst_trans[:6]), "transform mismatch"
print("Grid and transform match the existing tif; no resampling needed")

# 用现有 tif 掩膜把区外统一设为 nan
for bi in range(len(BAND_NAMES)):
    mosaic[bi][~ref_mask] = np.nan

profile = {
    "driver": "GTiff", "height": H, "width": W, "count": len(BAND_NAMES),
    "dtype": "float32", "crs": "EPSG:4326", "transform": dst_trans,
    "nodata": None, "compress": "lzw", "tiled": True,
}
with rasterio.open(OUT_TIF, "w", **profile) as out:
    for bi in range(len(BAND_NAMES)):
        out.write(mosaic[bi], bi + 1)
        out.set_band_description(bi + 1, BAND_NAMES[bi])
print("Written:", OUT_TIF)

soil_valid = ~np.isnan(mosaic[0])
print("\n=== Registration check ===")
print("Valid in existing tif:", int(ref_mask.sum()), "valid soil:", int(soil_valid.sum()), "masks identical:", bool((ref_mask == soil_valid).all()))

print("\n=== Island-wide statistics per band (land only) ===")
for bi, nm in enumerate(BAND_NAMES):
    v = mosaic[bi][soil_valid]
    print("  %-6s min=%-10.2f max=%-10.2f mean=%-10.3f" % (nm, v.min(), v.max(), v.mean()))
s = mosaic[0][soil_valid] + mosaic[1][soil_valid] + mosaic[2][soil_valid]
print("\nSand+silt+clay sum: min=%.2f max=%.2f mean=%.3f" % (s.min(), s.max(), s.mean()))
