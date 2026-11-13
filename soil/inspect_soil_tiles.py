# -*- coding: utf-8 -*-
"""检查 OpenLandMap 6 个分块的元数据、配准、取值与缺失情况"""
import glob
import os
import numpy as np
import rasterio
from rasterio.windows import Window
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

SRC = os.path.join(DATA_DIR, "gee_raw")
files = sorted(glob.glob(os.path.join(SRC, "*.tif")))

print("File count:", len(files))
for f in files:
    with rasterio.open(f) as ds:
        print("=" * 90)
        print("File:", os.path.basename(f))
        print("  shape (height, width):", (ds.height, ds.width), "band count:", ds.count,
              "dtype:", ds.dtypes[0], "nodata:", ds.nodata)
        print("  crs:", ds.crs)
        print("  transform:", ds.transform)
        print("  bounds:", ds.bounds)
        print("  band names:", list(ds.descriptions))
        names = list(ds.descriptions)
        # Per-band statistics (block reads to avoid memory issues).
        for bi in range(1, ds.count + 1):
            mn, mx, sm, cnt = None, None, 0.0, 0
            with rasterio.open(f) as d2:
                for _, w in d2.block_windows(bi):
                    a = d2.read(bi, window=w)
                    valid = ~np.isnan(a) if np.issubdtype(a.dtype, np.floating) else np.ones(a.shape, bool)
                    if valid.sum() > 0:
                        v = a[valid]
                        mn = v.min() if mn is None else min(mn, v.min())
                        mx = v.max() if mx is None else max(mx, v.max())
                        sm += float(v.sum())
                        cnt += int(v.size)
            name = names[bi - 1] if names[bi - 1] else ("band%d" % bi)
            if cnt == 0:
                print("    %-6s no valid data (all null)" % name)
            else:
                mean = sm / cnt
                print("    %-6s min=%-12.4f max=%-12.4f mean=%-10.4f valid pixels=%d" % (name, mn, mx, mean, cnt))
