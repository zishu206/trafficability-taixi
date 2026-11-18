import numpy as np
import rasterio

raster_path = r'C:\Users\莫名的wsr\Desktop\论文台西\深度学习承载力\CNN与transform\03_数据与模型\栅格数据\mech_label_3class.tif'
npz_path = r'C:\Users\莫名的wsr\Desktop\论文台西\深度学习承载力\CNN与transform\03_数据与模型\模型权重与训练产物\common_domain.npz'
points = [
    ('原斗', 120.42756413038035, 23.89540983059925),
    ('明德', 120.19918986641926, 23.652796199584998),
    ('东源', 120.61108097600629, 23.853877484658256),
]
z = np.load(npz_path)
gr_gc = set(zip(z['GR'].tolist(), z['GC'].tolist()))
with rasterio.open(raster_path) as src:
    print('transform', src.transform)
    for name, lon, lat in points:
        row, col = src.index(lon, lat)
        gr, gc = row // 128, col // 128
        rr, cc = row % 128, col % 128
        print(name, 'pixel', row, col, 'tile', gr, gc, 'within', rr, cc, 'in common', (gr, gc) in gr_gc)
        if (gr, gc) in gr_gc:
            idx = next(i for i, (a, b) in enumerate(zip(z['GR'], z['GC'])) if a == gr and b == gc)
            print(' npz index', idx, 'ahp', z['Y_ahp'][idx, rr, cc], 'mech', z['Y_mech'][idx, rr, cc])
