# -*- coding: utf-8 -*-
"""
探索性试算：在真实土壤数据上插值力学参数、计算 CI 与抗剪强度，确认单位与合理范围。
分块读取 + 规则采样，不写正式产物。
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

SOIL = os.path.join(DATA_DIR, "soil_physical_raw_30m.tif")

# 6 个参考类型：质心(SAND,SILT,CLAY)，c1,c2；c(kPa),phi(deg),n,kc(kN),ktheta(kN)
REF = np.array([
    [92, 3, 5],      # sand
    [65, 25, 10],    # sandy loam
    [20, 70, 10],    # silt
    [40, 40, 20],    # loam
    [30, 35, 35],    # clay loam
    [20, 20, 60],    # clay
], dtype=np.float64)
P = {
 "c1":   np.array([3.987, 8.749, 10.225, 10.998, 12.321, 13.816]),
 "c2":   np.array([0.815, -1.195, -1.565, -1.848, -2.044, -5.583]),
 "c":    np.array([1.0, 1.7, 3.0, 3.0, 3.5, 4.14]),
 "phi":  np.array([33.0, 29.0, 27.0, 26.0, 20.0, 13.0]),
 "n":    np.array([1.1, 0.7, 0.8, 1.01, 0.73, 0.5]),
 "kc":   np.array([0.9, 5.27, 3.0, 0.06, 41.6, 13.19]),
 "kth":  np.array([1523.4, 1515.04, 1200.0, 5880.0, 2471.0, 692.15]),
}

def interp(texture):
    """texture: (N,3) 砂粉黏；返回各参数 (N,)"""
    # 质地空间欧氏距离
    d = np.sqrt(((texture[:, None, :] - REF[None, :, :]) ** 2).sum(axis=2))  # (N,6)
    w = 1.0 / (d + 1e-6) ** 2
    w = w / w.sum(axis=1, keepdims=True)
    out = {k: (w * v[None, :]).sum(axis=1) for k, v in P.items()}
    return out

with rasterio.open(SOIL) as src:
    H, W = src.height, src.width
    samples = []
    BLOCK = 1024
    for r0 in range(0, H, BLOCK):
        r1 = min(r0 + BLOCK, H); bh = r1 - r0
        d = src.read(window=Window(0, r0, W, bh))
        valid = ~np.isnan(d[0])
        if valid.sum() == 0:
            continue
        # 每块规则采样约 1500 个
        idx = np.where(valid.reshape(-1))[0]
        if len(idx) > 1500:
            idx = idx[np.linspace(0, len(idx) - 1, 1500).round().astype(int)]
        sand = d[0].reshape(-1)[idx]; silt = d[1].reshape(-1)[idx]; clay = d[2].reshape(-1)[idx]
        bd = d[3].reshape(-1)[idx]; soc = d[4].reshape(-1)[idx]; w33 = d[5].reshape(-1)[idx]
        samples.append(np.stack([sand, silt, clay, bd, soc, w33], axis=1))

S = np.concatenate(samples)
print("Sampled pixels:", len(S))
sand, silt, clay, bd, soc, w33 = [S[:, i] for i in range(6)]
print("\nInput ranges: SAND %.0f-%.0f  BD %.0f-%.0f  SOC %.0f-%.0f  W33 %.0f-%.0f"
      % (sand.min(), sand.max(), bd.min(), bd.max(), soc.min(), soc.max(), w33.min(), w33.max()))

ip = interp(S[:, :3])
print("\nInterpolated parameter ranges:")
for k in ["c1", "c2", "c", "phi", "n", "kc", "kth"]:
    print("  %-4s min=%8.3f max=%8.3f mean=%8.3f" % (k, ip[k].min(), ip[k].max(), ip[k].mean()))

# Gravimetric moisture MC: volumetric moisture W33 (%) x dry bulk density (g/cm3)
rho_d = bd / 1000.0
MC_pct = w33 * rho_d          # percentage convention (e.g. 30 means 30%)
MC_frac = MC_pct / 100.0      # fraction convention (0.3)
CI_pct = np.exp(ip["c1"] + ip["c2"] * np.log(MC_pct))
CI_frac = np.exp(ip["c1"] + ip["c2"] * np.log(MC_frac))
print("\nCI (MC percentage convention): min=%.1f max=%.1f median=%.1f" % (CI_pct.min(), CI_pct.max(), np.median(CI_pct)))
print("CI (MC fraction convention): min=%.2f max=%.2f median=%.2f" % (CI_frac.min(), CI_frac.max(), np.median(CI_frac)))

# Shear strength (Mohr-Coulomb, reference ground pressure sigma0 = 138 kPa ~ 20 psi)
sigma0 = 138.0
tau = ip["c"] + sigma0 * np.tan(np.deg2rad(ip["phi"]))
print("\nMohr-Coulomb shear strength tau(kPa, sigma=138): min=%.1f max=%.1f median=%.1f" % (tau.min(), tau.max(), np.median(tau)))

# Organic-carbon softening factor (0.75 to 1.0)
soc_f = np.clip(1.0 - 0.0025 * soc, 0.75, 1.0)
print("Organic-carbon softening factor: min=%.3f max=%.3f" % (soc_f.min(), soc_f.max()))

# Remolding factor: fine-grained soil loses strength after repeated traffic (0.6 to 1.0); high sand content approaches 1.
rem_f = np.clip(1.0 - 0.4 * (clay / 100.0), 0.6, 1.0)
RCI = CI_pct * soc_f * rem_f
print("\n=== RCI (final mechanistic strength, psi) ===")
print("min=%.2f max=%.2f mean=%.2f" % (RCI.min(), RCI.max(), RCI.mean()))
qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("Quantiles: " + " ".join("P%d=%.1f" % (q, np.percentile(RCI, q)) for q in qs))

# Three-class shares under engineering thresholds (typical US Army VCI criteria)
for lo, hi in [(25, 50), (30, 60), (35, 70)]:
    y = np.where(RCI < lo, 0, np.where(RCI < hi, 1, 2))
    u, c = np.unique(y, return_counts=True)
    dist = {int(k): round(100 * v / len(y), 1) for k, v in zip(u, c)}
    print("Thresholds %d/%d psi three-class shares (low/medium/high):" % (lo, hi), dist)

# Jenks natural breaks (study-area adaptive)
import mapclassify as mc
j = mc.JenksCaspall(RCI, k=3)
print("RCI Jenks three-class upper bounds:", np.round(j.bins, 2))
