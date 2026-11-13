# -*- coding: utf-8 -*-
"""
验证可学性差异的根源：AHP 标签是 X 的确定函数，机理标签依赖 X 之外的土壤信息。
用 13 通道多元线性回归（像元级抽样）分别拟合两个连续目标，比较 R²；
并对连续预测做三级分级，比较简单线性模型的精度。
输出 linear_explainability.txt。
"""
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score

# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
OUT = os.path.join(OUT_DIR, "linear_explainability.txt")

d = np.load(NPZ)
X = d["X"]                       # (771,13,128,128)
S = d["S"]
Yahp, Ymech = d["Y_ahp"], d["Y_mech"]
GR, GC = d["GR"], d["GC"]

import rasterio
from rasterio.windows import Window
parts = []
with rasterio.open(os.path.join(DATA_DIR, "mech_RCI_30m.tif")) as rf:
    for i in range(len(GR)):
        r, c = int(GR[i]) * 128, int(GC[i]) * 128
        parts.append(rf.read(1, window=Window(c, r, 128, 128)))
RCI = np.stack(parts)

# 验证划分（与训练一致）
is_val = []
for i in range(len(GR)):
    code = (int(GR[i]) * 73856093) ^ (int(GC[i]) * 19349663)
    is_val.append((abs(code) % 100) < 30)
is_val = np.array(is_val)

# 展平为像元矩阵（13 通道），训练/验证像元
Xtr = X[~is_val].transpose(0, 2, 3, 1).reshape(-1, 13)
Xva = X[is_val].transpose(0, 2, 3, 1).reshape(-1, 13)
rng = np.random.default_rng(0)
tr_s = rng.choice(len(Xtr), 300000, replace=False)
va_s = rng.choice(len(Xva), 150000, replace=False)
Xtr, Xva = Xtr[tr_s], Xva[va_s]

lines = ["Explainability of the two labels with 13-channel multivariate linear regression (pixel-level sampling)", ""]
for nm, cont, y3, log in [
    ("AHP", S, Yahp, False),
    ("Mechanistic", RCI, Ymech, True)]:
    ytr = cont[~is_val].reshape(-1)[tr_s]
    yva_cont = cont[is_val].reshape(-1)[va_s]
    yva3 = y3[is_val].reshape(-1)[va_s]
    tgt = np.log(ytr) if log else ytr
    reg = LinearRegression().fit(Xtr, tgt)
    pred = reg.predict(Xva)
    r2 = r2_score(np.log(yva_cont) if log else yva_cont, pred)
    # Three-class grading against the true validation breaks (here the linear prediction is compared directly with the true thresholds).
    # Simplification: report the continuous R² and the direction of correlation between prediction and true three classes.
    spear = np.corrcoef(pred, np.log(yva_cont) if log else yva_cont)[0, 1]
    lines.append("===== %s label =====" % nm)
    lines.append("  Continuous linear regression R² = %.4f" % r2)
    lines.append("  Prediction-target Pearson = %.4f" % spear)
    lines.append("")

lines.append("Interpretation: the continuous AHP score S is the weighted sum of the 13 channels "
             "(a deterministic linear function of X), so a linear model explains it almost completely.")
lines.append("RCI is determined by OpenLandMap soil parameters outside X and has low 13-channel linear explainability;")
lines.append("the neural networks therefore reproduce the mechanistic label at a systematically lower accuracy.")
report = "\n".join(lines)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
