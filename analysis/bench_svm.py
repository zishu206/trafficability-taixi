# -*- coding: utf-8 -*-
"""基准：RBF SVC vs LinearSVC 的训练/预测耗时与精度（ahp 标签，抽样）。"""
import os
import time
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
d = np.load(NPZ)
X, GR, GC = d["X"], d["GR"], d["GC"]
is_val = np.array([(abs((int(GR[i]) * 73856093) ^ (int(GC[i]) * 19349663)) % 100) < 30
                   for i in range(len(GR))])
tr, va = np.where(~is_val)[0], np.where(is_val)[0]


def flat(idx):
    x = np.transpose(X[idx], (0, 2, 3, 1)).reshape(-1, 13)
    y = d["Y_ahp"][idx].reshape(-1)
    return x, y


Xtr, Ytr = flat(tr)
Xva, Yva = flat(va)
rng = np.random.default_rng(42)
# 训练抽样 21000
sidx = []
for c in range(3):
    ci = np.where(Ytr == c)[0]
    sidx.append(rng.choice(ci, 7000, replace=False))
sidx = np.concatenate(sidx)
# 验证基准 60000
v60 = rng.choice(Yva.size, 60000, replace=False)

# RBF
t0 = time.time()
rbf = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=2000).fit(Xtr[sidx], Ytr[sidx])
t_tr = time.time() - t0
t0 = time.time()
p = rbf.predict(Xva[v60])
t_pr = time.time() - t0
print("RBF     train %.1fs predict 60k %.1fs (support vectors %d) OA %.4f macroF1 %.4f"
      % (t_tr, t_pr, rbf.support_vectors_.shape[0], accuracy_score(Yva[v60], p),
         f1_score(Yva[v60], p, average="macro")))

# LinearSVC
t0 = time.time()
lin = LinearSVC(C=1.0, dual=False, max_iter=5000).fit(Xtr[sidx], Ytr[sidx])
t_tr2 = time.time() - t0
t0 = time.time()
p2 = lin.predict(Xva[v60])
t_pr2 = time.time() - t0
print("Linear  train %.1fs predict 60k %.1fs OA %.4f macroF1 %.4f"
      % (t_tr2, t_pr2, accuracy_score(Yva[v60], p2),
         f1_score(Yva[v60], p2, average="macro")))

# Extrapolated prediction time over the full validation set (3.6864M)
print("Estimated full-validation prediction: RBF about %.1f min; Linear about %.1f s"
      % (t_pr * 3686400 / 60000 / 60, t_pr2 * 3686400 / 60000))
