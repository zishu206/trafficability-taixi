# -*- coding: utf-8 -*-
"""
公共域 RF / SVM 像元级分类：与深度模型同一 771 切片、同一空间划分（546/225）。
对 AHP、机理两类标签分别训练随机森林与 RBF-SVM，在完整验证集（225 切片）评估。
训练像元分层抽样（像元级传统方法的标准做法），评估覆盖全部验证像元。
"""
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, cohen_kappa_score,
                             f1_score, accuracy_score)

# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
OUT_TXT = os.path.join(OUT_DIR, "rf_svm_metrics.txt")
SEED = 42
RF_PER_CLASS = 20000
SVM_PER_CLASS = 7000


def split_indices(GR, GC):
    is_val = np.array([(abs((int(GR[i]) * 73856093) ^ (int(GC[i]) * 19349663)) % 100) < 30
                       for i in range(len(GR))])
    return np.where(~is_val)[0], np.where(is_val)[0]


def flatten(tiles_X, tiles_Y):
    # (n,13,128,128)->(n*16384,13); (n,128,128)->(n*16384,)
    n = tiles_X.shape[0]
    x = np.transpose(tiles_X, (0, 2, 3, 1)).reshape(-1, 13)
    y = tiles_Y.reshape(-1)
    return x, y


def stratified_sample(y, per_class, rng):
    idx = []
    for c in range(3):
        ci = np.where(y == c)[0]
        take = min(per_class, len(ci))
        idx.append(rng.choice(ci, take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def chunk_predict(model, X, chunk=100000, tag=""):
    preds = np.empty(X.shape[0], dtype=np.int64)
    nb = (X.shape[0] + chunk - 1) // chunk
    for b, s in enumerate(range(0, X.shape[0], chunk)):
        preds[s:s + chunk] = model.predict(X[s:s + chunk])
        print("  [%s] prediction progress %d/%d (%.0f%%)" % (tag, b + 1, nb, 100 * (b + 1) / nb), flush=True)
    return preds


d = np.load(NPZ)
X, GR, GC = d["X"], d["GR"], d["GC"]
tr_idx, va_idx = split_indices(GR, GC)
print("Training tiles %d, validation tiles %d" % (len(tr_idx), len(va_idx)))

rng = np.random.default_rng(SEED)
lines = ["Common-domain RF/SVM pixel-level classification (validation set, 225 tiles, %d pixels)" % (len(va_idx) * 128 * 128), ""]
results = {}

for tag, ykey in [("ahp", "Y_ahp"), ("mech", "Y_mech")]:
    Y = d[ykey]
    Xtr, Ytr = flatten(X[tr_idx], Y[tr_idx])
    Xva, Yva = flatten(X[va_idx], Y[va_idx])
    print("\n=== Label %s  training pixels %d  validation pixels %d ===" % (tag, Ytr.size, Yva.size))

    # ---- Random forest ----
    sidx = stratified_sample(Ytr, RF_PER_CLASS, rng)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(Xtr[sidx], Ytr[sidx])
    print("RF training %.1f s (%d pixels)" % (time.time() - t0, len(sidx)))
    t0 = time.time()
    pred_rf = chunk_predict(rf, Xva, tag=tag + "/rf")
    print("RF prediction %.1f s" % (time.time() - t0))

    # ---- SVM ----
    sidx_s = stratified_sample(Ytr, SVM_PER_CLASS, rng)
    t0 = time.time()
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=2000, random_state=SEED)
    svm.fit(Xtr[sidx_s], Ytr[sidx_s])
    print("SVM training %.1f s (%d pixels, support vectors %d)" %
          (time.time() - t0, len(sidx_s), svm.support_vectors_.shape[0]))
    t0 = time.time()
    pred_svm = chunk_predict(svm, Xva, tag=tag + "/svm")
    print("SVM prediction %.1f s" % (time.time() - t0))

    for name, pred in [("rf", pred_rf), ("svm", pred_svm)]:
        cm = confusion_matrix(Yva, pred, labels=[0, 1, 2])
        np.savetxt(r"%s\rf_svm_confusion_%s_%s.csv" % (OUT_DIR, tag, name),
                   cm, fmt="%d", delimiter=",")
        oa = accuracy_score(Yva, pred)
        kp = cohen_kappa_score(Yva, pred)
        f1m = f1_score(Yva, pred, average="macro")
        f1c = f1_score(Yva, pred, average=None, labels=[0, 1, 2])
        results["%s_%s" % (tag, name)] = (oa, kp, f1m, f1c, cm)
        lines.append("[%s/%s] OA=%.4f Kappa=%.4f macroF1=%.4f | low F1 %.4f medium F1 %.4f high F1 %.4f"
                     % (tag, name, oa, kp, f1m, f1c[0], f1c[1], f1c[2]))
        lines.append("  confusion matrix (rows true, cols predicted): " +
                     " / ".join(" ".join(str(v) for v in row) for row in cm))

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("\nWritten:", OUT_TXT)
