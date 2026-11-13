# -*- coding: utf-8 -*-
"""
机理标签 RF/SVM 补跑（ahp 两组已完成并保存混淆矩阵）。
每个模型预测完成后立即保存混淆矩阵，避免中断丢失；最后汇总四组指标写出文本。
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


def split_indices(GR, GC):
    is_val = np.array([(abs((int(GR[i]) * 73856093) ^ (int(GC[i]) * 19349663)) % 100) < 30
                       for i in range(len(GR))])
    return np.where(~is_val)[0], np.where(is_val)[0]


def flatten(tiles_X, tiles_Y):
    x = np.transpose(tiles_X, (0, 2, 3, 1)).reshape(-1, 13)
    y = tiles_Y.reshape(-1)
    return x, y


def stratified_sample(y, per_class, rng):
    idx = []
    for c in range(3):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, min(per_class, len(ci)), replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return idx


def chunk_predict(model, X, chunk=100000, tag=""):
    preds = np.empty(X.shape[0], dtype=np.int64)
    nb = (X.shape[0] + chunk - 1) // chunk
    for b, s in enumerate(range(0, X.shape[0], chunk)):
        preds[s:s + chunk] = model.predict(X[s:s + chunk])
        print("  [%s] %d/%d (%.0f%%)" % (tag, b + 1, nb, 100 * (b + 1) / nb), flush=True)
    return preds


d = np.load(NPZ)
X, GR, GC = d["X"], d["GR"], d["GC"]
tr_idx, va_idx = split_indices(GR, GC)
rng = np.random.default_rng(SEED)

Xtr, Ytr = flatten(X[tr_idx], d["Y_mech"][tr_idx])
Xva, Yva = flatten(X[va_idx], d["Y_mech"][va_idx])
print("mech training pixels %d, validation pixels %d" % (Ytr.size, Yva.size))

# ---- RF (save immediately) ----
sidx = stratified_sample(Ytr, 20000, rng)
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(Xtr[sidx], Ytr[sidx])
print("RF training %.1fs" % (time.time() - t0))
pred_rf = chunk_predict(rf, Xva, tag="mech/rf")
cm = confusion_matrix(Yva, pred_rf, labels=[0, 1, 2])
np.savetxt(r"%s\rf_svm_confusion_mech_rf.csv" % OUT_DIR, cm, fmt="%d", delimiter=",")
print("mech/rf confusion matrix saved", flush=True)

# ---- SVM (save immediately) ----
sidx_s = stratified_sample(Ytr, 7000, rng)
t0 = time.time()
svm = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=2000, random_state=SEED)
svm.fit(Xtr[sidx_s], Ytr[sidx_s])
print("SVM training %.1fs (support vectors %d)" % (time.time() - t0, svm.support_vectors_.shape[0]))
pred_svm = chunk_predict(svm, Xva, tag="mech/svm")
cm = confusion_matrix(Yva, pred_svm, labels=[0, 1, 2])
np.savetxt(r"%s\rf_svm_confusion_mech_svm.csv" % OUT_DIR, cm, fmt="%d", delimiter=",")
print("mech/svm confusion matrix saved", flush=True)

# ---- Summarize all four groups (recomputed from confusion matrices to keep consistent) ----
lines = ["Common-domain RF/SVM pixel-level classification (validation set, 225 tiles, %d pixels)" % Yva.size, ""]
for tag in ("ahp", "mech"):
    for name in ("rf", "svm"):
        cm = np.loadtxt(r"%s\rf_svm_confusion_%s_%s.csv" % (OUT_DIR, tag, name),
                        delimiter=",").astype(np.int64)
        tot = cm.sum()
        oa = np.trace(cm) / tot
        # Kappa
        row = cm.sum(axis=1); col = cm.sum(axis=0)
        pe = np.sum(row * col) / tot ** 2
        kp = (oa - pe) / (1 - pe)
        f1c = []
        for c in range(3):
            pa = cm[c, c] / row[c]
            ua = cm[c, c] / col[c]
            f1c.append(2 * pa * ua / (pa + ua))
        f1m = np.mean(f1c)
        lines.append("[%s/%s] OA=%.4f Kappa=%.4f macroF1=%.4f | low F1 %.4f medium F1 %.4f high F1 %.4f"
                     % (tag, name, oa, kp, f1m, f1c[0], f1c[1], f1c[2]))
        lines.append("  confusion matrix: " + " / ".join(" ".join(str(v) for v in r) for r in cm))

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("Written:", OUT_TXT)
