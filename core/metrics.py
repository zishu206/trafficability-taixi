# -*- coding: utf-8 -*-
"""精度评价：混淆矩阵、OA、Kappa、PA/UA、F1、ROC-AUC。"""
import numpy as np


class MetricTracker:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        self._true_all = []
        self._prob_all = []

    def update(self, y_true, y_pred, probs=None):
        """y_true/y_pred：(N,) 类索引；probs：(N,C) 概率（用于 AUC）。"""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        idx = self.n_classes * y_true + y_pred
        cm = np.bincount(idx, minlength=self.n_classes ** 2)
        self.confusion += cm.reshape(self.n_classes, self.n_classes)
        if probs is not None:
            probs = np.asarray(probs)
            if probs.ndim == 4:
                # (N,C,H,W) -> (N,H,W,C)，保证同一空间位置的各类概率对齐
                probs = probs.transpose(0, 2, 3, 1)
            self._true_all.append(y_true)
            self._prob_all.append(probs.reshape(-1, self.n_classes))

    def scores(self):
        cm = self.confusion.astype(np.float64)
        total = cm.sum()
        oa = np.trace(cm) / total if total else 0.0
        row_sum = cm.sum(axis=1)          # 真值各分类数
        col_sum = cm.sum(axis=0)          # 预测各分类数
        pe = np.sum(row_sum * col_sum) / total ** 2 if total else 0.0
        kappa = (oa - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

        pa = np.zeros(self.n_classes)     # 生产者精度/召回
        ua = np.zeros(self.n_classes)     # 用户精度/精确率
        f1 = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            pa[i] = cm[i, i] / row_sum[i] if row_sum[i] else 0.0
            ua[i] = cm[i, i] / col_sum[i] if col_sum[i] else 0.0
            f1[i] = 2 * pa[i] * ua[i] / (pa[i] + ua[i]) if (pa[i] + ua[i]) else 0.0
        return {
            "OA": oa,
            "Kappa": kappa,
            "PA": pa,
            "UA": ua,
            "F1": f1,
            "macro_F1": float(np.mean(f1)),
        }

    def auc(self):
        """宏平均 ROC-AUC（一对多）。需要 scikit-learn。"""
        if not self._true_all:
            return None
        y_true = np.concatenate(self._true_all)
        probs = np.concatenate(self._prob_all, axis=0)
        try:
            from sklearn.metrics import roc_auc_score
            present = np.unique(y_true)
            if len(present) < 2:
                return None
            sub_probs = probs[:, present]
            sub_probs = sub_probs / sub_probs.sum(axis=1, keepdims=True)
            if len(present) == 2:
                return float(roc_auc_score(y_true, sub_probs[:, 1]))
            return float(roc_auc_score(y_true, sub_probs, multi_class="ovr",
                                       average="macro", labels=list(present)))
        except ImportError:
            return None

    def save_confusion(self, path, class_names):
        """保存混淆矩阵为 CSV（行列均为类名，行=真值、列=预测）。"""
        header = "True\\Pred," + ",".join(class_names) + ",Row total"
        lines = [header]
        for i, name in enumerate(class_names):
            row = self.confusion[i]
            lines.append("%s,%s,%d" % (name, ",".join(map(str, row)), int(row.sum())))
        col_tot = self.confusion.sum(axis=0)
        lines.append("Column total,%s,%d" % (",".join(map(str, col_tot)), col_tot.sum()))
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(lines))


def format_report(tracker, class_names):
    s = tracker.scores()
    lines = [
        "Overall accuracy OA = %.4f" % s["OA"],
        "Kappa = %.4f" % s["Kappa"],
        "Macro-F1 = %.4f" % s["macro_F1"],
    ]
    auc = tracker.auc()
    if auc is not None:
        lines.append("Macro ROC-AUC = %.4f" % auc)
    for i, name in enumerate(class_names):
        lines.append("Class[%s] PA=%.4f  UA=%.4f  F1=%.4f"
                     % (name, s["PA"][i], s["UA"][i], s["F1"][i]))
    return "\n".join(lines)
