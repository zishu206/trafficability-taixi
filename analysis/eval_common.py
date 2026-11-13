# -*- coding: utf-8 -*-
"""
三标签公共域：加载 4 个权重在验证集（225 切片）重算完整指标（含 AUC），
并挑选代表性切片生成预测对比图。
输出 common_eval_metrics.txt、fig_pred_compare.png。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from data_utils import TrafficDataset
from metrics import MetricTracker, format_report
from unet import UNet
from segformer import SegFormer
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "common_domain.npz")
OUTDIR = OUT_DIR
os.makedirs(OUTDIR, exist_ok=True)
device = torch.device("cuda")


def split_indices(GR, GC):
    is_val = []
    for i in range(len(GR)):
        code = (int(GR[i]) * 73856093) ^ (int(GC[i]) * 19349663)
        is_val.append((abs(code) % 100) < 30)
    return np.where(np.array(is_val))[0]


def build_model(name):
    if name == "unet":
        return UNet(13, 3), 16
    return SegFormer(13, 3, dims=CONFIG.SEG_DIMS, heads=CONFIG.SEG_HEADS,
                     sr_ratios=CONFIG.SEG_SR_RATIOS,
                     decoder_embed=CONFIG.SEG_DECODER_EMBED,
                     ffn_ratio=CONFIG.SEG_FFN_RATIO), 32


d = np.load(NPZ)
X, GR, GC = d["X"], d["GR"], d["GC"]
labels = {"ahp": d["Y_ahp"], "mech": d["Y_mech"]}
val_idx = split_indices(GR, GC)
print("Validation tiles:", len(val_idx))

reports, preds_store = {}, {}
for tag in ["ahp", "mech"]:
    for name in ["unet", "segformer"]:
        model, bs = build_model(name)
        wpath = os.path.join(OUT_DIR,
                             "common_%s_%s.pth" % (tag, name))
        model.load_state_dict(torch.load(wpath, map_location=device))
        model.to(device); model.eval()
        val_p = [(np.ascontiguousarray(X[i]), np.ascontiguousarray(labels[tag][i].astype(np.int64)),
                  0, 0) for i in val_idx]
        loader = DataLoader(TrafficDataset(val_p, augment=False), batch_size=bs, num_workers=0)
        tracker = MetricTracker(3)
        all_pred = []
        with torch.no_grad():
            for x, y in loader:
                probs = torch.softmax(model(x.to(device)), dim=1)
                pred = torch.argmax(probs, 1).cpu().numpy()
                tracker.update(y.numpy(), pred, probs.cpu().numpy())
                all_pred.append(pred)
        key = "%s_%s" % (tag, name)
        reports[key] = format_report(tracker, CONFIG.CLASS_NAMES)
        preds_store[key] = np.concatenate(all_pred, axis=0)  # (225,128,128)
        print(reports[key], "\n")

with open(os.path.join(OUTDIR, "common_eval_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("Common-domain validation set (225 tiles, 3,686,400 pixels), recomputed from checkpoints\n\n")
    for k, v in reports.items():
        f.write("===== %s =====\n%s\n\n" % (k, v))

# ---- 预测对比图：选 4 个代表性切片 ----
# 计算每个验证切片两标签分歧率
Yahp_v = labels["ahp"][val_idx]; Ymech_v = labels["mech"][val_idx]
disagree = (Yahp_v != Ymech_v).reshape(len(val_idx), -1).mean(axis=1)
order = np.argsort(-disagree)
sel = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
sel_local = [int(s) for s in sel]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

risk = ["#C00000", "#ED7D31", "#548235"]
from matplotlib.colors import ListedColormap
cmap3 = ListedColormap(risk)

rows_title = ["AHP ground truth", "Mechanistic ground truth", "AHP + U-net", "AHP + SegFormer",
              "Mechanistic + U-net", "Mechanistic + SegFormer"]
fig, axes = plt.subplots(6, 4, figsize=(13.5, 19))
for col, sl in enumerate(sel_local):
    gi = val_idx[sl]
    grids = [Yahp_v[sl], Ymech_v[sl],
             preds_store["ahp_unet"][sl], preds_store["ahp_segformer"][sl],
             preds_store["mech_unet"][sl], preds_store["mech_segformer"][sl]]
    for r, g in enumerate(grids):
        ax = axes[r, col]
        ax.imshow(g, cmap=cmap3, vmin=0, vmax=2)
        if col == 0:
            ax.set_ylabel(rows_title[r], fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, col].set_title("Tile #%d (label disagreement %.2f)" % (gi, disagree[sl]), fontsize=10)
fig.suptitle("Representative common-domain tiles: AHP/mechanistic ground truth and "
             "dual-model predictions", fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(OUTDIR, "fig_pred_compare.png"), dpi=170)
print("saved fig_pred_compare.png")
