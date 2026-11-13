# -*- coding: utf-8 -*-
"""真实数据训练：读取 real_data.npz，空间 7:3 划分，训练 U-net 与 SegFormer。
运行：python real_train.py
说明：标签为 AHP 弱标签（Jenks 三级），指标衡量模型对该弱标签的像素级复现能力。"""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from data_utils import TrafficDataset, spatial_split
from metrics import MetricTracker, format_report
from unet import UNet
from segformer import SegFormer
# Repository-relative locations; environment variables override the defaults.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("TRAFFIC_DATA", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("TRAFFIC_OUT", os.path.join(ROOT, "outputs"))

NPZ = os.path.join(DATA_DIR, "real_data.npz")
OUTDIR = OUT_DIR
EPOCHS = 15
BATCH = 16


def load_patches():
    d = np.load(NPZ)
    X, Y, GR, GC = d["X"], d["Y"], d["GR"], d["GC"]
    patches = [(X[i], Y[i], int(GR[i]), int(GC[i])) for i in range(X.shape[0])]
    return patches, np.asarray(d["breaks"])


def run_model(name, train_p, val_p, n_classes, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "unet":
        model = UNet(13, n_classes)
    else:
        model = SegFormer(13, n_classes,
                          dims=CONFIG.SEG_DIMS, heads=CONFIG.SEG_HEADS,
                          sr_ratios=CONFIG.SEG_SR_RATIOS,
                          decoder_embed=CONFIG.SEG_DECODER_EMBED,
                          ffn_ratio=CONFIG.SEG_FFN_RATIO)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_loader = DataLoader(TrafficDataset(train_p, augment=True),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TrafficDataset(val_p, augment=False), batch_size=batch_size)

    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        tracker = MetricTracker(n_classes)
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                tracker.update(y.numpy(), preds.cpu().numpy(), probs.cpu().numpy())
        f1 = tracker.scores()["macro_F1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print("[%s] epoch %2d  OA=%.4f macroF1=%.4f Kappa=%.4f"
              % (name, epoch, tracker.scores()["OA"], f1, tracker.scores()["Kappa"]))

    model.load_state_dict(best_state)
    tracker = MetricTracker(n_classes)
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            tracker.update(y.numpy(), torch.argmax(probs, 1).cpu().numpy(),
                           probs.cpu().numpy())
    tracker.save_confusion(os.path.join(OUTDIR, "real_confusion_%s.csv" % name),
                           CONFIG.CLASS_NAMES)
    torch.save(best_state, os.path.join(OUTDIR, "real_%s.pth" % name))
    return format_report(tracker, CONFIG.CLASS_NAMES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "segformer", "both"], default="both")
    args = parser.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)

    patches, breaks = load_patches()
    train_p, val_p = spatial_split(patches, 0.30)
    print("Training patches %d, validation patches %d, Jenks breaks %s\n"
          % (len(train_p), len(val_p), np.round(breaks, 4)))

    models = ["unet", "segformer"] if args.model == "both" else [args.model]
    summaries = []
    for m in models:
        bs = 4 if m == "unet" else 16
        report = run_model(m, train_p, val_p, CONFIG.N_CLASSES, bs)
        summaries.append("===== %s =====\n%s" % (m, report))
        print("\n" + report + "\n")

    with open(os.path.join(OUTDIR, "real_report.txt"), "w", encoding="utf-8") as f:
        f.write("Jenks three-class breaks: %s\nTraining patches %d validation patches %d\n\n%s"
                % (np.round(breaks, 4), len(train_p), len(val_p),
                   "\n\n".join(summaries)))
    print("Report saved:", os.path.join(OUTDIR, "real_report.txt"))


if __name__ == "__main__":
    main()
