# -*- coding: utf-8 -*-
"""
三标签公共域 GPU 训练：
同一 771 切片、同一空间划分（GR/GC 哈希），对 AHP 与机理两个标签分别训练
U-net 与 SegFormer（30 轮，同超参），仅标签 Y 不同。
输出 common_<tag>_<model>.pth、混淆矩阵、common_report.txt、history（供收敛曲线）。
"""
import argparse
import os
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
EPOCHS = 30


def split_indices(GR, GC):
    is_val = []
    for i in range(len(GR)):
        code = (int(GR[i]) * 73856093) ^ (int(GC[i]) * 19349663)
        is_val.append((abs(code) % 100) < 30)
    is_val = np.array(is_val)
    return np.where(~is_val)[0], np.where(is_val)[0]


def run_model(name, tag, X, Y, tr_idx, va_idx):
    device = torch.device("cuda")
    if name == "unet":
        model = UNet(13, 3)
        bs = 16
    else:
        model = SegFormer(13, 3, dims=CONFIG.SEG_DIMS, heads=CONFIG.SEG_HEADS,
                          sr_ratios=CONFIG.SEG_SR_RATIOS,
                          decoder_embed=CONFIG.SEG_DECODER_EMBED,
                          ffn_ratio=CONFIG.SEG_FFN_RATIO)
        bs = 32
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def patches(idx):
        return [(np.ascontiguousarray(X[i]), np.ascontiguousarray(Y[i].astype(np.int64)),
                 0, 0) for i in idx]
    train_p, val_p = patches(tr_idx), patches(va_idx)
    train_loader = DataLoader(TrafficDataset(train_p, augment=True),
                              batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(TrafficDataset(val_p, augment=False),
                            batch_size=bs, num_workers=0)

    history = []
    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        tracker = MetricTracker(3)
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                probs = torch.softmax(logits, dim=1)
                tracker.update(y.numpy(), torch.argmax(probs, 1).cpu().numpy(),
                               probs.cpu().numpy())
        sc = tracker.scores()
        history.append((epoch, sc["OA"], sc["macro_F1"], sc["Kappa"]))
        if sc["macro_F1"] > best_f1:
            best_f1 = sc["macro_F1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print("[%s/%s] epoch %2d OA=%.4f macroF1=%.4f Kappa=%.4f"
              % (tag, name, epoch, sc["OA"], sc["macro_F1"], sc["Kappa"]), flush=True)

    model.load_state_dict(best_state)
    tracker = MetricTracker(3)
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            tracker.update(y.numpy(), torch.argmax(probs, 1).cpu().numpy(),
                           probs.cpu().numpy())
    tracker.save_confusion(os.path.join(OUTDIR, "common_confusion_%s_%s.csv" % (tag, name)),
                           CONFIG.CLASS_NAMES)
    torch.save(best_state, os.path.join(OUTDIR, "common_%s_%s.pth" % (tag, name)))
    report = format_report(tracker, CONFIG.CLASS_NAMES)
    return report, history, tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["ahp", "mech", "both"], default="both")
    parser.add_argument("--model", choices=["unet", "segformer", "both"], default="both")
    args = parser.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)

    d = np.load(NPZ)
    X = d["X"]; GR, GC = d["GR"], d["GC"]
    labels = {"ahp": d["Y_ahp"], "mech": d["Y_mech"]}
    tr_idx, va_idx = split_indices(GR, GC)
    print("Common-domain training patches %d, validation patches %d (total %d)\n"
          % (len(tr_idx), len(va_idx), len(X)), flush=True)

    use_labels = ["ahp", "mech"] if args.label == "both" else [args.label]
    use_models = ["unet", "segformer"] if args.model == "both" else [args.model]
    all_reports, hist_store = {}, {}
    for tag in use_labels:
        for m in use_models:
            report, history, _ = run_model(m, tag, X, labels[tag], tr_idx, va_idx)
            all_reports["%s_%s" % (tag, m)] = report
            hist_store["%s_%s" % (tag, m)] = np.array(history)
            print("\n" + report + "\n", flush=True)

    with open(os.path.join(OUTDIR, "common_report.txt"), "w", encoding="utf-8") as f:
        f.write("Common-domain training (30 epochs, 771 tiles)\n")
        f.write("Training %d validation %d\n\n" % (len(tr_idx), len(va_idx)))
        for k, v in all_reports.items():
            f.write("===== %s =====\n%s\n\n" % (k, v))
    np.savez(os.path.join(OUTDIR, "common_history.npz"), **hist_store)
    print("Report and history saved:", OUTDIR)


if __name__ == "__main__":
    main()
