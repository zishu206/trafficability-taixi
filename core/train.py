# -*- coding: utf-8 -*-
"""训练入口：U-net / SegFormer 双模型。
运行示例：
    python train.py --model unet
    python train.py --model segformer
路径与超参在 config.py 中修改。"""
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import CONFIG
from data_utils import build_dataloaders
from metrics import MetricTracker, format_report
from unet import UNet
from segformer import SegFormer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name, cfg):
    if name == "unet":
        return UNet(cfg.N_CHANNELS, cfg.N_CLASSES)
    if name == "segformer":
        return SegFormer(
            cfg.N_CHANNELS, cfg.N_CLASSES,
            dims=cfg.SEG_DIMS, heads=cfg.SEG_HEADS,
            sr_ratios=cfg.SEG_SR_RATIOS, decoder_embed=cfg.SEG_DECODER_EMBED,
            ffn_ratio=cfg.SEG_FFN_RATIO,
        )
    raise ValueError("Unknown model: %s" % name)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    tracker = MetricTracker(cfg.N_CLASSES)
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        tracker.update(y.numpy(), preds.cpu().numpy(), probs.cpu().numpy())
    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "segformer"], default=CONFIG.MODEL)
    args = parser.parse_args()
    cfg = CONFIG
    cfg.MODEL = args.model
    cfg.ensure_dirs()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.AMP and device.type == "cuda"
    print("Device:", device, "| Mixed precision:", use_amp)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg.MODEL, cfg).to(device)

    weight = None
    if cfg.CLASS_WEIGHTS is not None:
        weight = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                 weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = -1.0
    ckpt_path = os.path.join(cfg.OUT_DIR, "best_%s.pth" % cfg.MODEL)
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc="Epoch %d/%d" % (epoch, cfg.EPOCHS))
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            pbar.set_postfix(loss="%.4f" % loss.item())
        train_loss = running / len(train_loader.dataset)

        tracker = evaluate(model, val_loader, device, cfg)
        scores = tracker.scores()
        print("Epoch %d training loss %.4f | validation OA %.4f  macroF1 %.4f  Kappa %.4f"
              % (epoch, train_loss, scores["OA"], scores["macro_F1"], scores["Kappa"]))

        if scores["macro_F1"] > best_metric:
            best_metric = scores["macro_F1"]
            torch.save({"model": cfg.MODEL, "state_dict": model.state_dict(),
                        "epoch": epoch}, ckpt_path)
            tracker.save_confusion(
                os.path.join(cfg.OUT_DIR, "confusion_%s.csv" % cfg.MODEL),
                cfg.CLASS_NAMES)
            print("  -> Saved best checkpoint and confusion matrix")

    # 训练结束，加载最佳模型输出完整报告
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    tracker = evaluate(model, val_loader, device, cfg)
    report = format_report(tracker, cfg.CLASS_NAMES)
    report_path = os.path.join(cfg.OUT_DIR, "report_%s.txt" % cfg.MODEL)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Model: %s\n%s\n" % (cfg.MODEL, report))
    print("\n===== Final validation report =====\n" + report)
    print("\nReport saved:", report_path)


if __name__ == "__main__":
    main()
