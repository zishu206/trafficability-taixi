# -*- coding: utf-8 -*-
"""自检脚本：用合成数据跑通 Dataset -> 模型 -> 损失/反传 -> 指标 的完整闭环，
不依赖 GeoTIFF/rasterio，用于确认环境与代码逻辑正确。
运行：python smoke_test.py"""
import os
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import TrafficDataset, spatial_split
from metrics import MetricTracker, format_report
from unet import UNet
from segformer import SegFormer

P = 64
N_PATCH = 16
rng = np.random.default_rng(0)
patches = []
for i in range(N_PATCH):
    x = rng.random((13, P, P), dtype=np.float32)
    y = rng.integers(0, 3, size=(P, P)).astype(np.int64)
    patches.append((x, y, i % 4, i // 4))

train_p, val_p = spatial_split(patches, 0.3)
train_loader = DataLoader(TrafficDataset(train_p, augment=True), batch_size=4, shuffle=True)
val_loader = DataLoader(TrafficDataset(val_p, augment=False), batch_size=4)

for name, model in [("unet", UNet(13, 3)), ("segformer", SegFormer(13, 3))]:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    tracker = MetricTracker(3)
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.softmax(model(x), dim=1)
            preds = torch.argmax(probs, dim=1)
            tracker.update(y.numpy(), preds.numpy(), probs.numpy())
    with tempfile.TemporaryDirectory() as d:
        tracker.save_confusion(os.path.join(d, "cm.csv"), ["Low", "Medium", "High"])
    print("==== %s smoke test passed ====" % name)
    print(format_report(tracker, ["Low", "Medium", "High"]))
print("\nSMOKE TEST PASSED")
