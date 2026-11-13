# -*- coding: utf-8 -*-
"""数据读取、切片、空间划分、增强与 PyTorch Dataset。"""
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None


def read_raster(path):
    """读取栅格为 (C,H,W) float32 数组。单波段标签返回 (1,H,W)。"""
    if rasterio is None:
        raise ImportError("rasterio is missing; install it with: python -m pip install rasterio")
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile
    return arr, profile


def make_label_from_ahp(ahp_2d, thresholds):
    """连续 AHP 分数 -> 三级标签。阈值 (t1,t2)：
    <t1 为 0(低)，t1~t2 为 1(中)，>=t2 为 2(高)。"""
    t1, t2 = thresholds
    label = np.zeros_like(ahp_2d, dtype=np.int64)
    label[ahp_2d >= t1] = 1
    label[ahp_2d >= t2] = 2
    return label


def extract_patches(features, label, cfg):
    """滑动窗口切片。features:(C,H,W)，label:(H,W)。
    返回 list[(x_patch, y_patch, grid_r, grid_c)]。"""
    _, H, W = features.shape
    p = cfg.PATCH_SIZE
    stride = cfg.PATCH_STRIDE
    rows = list(range(0, H - p + 1, stride))
    cols = list(range(0, W - p + 1, stride))
    patches = []
    for gr, r in enumerate(rows):
        for gc, c in enumerate(cols):
            x = features[:, r:r + p, c:c + p]
            y = label[r:r + p, c:c + p]
            if cfg.DROP_NO_DATA:
                # 含 nodata 或无效标签（海域）则丢弃
                if np.isnan(x).any() or (x == cfg.NO_DATA_VALUE).any():
                    continue
                if np.isnan(y).any() or (y < 0).any():
                    continue
            patches.append((np.ascontiguousarray(x),
                            np.ascontiguousarray(y.astype(np.int64)),
                            gr, gc))
    return patches


def spatial_split(patches, valid_ratio, seed=42):
    """空间分块划分（不随机打散像元）：以切片网格坐标做确定性取模，
    每个完整切片整体归入训练或验证，两边无重叠像元。"""
    rng_mod = 100
    valid_thr = int(round(valid_ratio * rng_mod))
    train, val = [], []
    for item in patches:
        gr, gc = item[2], item[3]
        code = (gr * 73856093) ^ (gc * 19349663)
        if (abs(code) % rng_mod) < valid_thr:
            val.append(item)
        else:
            train.append(item)
    return train, val


def geometric_augment(x, y):
    """几何增强：随机水平/垂直翻转与 90° 旋转，特征与标签同步。"""
    if np.random.rand() < 0.5:
        x = x[:, :, ::-1]; y = y[:, ::-1]
    if np.random.rand() < 0.5:
        x = x[:, ::-1, :]; y = y[::-1, :]
    k = np.random.randint(0, 4)
    if k:
        x = np.rot90(x, k, axes=(1, 2))
        y = np.rot90(y, k, axes=(0, 1))
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class TrafficDataset(Dataset):
    def __init__(self, patches, augment=False):
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x, y, _, _ = self.patches[idx]
        if self.augment:
            x, y = geometric_augment(x, y)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64))


def build_dataloaders(cfg):
    """一站式：读图 -> 切片 -> 空间划分 -> DataLoader。"""
    from torch.utils.data import DataLoader
    features, _ = read_raster(cfg.FEATURE_TIF)
    assert features.shape[0] == cfg.N_CHANNELS, (
        "feature channel count = %d does not match N_CHANNELS = %d in config"
        % (features.shape[0], cfg.N_CHANNELS))
    lab_arr, _ = read_raster(cfg.LABEL_TIF)
    label = lab_arr[0]
    patches = extract_patches(features, label, cfg)
    if not patches:
        raise RuntimeError("No valid patches were extracted; check the nodata settings or PATCH_SIZE/STRIDE")
    train_p, val_p = spatial_split(patches, cfg.VALID_RATIO, cfg.SEED)
    if not train_p or not val_p:
        raise RuntimeError("Train/validation set is empty after spatial split; enlarge the study area or reduce PATCH_SIZE/STRIDE")
    train_ds = TrafficDataset(train_p, augment=cfg.AUGMENT)
    val_ds = TrafficDataset(val_p, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS)
    print("Training patches %d, validation patches %d" % (len(train_ds), len(val_ds)))
    return train_loader, val_loader
