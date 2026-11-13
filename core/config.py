# -*- coding: utf-8 -*-
"""全局配置：路径、通道顺序、切片与训练超参数。
所有定量超参集中在此，便于复现与修改；通道顺序与论文一严格一致。"""
import os

# Repository root (core/ sits one level below the project root).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # ---------- Data paths (repository-relative; override after importing if needed) ----------
    # 13-band feature GeoTIFF (30 m, 32-bit float; marine cells are nodata)
    FEATURE_TIF = os.path.join(_ROOT, "data", "features_13ch.tif")
    # Label GeoTIFF (single band, values 0/1/2; for a continuous AHP score see data_utils.make_label_from_ahp)
    LABEL_TIF = os.path.join(_ROOT, "data", "label_3class.tif")
    # Output directory (model weights, confusion matrices, logs)
    OUT_DIR = os.path.join(_ROOT, "outputs")

    # ---------- 13 通道顺序（与论文一 4 准则组一致，勿改顺序）----------
    CHANNEL_NAMES = [
        # B1 岩土性质
        "NLI", "NBD", "NCLAY", "NSAL",
        # B2 地形地貌
        "NAI", "NSI", "NCUR", "NDW",
        # B3 水文与微波
        "NWDI", "NSAR",
        # B4 水热与地物
        "NLST", "NVFI", "NLULC",
    ]
    N_CHANNELS = 13
    CLASS_NAMES = ["Low", "Medium", "High"]   # indices 0/1/2
    N_CLASSES = 3
    # AHP 连续分数 -> 三级标签的 Jenks 阈值（仅在由连续分数生成标签时使用）
    AHP_THRESHOLDS = (0.606, 0.672)

    # ---------- 切片 ----------
    PATCH_SIZE = 256          # 切片边长（像元）
    PATCH_STRIDE = 256        # 步长；小于 PATCH_SIZE 即重叠采样
    VALID_RATIO = 0.30        # 空间分块验证比例（7:3）
    DROP_NO_DATA = True       # 含 nodata/海域的切片直接丢弃
    NO_DATA_VALUE = -9999.0

    # ---------- 训练超参（对齐参考博士论文范式）----------
    MODEL = "unet"            # "unet" 或 "segformer"
    EPOCHS = 50
    BATCH_SIZE = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 0           # Windows 下建议 0，避免多进程报错
    SEED = 42
    AUGMENT = True            # 训练集几何增强（翻转/旋转）
    CLASS_WEIGHTS = None      # 类别不平衡时可设为列表，如 [1.0, 1.0, 1.5]
    AMP = True                # 混合精度（GPU 加速；无 GPU 自动关闭）

    # ---------- SegFormer 结构（轻量 B0 量级）----------
    SEG_DIMS = (32, 64, 160, 256)      # 四阶段通道 C1..C4
    SEG_HEADS = (1, 2, 5, 8)
    SEG_FFN_RATIO = 4
    SEG_SR_RATIOS = (8, 4, 2, 1)       # 序列缩减比 R
    SEG_DECODER_EMBED = 256

    def ensure_dirs(self):
        os.makedirs(self.OUT_DIR, exist_ok=True)


CONFIG = Config()
