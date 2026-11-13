# -*- coding: utf-8 -*-
"""U-net 全卷积语义分割网络（PyTorch 纯实现）。
对应论文结构图：5 层、27 个卷积层、编码-解码 + 跳跃连接，输入 13 通道、输出 3 类。"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """每级两个 3×3 卷积 + BN + ReLU（U-net 基本单元）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """下采样：2×2 最大池化后接 DoubleConv。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """上采样：双线性放大后与编码器同尺度特征拼接，再 DoubleConv。"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸奇偶不一致时做补齐
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=13, n_classes=3):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)          # 瓶颈层
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)                    # 瓶颈特征
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)
        return self.outc(d4)                   # logits，softmax 在损失/评估处做


if __name__ == "__main__":
    net = UNet()
    n_params = sum(p.numel() for p in net.parameters())
    print("U-net parameter count: %.4f M" % (n_params / 1e6))
    dummy = torch.randn(2, 13, 256, 256)
    print("Output shape:", net(dummy).shape)
