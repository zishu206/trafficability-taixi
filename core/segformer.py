# -*- coding: utf-8 -*-
"""SegFormer 语义分割网络（PyTorch 纯实现，B0 量级）。
忠实复现论文核心：重叠分块嵌入、带序列缩减的高效自注意力、MixFFN（无位置编码）、
分层 MiT 四阶段编码器，以及轻量全 MLP 解码器。输入 13 通道、输出 3 类。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """重叠分块：用卷积实现，相邻 patch 有重叠，缓解边界不连续。"""

    def __init__(self, in_ch, embed, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, kernel_size=kernel,
                              stride=stride, padding=kernel // 2)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """高效自注意力：通过序列缩减比 R 降低 K/V 的 token 数，复杂度由 O(N²) 降至 O(N²/R)。"""

    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.heads = heads
        self.sr_ratio = sr_ratio
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            xr = x.permute(0, 2, 1).reshape(B, C, H, W)
            xr = self.sr_conv(xr).reshape(B, C, -1).permute(0, 2, 1)
            xr = self.sr_norm(xr)
            kv = self.kv(xr)
        else:
            kv = self.kv(x)
        kv = kv.reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MixFFN(nn.Module):
    """混合前馈网络：含 3×3 深度卷积，把位置信息融入 FFN，从而无需位置编码。"""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = F.gelu(self.dw(x))
        x = x.reshape(B, x.shape[1], -1).transpose(1, 2)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, sr_ratio, ffn_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, dim * ffn_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class MiTStage(nn.Module):
    """MiT 一个阶段：重叠分块 + 若干 Transformer 块。"""

    def __init__(self, in_ch, dim, heads, n_blocks, sr_ratio, ffn_ratio,
                 kernel, stride):
        super().__init__()
        self.patch = OverlapPatchEmbed(in_ch, dim, kernel, stride)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, sr_ratio, ffn_ratio)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        x, H, W = self.patch(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        return x.transpose(1, 2).reshape(x.shape[0], -1, H, W)


class SegFormer(nn.Module):
    def __init__(self, n_channels=13, n_classes=3,
                 dims=(32, 64, 160, 256), heads=(1, 2, 5, 8),
                 sr_ratios=(8, 4, 2, 1), decoder_embed=256,
                 ffn_ratio=4, blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()
        # 第一阶段下采样 4 倍，后续每阶段下采样 2 倍
        self.stage1 = MiTStage(n_channels, dims[0], heads[0], blocks_per_stage[0],
                               sr_ratios[0], ffn_ratio, kernel=7, stride=4)
        self.stage2 = MiTStage(dims[0], dims[1], heads[1], blocks_per_stage[1],
                               sr_ratios[1], ffn_ratio, kernel=3, stride=2)
        self.stage3 = MiTStage(dims[1], dims[2], heads[2], blocks_per_stage[2],
                               sr_ratios[2], ffn_ratio, kernel=3, stride=2)
        self.stage4 = MiTStage(dims[2], dims[3], heads[3], blocks_per_stage[3],
                               sr_ratios[3], ffn_ratio, kernel=3, stride=2)
        # 全 MLP 解码器
        self.linear_c = nn.ModuleList([nn.Linear(d, decoder_embed) for d in dims])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_embed * 4, decoder_embed, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_embed),
            nn.GELU(),
        )
        self.linear_pred = nn.Conv2d(decoder_embed, n_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        input_size = x.shape[2:]
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        feats = [f1, f2, f3, f4]
        target = f1.shape[2:]
        outs = []
        for i, f in enumerate(feats):
            h, w = f.shape[2], f.shape[3]
            f = f.flatten(2).transpose(1, 2)
            f = self.linear_c[i](f)
            f = f.transpose(1, 2).reshape(B, -1, h, w)
            outs.append(F.interpolate(f, size=target, mode="bilinear", align_corners=False))
        x = self.linear_fuse(torch.cat(outs, dim=1))
        x = self.linear_pred(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    net = SegFormer()
    print("SegFormer parameter count: %.4f M" % (sum(p.numel() for p in net.parameters()) / 1e6))
    dummy = torch.randn(2, 13, 256, 256)
    print("Output shape:", net(dummy).shape)
