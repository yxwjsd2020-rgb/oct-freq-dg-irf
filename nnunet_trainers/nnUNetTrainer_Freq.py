# -*- coding: utf-8 -*-
"""
文章D：频域模块训练器（nnU-Net v2 ≥ 2.5；本文件需复制到 nnunetv2/training/nnUNetTrainer/ 下，训练和推理两端都要有）
-----------------------------------------------------------------------------------------------
本文件不依赖 dme_code 的其他文件，可单独复制。全部训练器：250 epochs × 250 iterations，与文章B 基线
nnUNetTrainer_250epochs 相同；除频域模块外不改动任何 nnU-Net 默认设置（数据增强、损失、优化器、学习率）。

训练器                          内容                                                         对应原文
nnUNetTrainer_FreqBase          基线（= nnUNetTrainer_250epochs + 随机种子控制）               —
nnUNetTrainer_RASS              随机幅度谱合成（训练时增强）                                  RAS4DG 的 RASS（Eq.1–2）
nnUNetTrainer_RAS               RASS + 随机掩膜打乱（RMS）                                    RAS4DG 的 RASS + RMS（未含 RSD，见下）
nnUNetTrainer_WTConv            编码器第 1–3 级插入小波卷积（WTConv2d，Haar，3 级）            WTConv（ECCV 2024）
nnUNetTrainer_FFC               编码器第 2–4 级末尾加全局频域分支（快速傅里叶卷积的谱变换）    FFC / LaMa 的 Fourier Unit
nnUNetTrainer_RAS_WTConv        RAS + WTConv 组合                                             —
nnUNetTrainer_SpecTilt          频谱斜率随机化（本研究提出，探索性）                          —
nnUNetTrainer_FreqSmoke         全部模块、2 epochs × 10 iterations，只用于上云前跑通流程      —

与原文的偏离（全部写入方法学和预注册）：
1. RAS4DG 原文代码未能获取（GitHub 不在本环境白名单），依据论文 arXiv:2409.04768 的公式实现：
   δ ~ N(1, σ²)，σ(r) = (2αr)^γ + β，r = √(m²+n²)/√(H²+W²)（m、n 为中心化频率下标，r ≤ 0.5），
   α=3.0, β=0.25, γ=2.0（原文取值）。原文未说明：施加概率、强度归一化、RMS 的区域数和大小——
   本实现取：每个样本以 P_RASS=0.5 施加；增强后按原图均值和标准差重新标准化；RMS 在已做 RASS 的样本上
   以 0.5 概率施加，1–4 个矩形区域，边长为图像边长的 5%–20%，区域内像素随机打乱（标签不变）。
2. RAS4DG 的瓶颈层重建模块（RSD）原文无公式，且其消融中贡献最小（+1.0 Dice），本研究不实现。
3. 原文骨干网络为 SegResNet；本研究放在 nnU-Net 2D 中，增强施加在 nnU-Net 默认增强之后（GPU 上）。
4. WTConv 原文用于替换 ConvNeXt 的深度卷积；U-Net 的卷积是普通卷积，本实现在原 3×3 卷积之前串联一个
   深度 WTConv2d（kernel 5，3 级 Haar），保留原卷积容量。只替换每级中步长为 1、输入=输出通道的卷积。
5. FFC 原文把通道分为局部/全局两路并交叉；本实现只取其谱变换（Spectral Transform + Fourier Unit），
   作为残差分支 x + γ·ST(x) 插在编码器级末尾，γ 初始化为 0（训练开始时等同基线）。
   归一化用 InstanceNorm（与 nnU-Net 一致），原文为 BatchNorm。
"""
import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# =============================================================================================
# 1. 训练时频域增强
# =============================================================================================
def _radius(h, w, device):
    """归一化频率半径 r = √(m²+n²)/√(H²+W²)，m、n 为 FFT 排列下的中心化下标（DC 处 r=0，最大约 0.5）"""
    m = torch.fft.fftfreq(h, device=device) * h
    n = torch.fft.fftfreq(w, device=device) * w
    return torch.sqrt(m[:, None] ** 2 + n[None, :] ** 2) / math.sqrt(h * h + w * w)


def _restandardize(y, x):
    """把增强后的图像拉回原图的均值和标准差（逐样本、逐通道）"""
    dims = (-2, -1)
    ym, ys = y.mean(dims, keepdim=True), y.std(dims, keepdim=True)
    xm, xs = x.mean(dims, keepdim=True), x.std(dims, keepdim=True)
    return (y - ym) / (ys + 1e-6) * xs + xm


def rass(x, alpha=3.0, beta=0.25, gamma=2.0):
    """随机幅度谱合成（RAS4DG Eq.1–2）。x: (B,C,H,W) float32。相位不变，幅度逐频点乘以 δ ~ N(1, σ(r)²)。
    δ 为实数，所以 δ·|F|·e^{iφ} = δ·F。"""
    h, w = x.shape[-2:]
    sigma = (2 * alpha * _radius(h, w, x.device)) ** gamma + beta
    delta = 1 + sigma * torch.randn_like(x)
    y = torch.fft.ifft2(torch.fft.fft2(x) * delta).real
    return _restandardize(y, x)


def spec_tilt(x, k_max=2.0):
    """频谱斜率随机化（本研究提出，探索性）：幅度乘以 exp(k·r/r_max)，k ~ U(−k_max, k_max)，逐样本一个 k。
    k<0 压低高频（模拟多帧平均、较低分辨率，如 Spectralis ART），k>0 抬高高频（更强散斑）。DC 不变。"""
    b = x.shape[0]
    h, w = x.shape[-2:]
    r = _radius(h, w, x.device)
    k = (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * k_max
    y = torch.fft.ifft2(torch.fft.fft2(x) * torch.exp(k * r / r.max())).real
    return _restandardize(y, x)


def random_mask_shuffle(x, n_regions=(1, 4), frac=(0.05, 0.2)):
    """随机掩膜打乱（RAS4DG 的 RMS）：在随机矩形区域内打乱像素位置。x: (C,H,W)，原地修改的副本。"""
    y = x.clone()
    _, h, w = y.shape
    for _ in range(random.randint(*n_regions)):
        rh = max(2, int(h * random.uniform(*frac)))
        rw = max(2, int(w * random.uniform(*frac)))
        i, j = random.randint(0, h - rh), random.randint(0, w - rw)
        patch = y[:, i:i + rh, j:j + rw].reshape(y.shape[0], -1)
        perm = torch.randperm(patch.shape[1], device=y.device)
        y[:, i:i + rh, j:j + rw] = patch[:, perm].reshape(y.shape[0], rh, rw)
    return y


def freq_augment(data, use_rass=False, use_rms=False, use_tilt=False, p_rass=0.5, p_rms=0.5, p_tilt=0.5):
    """对一个 batch 施加频域增强（逐样本随机决定是否施加）。在 float32 下计算，不受 AMP 影响。"""
    if not (use_rass or use_tilt):
        return data
    orig_dtype = data.dtype
    x = data.float().clone()
    b = x.shape[0]
    if use_rass:
        sel = torch.rand(b, device=x.device) < p_rass
        if sel.any():
            x[sel] = rass(x[sel])
            if use_rms:
                for i in torch.nonzero(sel).flatten().tolist():
                    if random.random() < p_rms:
                        x[i] = random_mask_shuffle(x[i])
    if use_tilt:
        sel = torch.rand(b, device=x.device) < p_tilt
        if sel.any():
            x[sel] = spec_tilt(x[sel])
    return x.to(orig_dtype)


# =============================================================================================
# 2. 小波卷积 WTConv2d（Haar / db1），按 Finder 等 ECCV 2024 的结构实现
# =============================================================================================
class HaarWT(nn.Module):
    """单级正交 Haar 小波分解与重建（逐通道），子带顺序 [LL, LH, HL, HH]"""

    def __init__(self):
        super().__init__()
        f = 0.5 * torch.tensor([[[1, 1], [1, 1]],
                                [[1, 1], [-1, -1]],
                                [[1, -1], [1, -1]],
                                [[1, -1], [-1, 1]]], dtype=torch.float32)
        self.register_buffer("filt", f[:, None], persistent=False)  # (4,1,2,2)

    def wt(self, x):  # (B,C,H,W)，H、W 为偶数 → (B,C,4,H/2,W/2)
        b, c, h, w = x.shape
        y = F.conv2d(x, self.filt.to(x.dtype).repeat(c, 1, 1, 1), stride=2, groups=c)
        return y.reshape(b, c, 4, h // 2, w // 2)

    def iwt(self, y):  # (B,C,4,h,w) → (B,C,2h,2w)
        b, c, _, h, w = y.shape
        return F.conv_transpose2d(y.reshape(b, c * 4, h, w), self.filt.to(y.dtype).repeat(c, 1, 1, 1),
                                  stride=2, groups=c)


class WTConv2d(nn.Module):
    """深度小波卷积：输出 = base_scale·DWConv_k(x) + IWT 级联(wavelet_scale·DWConv_k(WT(x)))。输入=输出通道，步长 1。
    结构与原文代码一致：每级对 4 个子带做 k×k 深度卷积；下一级在“未卷积的 LL”上继续分解；重建时由深到浅
    把上一级重建结果加到本级卷积后的 LL 上再做逆变换。奇数尺寸先右/下补零，重建后裁回。"""

    def __init__(self, channels, kernel_size=5, wt_levels=3):
        super().__init__()
        self.levels = wt_levels
        self.haar = HaarWT()
        self.base_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=True)
        self.base_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(channels * 4, channels * 4, kernel_size, padding=kernel_size // 2, groups=channels * 4, bias=False)
             for _ in range(wt_levels)])
        self.wavelet_scale = nn.ParameterList(
            [nn.Parameter(torch.full((1, channels * 4, 1, 1), 0.1)) for _ in range(wt_levels)])

    def forward(self, x):
        ll, lls, hs, shapes = x, [], [], []
        for i in range(self.levels):
            h, w = ll.shape[-2:]
            if min(h, w) < 2:                 # 特征图过小时提前停止（深层、小图）
                break
            shapes.append((h, w))
            if h % 2 or w % 2:
                ll = F.pad(ll, (0, w % 2, 0, h % 2))
            y = self.haar.wt(ll)              # (B,C,4,h/2,w/2)
            ll = y[:, :, 0]
            b, c, _, hh, ww = y.shape
            t = (self.wavelet_convs[i](y.reshape(b, c * 4, hh, ww)) * self.wavelet_scale[i]).reshape(b, c, 4, hh, ww)
            lls.append(t[:, :, 0])
            hs.append(t[:, :, 1:4])
        nxt = 0
        for i in reversed(range(len(shapes))):
            cur = lls.pop() + nxt
            z = self.haar.iwt(torch.cat([cur.unsqueeze(2), hs.pop()], dim=2))
            nxt = z[..., :shapes[i][0], :shapes[i][1]]
        return self.base_conv(x) * self.base_scale + nxt


# =============================================================================================
# 3. 全局频域分支（FFC 的谱变换 / LaMa Fourier Unit）
# =============================================================================================
class FourierUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, 1, bias=False)
        self.norm = nn.InstanceNorm2d(channels * 2, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        with torch.autocast(device_type=x.device.type, enabled=False):   # cuFFT 半精度只支持 2 的幂尺寸
            x = x.float()
            ff = torch.fft.rfft2(x, norm="ortho")
            ff = torch.cat([ff.real, ff.imag], dim=1)
            ff = self.act(self.norm(self.conv(ff)))
            re, im = torch.chunk(ff, 2, dim=1)
            return torch.fft.irfft2(torch.complex(re, im), s=(h, w), norm="ortho")


class GlobalSpectralBranch(nn.Module):
    """x + γ·ST(x)；ST = 1×1 降维 → Fourier Unit（残差）→ 1×1 升维。γ 初始为 0。"""

    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 2, 8)
        self.reduce = nn.Sequential(nn.Conv2d(channels, mid, 1, bias=False), nn.InstanceNorm2d(mid, affine=True),
                                    nn.LeakyReLU(0.01, inplace=True))
        self.fu = FourierUnit(mid)
        self.expand = nn.Conv2d(mid, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        r = self.reduce(x)
        return x + self.gamma * self.expand(r + self.fu(r)).to(x.dtype)


# =============================================================================================
# 4. 把模块装进 nnU-Net 网络（PlainConvUNet 和 ResEnc 通用）
# =============================================================================================
WT_STAGES = (1, 2, 3)     # 编码器第几级（0 为全分辨率级，不改，以控制显存）
FFC_STAGES = (2, 3, 4)


def _conv_blocks(module):
    """找出 ConvDropoutNormReLU 一类的块：有 .conv（Conv2d）和 .all_modules（Sequential）"""
    for m in module.modules():
        if isinstance(getattr(m, "conv", None), nn.Conv2d) and isinstance(getattr(m, "all_modules", None), nn.Sequential):
            yield m


def add_wtconv(network, stages=WT_STAGES, kernel_size=5, wt_levels=3):
    enc_stages = network.encoder.stages
    n = 0
    for s in stages:
        if s >= len(enc_stages):
            continue
        for blk in list(_conv_blocks(enc_stages[s])):
            conv = blk.conv
            if tuple(conv.stride) != (1, 1) or conv.in_channels != conv.out_channels or conv.groups != 1:
                continue
            new = nn.Sequential(WTConv2d(conv.in_channels, kernel_size, wt_levels), conv)
            idx = [i for i, mm in enumerate(blk.all_modules) if mm is conv]
            if not idx:
                continue
            blk.all_modules[idx[0]] = new
            blk.conv = new
            n += 1
    if n == 0:
        raise RuntimeError("WTConv：没有找到可替换的卷积，请检查网络结构（dynamic_network_architectures 版本）")
    return network


def add_ffc(network, stages=FFC_STAGES):
    enc_stages = network.encoder.stages
    chans = network.encoder.output_channels
    n = 0
    for s in stages:
        if s >= len(enc_stages):
            continue
        enc_stages[s] = nn.Sequential(enc_stages[s], GlobalSpectralBranch(chans[s]))
        n += 1
    if n == 0:
        raise RuntimeError("FFC：编码器级数不足，未插入任何模块")
    return network


def _build(arch_name, arch_kwargs, req_import, n_in, n_out, deep_sup, wt=False, ffc=False):
    net = nnUNetTrainer.build_network_architecture(arch_name, arch_kwargs, req_import, n_in, n_out, deep_sup)
    if wt:
        add_wtconv(net)
    if ffc:
        add_ffc(net)
    return net


# =============================================================================================
# 5. 训练器
# =============================================================================================
class nnUNetTrainer_FreqBase(nnUNetTrainer):
    """250 epochs；环境变量 DME_SEED（整数）控制随机种子。不重写 __init__（nnU-Net ≥ 2.5 会检查其签名）。"""
    USE_RASS = False
    USE_RMS = False
    USE_TILT = False
    EPOCHS, ITERS = 250, 250

    def initialize(self):
        self.num_epochs, self.num_iterations_per_epoch = self.EPOCHS, self.ITERS
        seed = int(os.environ.get("DME_SEED", "0"))
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        super().initialize()
        self.print_to_log_file(f"[文章D] {type(self).__name__} seed={seed} RASS={self.USE_RASS} RMS={self.USE_RMS} "
                               f"TILT={self.USE_TILT} epochs={self.EPOCHS}x{self.ITERS} "
                               f"params={sum(p.numel() for p in self.network.parameters()) / 1e6:.2f}M")

    def train_step(self, batch):
        if self.USE_RASS or self.USE_TILT:
            batch = dict(batch)
            data = batch["data"].to(self.device, non_blocking=True)
            batch["data"] = freq_augment(data, self.USE_RASS, self.USE_RMS, self.USE_TILT)
        return super().train_step(batch)


class nnUNetTrainer_RASS(nnUNetTrainer_FreqBase):
    USE_RASS = True


class nnUNetTrainer_RAS(nnUNetTrainer_FreqBase):
    USE_RASS, USE_RMS = True, True


class nnUNetTrainer_SpecTilt(nnUNetTrainer_FreqBase):
    USE_TILT = True


class nnUNetTrainer_WTConv(nnUNetTrainer_FreqBase):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return _build(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                      num_input_channels, num_output_channels, enable_deep_supervision, wt=True)


class nnUNetTrainer_FFC(nnUNetTrainer_FreqBase):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return _build(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                      num_input_channels, num_output_channels, enable_deep_supervision, ffc=True)


class nnUNetTrainer_RAS_WTConv(nnUNetTrainer_FreqBase):
    USE_RASS, USE_RMS = True, True

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return _build(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                      num_input_channels, num_output_channels, enable_deep_supervision, wt=True)


class nnUNetTrainer_FreqSmoke(nnUNetTrainer_FreqBase):
    """上云自测：全部模块，2 epochs × 10 iterations（约几分钟），验证训练—验证—推理全流程不报错"""
    USE_RASS, USE_RMS, USE_TILT = True, True, True
    EPOCHS, ITERS = 2, 10

    def initialize(self):
        super().initialize()
        self.num_val_iterations_per_epoch = 5

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return _build(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                      num_input_channels, num_output_channels, enable_deep_supervision, wt=True, ffc=True)
