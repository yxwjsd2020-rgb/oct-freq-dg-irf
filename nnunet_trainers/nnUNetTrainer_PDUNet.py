# -*- coding: utf-8 -*-
"""
文章D 探索性配置：按开题报告与早期稿件重新实现的 PDU-Net（nnU-Net v2 ≥ 2.5）
-----------------------------------------------------------------------------------------------
本文件需复制到 nnunetv2/training/nnUNetTrainer/ 下，训练端和推理端都要有。它依赖同目录的
nnUNetTrainer_Freq.py（继承 nnUNetTrainer_FreqBase，以使用同一套随机种子控制），不依赖 torchvision。

PDU-Net 没有公开代码，开题时的工程团队也没有保留代码。本实现依据：
  [K] 开题报告《基于PDU-Net的OCT影像组学与GAGs转录组学的DME多模态关联机制探索研究》2.1.2 节；
  [M] 早期稿件（桌面\\PDU-NET\\PDU-Net稿件_修订版）2.3 节，开题未写明的细节取自这里。
  两者冲突时以 [K] 为准（文章D 方案文档"与开题 PDU-Net 的关系"一节已定）。

结构（输入先补零到 32 的倍数，输出再裁回原尺寸）
  编码器   ResNet-34（[K][M]）：7×7/2 卷积 → 3×3/2 最大池化 → 4 个残差阶段；五个层级 e0–e4，
           分辨率 1/2、1/4、1/8、1/16、1/32，通道 64、64、128、256、512。ImageNet 预训练（[K][M]）。
  PANet    在跳跃连接路径上加自底向上的路径聚合（[K][M]，细节按 [M]）：
           N0 = e0；N(i+1) = Conv3×3( Conv3×3/2(Ni) + Conv1×1(e(i+1)) )，各卷积后接 BN + ReLU，
           通道保持各层级原通道数。N0–N3 作为跳跃连接，N4 作为解码器起点。
  解码器   每级：2×2 转置卷积上采样 → 与对应层级的 Ni 拼接 → 双重注意力（DAN）→ 两个 3×3 卷积（融合）。
           解码器通道 256、128、64、32、16（[M] 只说"与编码器对称"，通道数为本实现设定，同常用 ResNet-UNet）。
           第 5 级上采样回全分辨率，没有可拼接的编码器特征，因此不做融合、不加 DAN。
  DAN      嵌在解码器每个特征融合模块之前（[K][M]）。通道注意力：全局平均池化 → FC（降维比 16）→ ReLU → FC → Sigmoid；
           空间注意力：通道维最大池化与平均池化拼接 → 7×7 卷积 → Sigmoid（[K][M]）。
           组合方式：[K] 写"并行计算空间与通道注意力权重"，两组权重都由输入特征算出，再依次相乘（nnUNetTrainer_PDUNet，主用）；
           [M] 写"串行、先通道后空间"，空间权重由乘过通道权重的特征算出（nnUNetTrainer_PDUNetSerial，默认不跑）。
           两者参数完全相同，只是前向计算不同，所以做成两个训练器（推理时由训练器名决定），不用环境变量切换。
  损失     L = 0.6·Dice + 0.4·Focal（[M] 的取值；[M] 称由验证集网格搜索确定，但该稿从未训练，本研究直接采用该值，不再搜索）。
           Focal：α_t = 0.25（前景）/ 0.75（背景），γ = 2（[M]，同 Lin 2017）。Dice 只计前景类，平滑项 1e-5。
  优化     Adam，学习率 1e-4，余弦退火到 0，批量 8（[K][M]）。

与开题 / 稿件的偏离（写入方法学）：
  1. 训练长度：[K][M] 为 200 个周期 + 早停（验证损失 15 轮不降即停，并恢复验证损失最低的模型）。
     本研究唯一的验证集（TowardPi 第 0 折验证集）同时是内部测试集，用它早停或选模型会造成信息泄露，
     因此改为与文章D 其他配置相同的固定训练长度（250 epochs × 250 iterations），用最终模型，余弦退火按 250 个 epoch。
  2. 数据预处理、patch 大小、数据增强、推理（滑窗 + 镜像）沿用 nnU-Net 默认，与其他配置一致，便于比较网络本身；
     [K][M] 未写这些细节。
  3. 输入为单通道 OCT（nnU-Net z-score 标准化）。ImageNet 预训练的首层卷积按常规做法把 RGB 三个通道的权重相加，
     得到单通道权重；没有使用 ImageNet 的均值/方差归一化。
  4. 不使用 nnU-Net 的深监督（PDU-Net 只有一个输出）。
  5. [M] 的结构示意图未找到，解码器通道数、BN/ReLU 的位置为本实现设定。

ImageNet 权重（resnet34-b627a593.pth，即 torchvision IMAGENET1K_V1，只在训练开始时加载一次）：
  优先读环境变量 PDU_RESNET34_WEIGHTS 指定的本地文件；否则用 torch.hub 从 download.pytorch.org 下载并校验哈希前缀。
  两者都失败时报错停止；确需从头训练（不符合开题设计）时设 PDU_ALLOW_SCRATCH=1，日志会写明。
"""
import hashlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Freq import nnUNetTrainer_FreqBase

RESNET34_URL = "https://download.pytorch.org/models/resnet34-b627a593.pth"
RESNET34_SHA_PREFIX = "b627a593"
DEC_CHANNELS = (256, 128, 64, 32, 16)


# =============================================================================================
# 1. ResNet-34 编码器（参数名与 torchvision 一致，便于直接加载 ImageNet 权重）
# =============================================================================================
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + idt)


def _layer(cin, cout, n, stride):
    return nn.Sequential(BasicBlock(cin, cout, stride), *[BasicBlock(cout, cout) for _ in range(n - 1)])


class ResNet34Encoder(nn.Module):
    channels = (64, 64, 128, 256, 512)

    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = _layer(64, 64, 3, 1)
        self.layer2 = _layer(64, 128, 4, 2)
        self.layer3 = _layer(128, 256, 6, 2)
        self.layer4 = _layer(256, 512, 3, 2)

    def forward(self, x):
        e0 = self.relu(self.bn1(self.conv1(x)))      # 1/2，64
        e1 = self.layer1(self.maxpool(e0))           # 1/4，64
        e2 = self.layer2(e1)                         # 1/8，128
        e3 = self.layer3(e2)                         # 1/16，256
        e4 = self.layer4(e3)                         # 1/32，512
        return [e0, e1, e2, e3, e4]


def adapt_imagenet_state(state, in_channels):
    """torchvision resnet34 的 state_dict → 本编码器：去掉 fc；首层卷积 RGB 权重求和（再按输入通道数分摊）"""
    out = {k: v for k, v in state.items() if not k.startswith("fc.")}
    w = out["conv1.weight"]                          # (64,3,7,7)
    out["conv1.weight"] = w.sum(1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels
    return out


def load_imagenet_weights(encoder, path=None, log=print):
    """加载 ImageNet 权重；返回描述字符串。失败时抛出异常（除非 PDU_ALLOW_SCRATCH=1）。"""
    path = path or os.environ.get("PDU_RESNET34_WEIGHTS", "")
    try:
        if path:
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            if not sha.startswith(RESNET34_SHA_PREFIX):
                raise RuntimeError(f"{path} 的 SHA-256 为 {sha[:12]}…，不是 resnet34-{RESNET34_SHA_PREFIX}.pth")
            state = torch.load(path, map_location="cpu")
            src = f"{path}（sha256 {sha[:12]}…）"
        else:
            state = torch.hub.load_state_dict_from_url(RESNET34_URL, map_location="cpu", check_hash=True, progress=False)
            src = f"{RESNET34_URL}（torch.hub，已校验哈希前缀 {RESNET34_SHA_PREFIX}）"
        missing, unexpected = encoder.load_state_dict(adapt_imagenet_state(state, encoder.in_channels), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"权重键不匹配：缺 {missing[:5]}，多 {unexpected[:5]}")
        return "ImageNet 预训练 ResNet-34：" + src
    except Exception as e:  # noqa: BLE001
        if os.environ.get("PDU_ALLOW_SCRATCH", "0") == "1":
            log(f"[PDU-Net] 警告：ImageNet 权重加载失败（{e}），PDU_ALLOW_SCRATCH=1，编码器从头训练（偏离开题设计）")
            return "编码器从头训练（ImageNet 权重加载失败，偏离开题设计）"
        raise RuntimeError(f"[PDU-Net] ImageNet 权重加载失败：{e}\n"
                           f"请先下载 {RESNET34_URL}，用环境变量 PDU_RESNET34_WEIGHTS 指定其路径后重试") from e


# =============================================================================================
# 2. PANet 自底向上路径聚合、双重注意力、解码器
# =============================================================================================
def _cbr(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BottomUpPANet(nn.Module):
    """N0 = e0；N(i+1) = smooth( down(Ni) + lateral(e(i+1)) )"""

    def __init__(self, channels):
        super().__init__()
        self.down = nn.ModuleList([_cbr(channels[i], channels[i + 1], 3, 2) for i in range(len(channels) - 1)])
        self.lateral = nn.ModuleList([_cbr(channels[i + 1], channels[i + 1], 1) for i in range(len(channels) - 1)])
        self.smooth = nn.ModuleList([_cbr(channels[i + 1], channels[i + 1], 3) for i in range(len(channels) - 1)])

    def forward(self, feats):
        outs = [feats[0]]
        for i in range(len(feats) - 1):
            outs.append(self.smooth[i](self.down[i](outs[-1]) + self.lateral[i](feats[i + 1])))
        return outs


class DualAttention(nn.Module):
    """通道注意力（SE 式）+ 空间注意力（CBAM 式）。mode="parallel"：两组权重都由 x 计算；"serial"：先通道后空间。"""

    def __init__(self, channels, reduction=16, kernel_size=7, mode="parallel"):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, mid), nn.ReLU(inplace=True), nn.Linear(mid, channels))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.mode = mode
        if mode not in ("parallel", "serial"):
            raise ValueError(f"DAN 组合方式只能是 parallel 或 serial，收到 {mode}")

    def channel_weight(self, x):
        return torch.sigmoid(self.fc(x.mean((2, 3))))[:, :, None, None]

    def spatial_weight(self, x):
        return torch.sigmoid(self.spatial(torch.cat([x.amax(1, keepdim=True), x.mean(1, keepdim=True)], 1)))

    def forward(self, x):
        ca = self.channel_weight(x)
        if self.mode == "parallel":
            sa = self.spatial_weight(x)
            return x * ca * sa
        y = x * ca
        return y * self.spatial_weight(y)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout, dan_mode="parallel"):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.att = DualAttention(cout + cskip, mode=dan_mode) if cskip else None
        self.fuse = nn.Sequential(_cbr(cout + cskip, cout), _cbr(cout, cout))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = self.att(torch.cat([x, skip], 1))
        return self.fuse(x)


class PDUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, dec_channels=DEC_CHANNELS, dan_mode="parallel"):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels)
        ch = ResNet34Encoder.channels
        self.panet = BottomUpPANet(ch)
        skips = (ch[3], ch[2], ch[1], ch[0], 0)
        blocks, cin = [], ch[4]
        for cs, co in zip(skips, dec_channels):
            blocks.append(DecoderBlock(cin, cs, co, dan_mode))
            cin = co
        self.decoder = nn.ModuleList(blocks)
        self.decoder.deep_supervision = False        # 与 nnU-Net 网络的接口保持一致（本网络无深监督）
        self.head = nn.Conv2d(dec_channels[-1], num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        n = self.panet(self.encoder(x))              # [N0..N4]
        y = n[4]
        for blk, skip in zip(self.decoder, (n[3], n[2], n[1], n[0], None)):
            y = blk(y, skip)
        return self.head(y)[..., :h, :w]


# =============================================================================================
# 3. 损失、学习率
# =============================================================================================
class DiceFocalLoss(nn.Module):
    """L = w_dice·(1 − 前景平均 soft Dice) + w_focal·Focal。target：(B,1,H,W) 整数标签（nnU-Net 格式）。"""

    def __init__(self, batch_dice=True, w_dice=0.6, w_focal=0.4, alpha=0.25, gamma=2.0, smooth=1e-5, ignore_label=None):
        super().__init__()
        self.batch_dice, self.w_dice, self.w_focal = batch_dice, w_dice, w_focal
        self.alpha, self.gamma, self.smooth, self.ignore_label = alpha, gamma, smooth, ignore_label

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):        # 防御：深监督输出只取全分辨率
            logits, target = logits[0], target[0]
        logits = logits.float()
        t = target[:, 0].long()
        valid = torch.ones_like(t, dtype=torch.bool)
        if self.ignore_label is not None:
            valid = t != self.ignore_label
            t = torch.where(valid, t, torch.zeros_like(t))
        logp = F.log_softmax(logits, 1)
        p = logp.exp()
        onehot = torch.zeros_like(p).scatter_(1, t[:, None], 1.0)
        m = valid[:, None].float()
        axes = [0] + list(range(2, p.ndim)) if self.batch_dice else list(range(2, p.ndim))
        inter = (p * onehot * m).sum(axes)
        denom = (p * m).sum(axes) + (onehot * m).sum(axes)
        dice = (2 * inter + self.smooth) / (denom + self.smooth).clamp_min(1e-8)
        dice_fg = dice[..., 1:].mean() if dice.shape[-1] > 1 else dice.mean()
        logpt = (logp * onehot).sum(1)
        pt = logpt.exp()
        alpha_t = torch.where(t > 0, torch.full_like(pt, self.alpha), torch.full_like(pt, 1 - self.alpha))
        focal = -(alpha_t * (1 - pt) ** self.gamma * logpt)
        focal = (focal * valid).sum() / valid.sum().clamp_min(1)
        return self.w_dice * (1 - dice_fg) + self.w_focal * focal


class CosineLR:
    """与 nnU-Net 的 PolyLRScheduler 接口相同：每个 epoch 开始时调用 step(当前 epoch)"""

    def __init__(self, optimizer, initial_lr, max_steps):
        self.optimizer, self.initial_lr, self.max_steps, self.ctr = optimizer, initial_lr, max_steps, 0
        self.step(0)

    def lr_at(self, step):
        return 0.5 * self.initial_lr * (1 + math.cos(math.pi * min(step, self.max_steps) / self.max_steps))

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr_at(current_step)


# =============================================================================================
# 4. 训练器
# =============================================================================================
def _unwrap(net):
    for attr in ("module", "_orig_mod"):            # DDP / torch.compile
        while hasattr(net, attr):
            net = getattr(net, attr)
    return net


class nnUNetTrainer_PDUNet(nnUNetTrainer_FreqBase):
    """按开题报告重新实现的 PDU-Net；种子用 DME_SEED（同 FreqBase）。不重写 __init__。"""
    EPOCHS, ITERS = 250, 250
    BATCH = 8
    LR = 1e-4
    LOSS_W = (0.6, 0.4)
    FOCAL = (0.25, 2.0)

    def initialize(self):
        first = not self.was_initialized
        self.enable_deep_supervision = False
        cm = self.configuration_manager
        if not isinstance(getattr(cm, "configuration", None), dict):
            raise RuntimeError("无法设置批量：ConfigurationManager 没有 configuration 字典（nnU-Net 版本不符？）")
        cm.configuration["batch_size"] = self.BATCH
        super().initialize()
        if first:
            src = load_imagenet_weights(_unwrap(self.network).encoder, log=self.print_to_log_file)
            self.print_to_log_file(f"[PDU-Net] {src}；DAN={_unwrap(self.network).decoder[0].att.mode}；"
                                   f"batch={self.batch_size}；Adam lr={self.LR} 余弦；loss={self.LOSS_W[0]}·Dice+"
                                   f"{self.LOSS_W[1]}·Focal(α={self.FOCAL[0]},γ={self.FOCAL[1]})；无深监督")

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        # 不使用 nnU-Net plans 里的网络结构；推理端调用时传入 enable_deep_supervision=False
        return PDUNet(num_input_channels, num_output_channels, dan_mode="parallel")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)
        return optimizer, CosineLR(optimizer, self.LR, self.num_epochs)

    def _build_loss(self):
        lm = self.label_manager
        if lm.has_regions:
            raise RuntimeError("PDU-Net 配置不支持基于区域的标签")
        return DiceFocalLoss(batch_dice=self.configuration_manager.batch_dice, w_dice=self.LOSS_W[0],
                             w_focal=self.LOSS_W[1], alpha=self.FOCAL[0], gamma=self.FOCAL[1],
                             ignore_label=lm.ignore_label if lm.has_ignore_label else None)

    def set_deep_supervision_enabled(self, enabled: bool):
        pass                                          # 本网络只有一个输出


class nnUNetTrainer_PDUNetSerial(nnUNetTrainer_PDUNet):
    """早期稿件的写法：DAN 串行（先通道后空间）。默认不跑，仅在需要说明 DAN 写法影响时使用。"""

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return PDUNet(num_input_channels, num_output_channels, dan_mode="serial")


class nnUNetTrainer_PDUNetSmoke(nnUNetTrainer_PDUNet):
    """上云冒烟：2 epochs × 10 iterations，验证训练—验证—推理全流程和 ImageNet 权重加载"""
    EPOCHS, ITERS = 2, 10

    def initialize(self):
        super().initialize()
        self.num_val_iterations_per_epoch = 5
