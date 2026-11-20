# -*- coding: utf-8 -*-
"""
文章D 频域模块自测（CPU 即可，约 1 分钟）。需要 torch；若装了 nnunetv2 / dynamic_network_architectures 会多测网络装配。
用法（在 dme_code 目录）：python tests/test_freq.py
"""
import os
import sys
import types
import importlib.util

if hasattr(sys.stdout, "reconfigure"):          # Windows 控制台/重定向时避免中文乱码
    sys.stdout.reconfigure(encoding="utf-8")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "nnunet_trainers", "nnUNetTrainer_Freq.py")


def load_module():
    """没有安装 nnunetv2 时，用一个空的 nnUNetTrainer 占位，以便单独测试各模块"""
    try:
        import nnunetv2  # noqa: F401
        have_nn = True
    except ImportError:
        have_nn = False
        for name in ["nnunetv2", "nnunetv2.training", "nnunetv2.training.nnUNetTrainer",
                     "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"]:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["nnunetv2.training.nnUNetTrainer.nnUNetTrainer"].nnUNetTrainer = type("nnUNetTrainer", (), {})
    spec = importlib.util.spec_from_file_location("freq", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, have_nn


def check(name, cond, info=""):
    print(("  通过  " if cond else "  失败  ") + name + (f"  ({info})" if info else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def main():
    torch.manual_seed(0)
    M, have_nn = load_module()
    print("1) Haar 小波分解—重建")
    haar = M.HaarWT()
    x = torch.randn(2, 3, 10, 14)
    err = (haar.iwt(haar.wt(x)) - x).abs().max().item()
    check("完全重建", err < 1e-5, f"最大误差 {err:.1e}")

    print("2) WTConv2d")
    for h, w in [(48, 64), (37, 51), (5, 7), (3, 3)]:
        m = M.WTConv2d(8, 5, 3)
        x = torch.randn(2, 8, h, w, requires_grad=True)
        y = m(x)
        y.mean().backward()
        check(f"尺寸 {h}x{w} 输出形状不变、可反传", y.shape == x.shape and x.grad is not None and torch.isfinite(y).all())
    m = M.WTConv2d(4, 5, 3)
    with torch.no_grad():
        m.base_conv.weight.zero_(); m.base_conv.weight[:, :, 2, 2] = 1; m.base_conv.bias.zero_()
        for s in m.wavelet_scale:
            s.zero_()
    x = torch.randn(1, 4, 33, 40)
    check("小波分支置零 + 中心为 1 的核 → 恒等映射", torch.allclose(m(x), x, atol=1e-5))
    m = M.WTConv2d(4, 5, 2)
    with torch.no_grad():
        m.base_scale.zero_()
        for c in m.wavelet_convs:
            c.weight.zero_(); c.weight[:, :, 2, 2] = 1
        for s in m.wavelet_scale:
            s.fill_(1.0)
    x = torch.randn(1, 4, 32, 32)
    y = m(x)
    # 2 级、各子带恒等：重建 = 第1级全部子带 + 第2级 LL 叠加到第1级 LL 上 → x + 上采样的 LL2 分量（非零即可，检验级联通路生效）
    check("小波级联通路生效", (y - x).abs().mean() > 1e-3 and torch.isfinite(y).all())

    print("3) 全局频域分支（FFC 谱变换）")
    g = M.GlobalSpectralBranch(16)
    x = torch.randn(2, 16, 24, 30)
    check("γ=0 时恒等", torch.allclose(g(x), x))
    with torch.no_grad():
        g.gamma.fill_(1.0)
    y = g(x)
    y.sum().backward()
    check("γ=1 时形状不变、可反传", y.shape == x.shape and g.fu.conv.weight.grad is not None)

    print("4) 随机幅度谱合成 RASS / 频谱斜率 / RMS")
    x = torch.randn(4, 1, 64, 96) * 2 + 1
    y = M.rass(x)
    check("RASS 形状、有限值", y.shape == x.shape and torch.isfinite(y).all())
    check("RASS 保持均值和标准差", torch.allclose(y.mean((-2, -1)), x.mean((-2, -1)), atol=1e-3)
          and torch.allclose(y.std((-2, -1)), x.std((-2, -1)), atol=1e-3))
    check("α=β=0 时不改变图像", torch.allclose(M.rass(x, 0.0, 0.0, 2.0), x, atol=1e-4))
    r = M._radius(64, 96, "cpu")
    check("最大归一化频率约 0.5", abs(r.max().item() - 0.5) < 0.01, f"{r.max().item():.3f}")
    sigma = (2 * 3.0 * r) ** 2.0 + 0.25
    check("σ 范围 0.25–9.25（原文 α=3, β=0.25, γ=2）", abs(sigma.min() - 0.25) < 1e-6 and sigma.max() < 9.26,
          f"{sigma.min():.2f}–{sigma.max():.2f}")
    t = M.spec_tilt(x)
    check("频谱斜率 形状、有限值", t.shape == x.shape and torch.isfinite(t).all())
    xi = torch.randn(1, 50, 60)
    yi = M.random_mask_shuffle(xi)
    check("RMS 只打乱位置、不改变像素值集合", torch.allclose(yi.flatten().sort().values, xi.flatten().sort().values))
    d = torch.randn(6, 1, 32, 32).half() if torch.cuda.is_available() else torch.randn(6, 1, 32, 32)
    out = M.freq_augment(d, True, True, True)
    check("freq_augment 保持 dtype 和形状", out.dtype == d.dtype and out.shape == d.shape)

    print("5) 装进 nnU-Net 网络")
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
    except ImportError:
        print("  跳过：未安装 dynamic_network_architectures（云端安装 nnunetv2 后会自动有）")
        PlainConvUNet = None
    if PlainConvUNet is not None:
        def net():
            return PlainConvUNet(1, 7, [32, 64, 128, 256, 512, 512, 512], torch.nn.Conv2d, 3, [1, 2, 2, 2, 2, 2, 2], 2, 2,
                                 [2] * 6, True, torch.nn.InstanceNorm2d, {"eps": 1e-5, "affine": True}, None, None,
                                 torch.nn.LeakyReLU, {"inplace": True}, deep_supervision=True)
        x = torch.randn(2, 1, 256, 320)
        base = net()
        n0 = sum(p.numel() for p in base.parameters())
        for name, fn in [("WTConv", M.add_wtconv), ("FFC", M.add_ffc)]:
            n = fn(net())
            outs = n(x)
            n1 = sum(p.numel() for p in n.parameters())
            check(f"{name}：前向（深监督 {len(outs)} 个输出）", outs[0].shape == (2, 2, 256, 320), f"参数 {n0/1e6:.1f}M → {n1/1e6:.1f}M")
        print("  （FFC γ=0 初始等价性已在第 3 步单独验证）")
    if have_nn:
        print("6) nnU-Net 训练器类")
        for c in ["nnUNetTrainer_FreqBase", "nnUNetTrainer_RASS", "nnUNetTrainer_RAS", "nnUNetTrainer_WTConv",
                  "nnUNetTrainer_FFC", "nnUNetTrainer_RAS_WTConv", "nnUNetTrainer_SpecTilt", "nnUNetTrainer_FreqSmoke"]:
            check(f"{c} 可导入", hasattr(M, c))
        if PlainConvUNet is not None:
            kw = {"n_stages": 7, "features_per_stage": [32, 64, 128, 256, 512, 512, 512], "conv_op": "torch.nn.modules.conv.Conv2d",
                  "kernel_sizes": [[3, 3]] * 7, "strides": [[1, 1]] + [[2, 2]] * 6, "n_conv_per_stage": [2] * 7,
                  "n_conv_per_stage_decoder": [2] * 6, "conv_bias": True,
                  "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d", "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                  "dropout_op": None, "dropout_op_kwargs": None, "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True}}
            req = ["conv_op", "norm_op", "dropout_op", "nonlin"]
            for c in ["nnUNetTrainer_WTConv", "nnUNetTrainer_FFC", "nnUNetTrainer_FreqSmoke"]:
                n = getattr(M, c).build_network_architecture(
                    "dynamic_network_architectures.architectures.unet.PlainConvUNet", kw, req, 1, 2, False)
                check(f"{c}.build_network_architecture（推理时的调用方式）", n(torch.randn(1, 1, 128, 192)).shape == (1, 2, 128, 192))
    print("=" * 40)
    print("全部通过" if check.failed == 0 else f"{check.failed} 项失败，请把以上输出发给 Claude")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
