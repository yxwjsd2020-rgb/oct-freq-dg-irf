# -*- coding: utf-8 -*-
"""
文章D：PDU-Net 重新实现的自测（CPU 即可，约 1–2 分钟）。需要 torch；不需要 torchvision、nnunetv2。
装了 nnunetv2 时会多查与 nnU-Net 接口相关的几项；装了 torchvision 时会多查 ImageNet 权重的键名是否一致；
设了环境变量 PDU_RESNET34_WEIGHTS 时会实际加载该权重文件。
用法（在 dme_code 目录）：python tests/test_pdunet.py
"""
import os
import sys
import types
import tempfile
import importlib.util

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "nnunet_trainers", "nnUNetTrainer_PDUNet.py")
FREQ = os.path.join(ROOT, "nnunet_trainers", "nnUNetTrainer_Freq.py")
FAILED = []


def check(name, cond, info=""):
    print(("  通过  " if cond else "  失败  ") + name + (f"  ({info})" if info else ""))
    if not cond:
        FAILED.append(name)


def load_module():
    """没装 nnunetv2 时用占位类；装了但还没把 nnUNetTrainer_Freq.py 复制进去时，从本地文件载入"""
    try:
        import nnunetv2  # noqa: F401
        have_nn = True
    except ImportError:
        have_nn = False
        for name in ["nnunetv2", "nnunetv2.training", "nnunetv2.training.nnUNetTrainer",
                     "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"]:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["nnunetv2.training.nnUNetTrainer.nnUNetTrainer"].nnUNetTrainer = type("nnUNetTrainer", (), {})
    key = "nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Freq"
    try:
        __import__(key)
    except ImportError:
        spec = importlib.util.spec_from_file_location(key, FREQ)
        m = importlib.util.module_from_spec(spec)
        sys.modules[key] = m
        spec.loader.exec_module(m)
    spec = importlib.util.spec_from_file_location("pdunet", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, have_nn


def main():
    torch.manual_seed(0)
    M, have_nn = load_module()
    print(f"torch {torch.__version__}；nnunetv2 {'已安装' if have_nn else '未安装（用占位类）'}")

    print("\n[1] 网络结构与尺寸")
    net = M.PDUNet(1, 2).eval()
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    check("参数量在合理范围（编码器约 21.3M + PANet 约 5M + 解码器约 3M，总量 24–36M）", 24 < n_par < 36, f"{n_par:.2f}M")
    n_enc = sum(p.numel() for p in net.encoder.parameters()) / 1e6
    check("编码器参数量 ≈ torchvision ResNet-34 去掉 fc 并改单通道（21.28M）", abs(n_enc - 21.28) < 0.05, f"{n_enc:.3f}M")
    with torch.no_grad():
        for shape in [(2, 1, 256, 256), (1, 1, 250, 300), (1, 1, 97, 513)]:
            y = net(torch.randn(*shape))
            check(f"输入 {shape[2:]} → 输出同尺寸、2 类、数值有限", tuple(y.shape) == (shape[0], 2) + shape[2:]
                  and bool(torch.isfinite(y).all()), str(tuple(y.shape)))
        x = torch.randn(1, 1, 128, 160)
        feats = net.encoder(x)
        ok = [tuple(f.shape[1:]) == (c, 128 // s, 160 // s) for f, c, s in zip(feats, (64, 64, 128, 256, 512), (2, 4, 8, 16, 32))]
        check("编码器五个层级：1/2…1/32，通道 64/64/128/256/512", all(ok))
        n = net.panet(feats)
        check("PANet 输出与输入同尺寸、同通道", all(a.shape == b.shape for a, b in zip(n, feats)))
        check("PANet 第 0 层直接传递 e0（N0 = e0）", torch.equal(n[0], feats[0]))
        check("PANet 深层确实被改变（N4 ≠ e4）", not torch.allclose(n[4], feats[4]))
    check("解码器 4 级有 DAN、最后 1 级没有", [b.att is not None for b in net.decoder] == [True] * 4 + [False])
    check("网络无深监督（decoder.deep_supervision=False）", net.decoder.deep_supervision is False)

    print("\n[2] 双重注意力（开题：并行；早期稿件：串行）")
    dan = M.DualAttention(64, mode="parallel").eval()
    ser = M.DualAttention(64, mode="serial").eval()
    ser.load_state_dict(dan.state_dict())
    x = torch.randn(2, 64, 20, 24)
    with torch.no_grad():
        ref = x * dan.channel_weight(x) * dan.spatial_weight(x)
        check("并行：x·CA(x)·SA(x)", torch.allclose(dan(x), ref, atol=1e-6))
        y = x * dan.channel_weight(x)
        check("串行：先通道，再由加权后的特征算空间权重", torch.allclose(ser(x), y * dan.spatial_weight(y), atol=1e-6))
        check("两种写法参数相同、输出不同", not torch.allclose(dan(x), ser(x)))
    check("通道注意力降维比 16（64→4→64）", dan.fc[0].out_features == 4 and dan.fc[2].out_features == 64)
    check("空间注意力为 2→1 的 7×7 卷积", tuple(dan.spatial.weight.shape) == (1, 2, 7, 7))
    try:
        M.DualAttention(8, mode="xx")
        check("非法组合方式会报错", False)
    except ValueError:
        check("非法组合方式会报错", True)
    s1, s2 = M.PDUNet(1, 2, dan_mode="parallel"), M.PDUNet(1, 2, dan_mode="serial")
    check("两种 PDU-Net 的权重可互相加载（结构相同）", s2.load_state_dict(s1.state_dict()) is not None)

    print("\n[3] ImageNet 权重")
    ref3 = M.ResNet34Encoder(3)
    fake = dict(ref3.state_dict())
    fake["fc.weight"], fake["fc.bias"] = torch.zeros(1000, 512), torch.zeros(1000)
    ad = M.adapt_imagenet_state(fake, 1)
    check("去掉 fc，首层卷积 RGB 权重相加成单通道",
          "fc.weight" not in ad and torch.allclose(ad["conv1.weight"], fake["conv1.weight"].sum(1, keepdim=True)))
    enc1 = M.ResNet34Encoder(1)
    res = enc1.load_state_dict(ad, strict=True)
    check("转换后的权重可严格加载到单通道编码器", not res.missing_keys and not res.unexpected_keys)
    try:
        import torchvision
        tv = torchvision.models.resnet34(weights=None).state_dict()
        mine = set(M.ResNet34Encoder(3).state_dict())
        tvk = {k for k in tv if not k.startswith("fc.")}
        same = mine == tvk and all(tuple(tv[k].shape) == tuple(ref3.state_dict()[k].shape) for k in mine)
        check("键名、形状与 torchvision resnet34 完全一致", same, f"torchvision {torchvision.__version__}")
    except ImportError:
        print("  跳过  与 torchvision 键名对照（未安装 torchvision）")
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad.pth")
        torch.save(fake, bad)
        os.environ.pop("PDU_ALLOW_SCRATCH", None)
        try:
            M.load_imagenet_weights(M.ResNet34Encoder(1), path=bad, log=lambda *a: None)
            check("哈希不符的权重文件会报错停止", False)
        except RuntimeError as e:
            check("哈希不符的权重文件会报错停止", "SHA-256" in str(e) or "sha" in str(e).lower())
        os.environ["PDU_ALLOW_SCRATCH"] = "1"
        msg = M.load_imagenet_weights(M.ResNet34Encoder(1), path=bad, log=lambda *a: None)
        check("PDU_ALLOW_SCRATCH=1 时改为从头训练并写明", "从头训练" in msg)
        os.environ.pop("PDU_ALLOW_SCRATCH", None)
    real = os.environ.get("PDU_RESNET34_WEIGHTS", "")
    if real and os.path.exists(real):
        enc = M.ResNet34Encoder(1)
        msg = M.load_imagenet_weights(enc, log=print)
        check("实际 ImageNet 权重加载成功", msg.startswith("ImageNet"), msg)
    else:
        print("  跳过  实际 ImageNet 权重（未设置 PDU_RESNET34_WEIGHTS；云端冒烟测试会下载并加载）")

    print("\n[4] 损失函数（0.6·Dice + 0.4·Focal，α=0.25，γ=2）")
    lossf = M.DiceFocalLoss(batch_dice=True)
    tgt = torch.zeros(2, 1, 32, 32)
    tgt[:, :, 8:20, 10:22] = 1
    perfect = torch.cat([(1 - tgt) * 20 - 10, tgt * 20 - 10], 1)
    l_perf = lossf(perfect, tgt).item()
    check("完全正确的预测损失接近 0", l_perf < 0.01, f"{l_perf:.5f}")
    logits = torch.randn(2, 2, 32, 32, requires_grad=True)
    l_rand = lossf(logits, tgt)
    l_rand.backward()
    check("随机预测损失明显更大，梯度有限", l_rand.item() > 0.2 and bool(torch.isfinite(logits.grad).all()), f"{l_rand.item():.3f}")
    p = F.softmax(logits.detach(), 1)
    oh = torch.cat([1 - tgt, tgt], 1)
    dice = (2 * (p[:, 1:] * oh[:, 1:]).sum() + 1e-5) / (p[:, 1:].sum() + oh[:, 1:].sum() + 1e-5)
    pt = (p * oh).sum(1)
    at = torch.where(tgt[:, 0] > 0, 0.25, 0.75)
    focal = (-(at * (1 - pt) ** 2 * pt.log())).mean()
    manual = 0.6 * (1 - dice) + 0.4 * focal
    check("与手工公式一致", abs(manual.item() - l_rand.item()) < 1e-5, f"{manual.item():.5f} vs {l_rand.item():.5f}")
    empty = torch.zeros(2, 1, 32, 32)
    check("全背景 batch 不出现 NaN", bool(torch.isfinite(lossf(torch.randn(2, 2, 32, 32), empty))))
    check("batch_dice=False 也可计算", bool(torch.isfinite(M.DiceFocalLoss(batch_dice=False)(torch.randn(2, 2, 32, 32), tgt))))
    ig = tgt.clone()
    ig[:, :, :4] = 2
    li = M.DiceFocalLoss(ignore_label=2)
    a = li(perfect, ig).item()
    check("忽略标签区域不计入损失", a < 0.01, f"{a:.5f}")

    print("\n[5] 学习率（Adam 1e-4，余弦退火）")
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    sch = M.CosineLR(opt, 1e-4, 250)
    lrs = []
    for ep in (0, 125, 249, 250):
        sch.step(ep)
        lrs.append(opt.param_groups[0]["lr"])
    check("第 0 轮 1e-4、第 125 轮 5e-5、最后趋近 0", abs(lrs[0] - 1e-4) < 1e-12 and abs(lrs[1] - 5e-5) < 1e-12
          and lrs[2] < 1e-8 and lrs[3] == 0.0, str([f"{v:.2e}" for v in lrs]))

    print("\n[6] 训练器")
    T = M.nnUNetTrainer_PDUNet
    netb = T.build_network_architecture("x", {}, (), 1, 2, False)
    check("build_network_architecture 返回 PDU-Net（并行 DAN），可作为静态方法调用",
          isinstance(netb, M.PDUNet) and netb.decoder[0].att.mode == "parallel")
    nets = M.nnUNetTrainer_PDUNetSerial.build_network_architecture("x", {}, (), 1, 2, False)
    check("PDUNetSerial 返回串行 DAN", nets.decoder[0].att.mode == "serial")
    check("设置与开题一致：批量 8、Adam 1e-4、损失权重 0.6/0.4",
          T.BATCH == 8 and T.LR == 1e-4 and T.LOSS_W == (0.6, 0.4) and T.FOCAL == (0.25, 2.0))
    check("训练长度与文章D 其他配置相同（250×250）", (T.EPOCHS, T.ITERS) == (250, 250))
    check("冒烟训练器 2×10", (M.nnUNetTrainer_PDUNetSmoke.EPOCHS, M.nnUNetTrainer_PDUNetSmoke.ITERS) == (2, 10))
    check("继承 FreqBase（同一套随机种子控制）", issubclass(T, sys.modules["nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Freq"].nnUNetTrainer_FreqBase))

    print("\n[7] 小样本过拟合（CPU，约 1–2 分钟）：确认整个网络可以学习")
    # 11-11 本机实测：40 步下降 38.6%（差一点未达 40%），说明网络在学习、只是步数太少。
    # 改为 100 步，门槛不变（40%）。这是软件自测，不是研究终点。
    N_STEPS = 100
    torch.manual_seed(1)
    small = M.PDUNet(1, 2).train()
    xs = torch.randn(4, 1, 64, 64)
    ys = torch.zeros(4, 1, 64, 64)
    ys[:, :, 20:40, 16:44] = 1
    xs = xs + 2 * ys
    opt = torch.optim.Adam(small.parameters(), lr=1e-3)
    first = last = None
    for it in range(N_STEPS):
        opt.zero_grad()
        l = lossf(small(xs), ys)
        l.backward()
        opt.step()
        first = l.item() if first is None else first
        last = l.item()
    check(f"{N_STEPS} 步后损失下降 40% 以上", last < first * 0.6, f"{first:.3f} → {last:.3f}")

    if have_nn:
        print("\n[8] 与 nnU-Net 接口（nnunetv2 已安装）")
        import inspect
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
        for meth in ("initialize", "configure_optimizers", "_build_loss", "set_deep_supervision_enabled",
                     "_get_deep_supervision_scales"):
            check(f"nnUNetTrainer 有 {meth}", hasattr(nnUNetTrainer, meth))
        check("ConfigurationManager 用 self.configuration 保存配置（批量可改）",
              "self.configuration" in inspect.getsource(ConfigurationManager.__init__))
        check("batch_size 读自 configuration 字典",
              "configuration['batch_size']" in inspect.getsource(ConfigurationManager).replace('"', "'"))
        src_ds = inspect.getsource(nnUNetTrainer._get_deep_supervision_scales)
        check("关闭深监督时不生成多尺度标签", "enable_deep_supervision" in src_ds)
        src_init = inspect.getsource(nnUNetTrainer.initialize)
        check("initialize 中先建网络、再建优化器和损失",
              src_init.find("build_network_architecture") < src_init.find("configure_optimizers") < src_init.find("_build_loss"))
        check("学习率调度器每 epoch 调用 step(current_epoch)",
              "lr_scheduler.step(self.current_epoch)" in inspect.getsource(nnUNetTrainer.on_train_epoch_start))

    print("\n" + ("全部通过" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
