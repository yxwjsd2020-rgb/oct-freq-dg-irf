# -*- coding: utf-8 -*-
"""
=====================================================================
 全局配置 —— 整个代码包只需要改这一个文件
=====================================================================
 标 【必改】 的项目必须按实际情况填写；其余一般保持默认。
 公开版说明：本文件与作者另一项登记研究共用；属于该研究的参数（本院数据实验设计、Topcon、小样本微调、
 无标注评估阈值等）已删除，不影响本研究（文章D）的任何分析；完整版待该研究发表后公开。
"""
import os

# ------------------------------------------------------------------
# 1. 路径
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("DME_WORK", os.path.join(ROOT, "work"))   # 所有中间结果和输出

RAW_TOWARDPI = os.environ.get("RAW_TOWARDPI", r"D:\公共数据集\TowardPi\AMD_DME_3D_Dataset")   # 【必改】TowardPi 解压目录
RAW_DUKE     = os.environ.get("RAW_DUKE",     r"D:\公共数据集\Duke_DME")   # 【必改】Duke 2015 DME .mat 文件所在目录
RAW_OCTDL    = os.environ.get("RAW_OCTDL",    r"D:\公共数据集\OCTDL")      # 【必改】OCTDL 解压目录（含 NO 文件夹）
RAW_LOCAL    = os.environ.get("RAW_LOCAL",    r"D:\DME_project\data_raw\local")      # 【必改】本院数据目录，结构见 README

# nnU-Net 需要的三个环境变量（第 00 步脚本会自动设置）
NNUNET_RAW          = os.path.join(WORK, "nnUNet_raw")
NNUNET_PREPROCESSED = os.path.join(WORK, "nnUNet_preprocessed")
NNUNET_RESULTS      = os.path.join(WORK, "nnUNet_results")

# ------------------------------------------------------------------
# 2. 统一的像素间距（所有图像都重采样到这个间距，单位 μm/像素）
# ------------------------------------------------------------------
TARGET_AXIAL_UM   = 3.9     # 轴向（纵向）
TARGET_LATERAL_UM = 11.7    # 横向

# ------------------------------------------------------------------
# 3. TowardPi 数据集（Huang et al., Sci Data 2026；先运行 01_prepare_towardpi.py --inspect 核对）
#    实际解压结构：AMD_DME_3D_Dataset/DME_labeled/input/*.tif 为图像，.../label/*_mask.tif 为掩膜（RAW_TOWARDPI 指向 AMD_DME_3D_Dataset）；
#    文件名 = 病种 + 眼编号(2位) + 图像编号(2位)，如 DME0101.tiff；每个文件是一个 512×512×512 的容积，
#    由原始 512 A-scan × 1044 深度像素的 B-scan 以中央凹为中心裁剪为 512 深度像素。
# ------------------------------------------------------------------
TP_INPUT_DIR = "DME_labeled/input"        # 相对 RAW_TOWARDPI
TP_LABEL_DIR = "DME_labeled/label"
TP_MASK_SUFFIX = "_mask"
TP_NAME_REGEX = r"(?i)^(DME)(\d{2})(\d{2})"   # 病种、眼编号、图像编号
TP_BSCAN_AXIS = 0                    # 【运行 --inspect 后确认】哪个维度是 B-scan 序号
TP_DEPTH_AXIS_IN_BSCAN = 0           # 【运行 --inspect 后确认】B-scan 内哪个维度是深度（0=行，1=列）
TP_AXIAL_UM = 2.7                    # 【作者未回复，经验校准值】2026-10-15 用 s01b_axial_check.py 以 Duke（3.87 μm）为参照，
                                     #   比较中心凹旁 1.5–2.8 mm 无积液处 ILM→RPE 厚度：Duke 301 μm（10 例）÷ TowardPi 112 像素（42 容积）≈ 2.7 μm/像素
                                     #   原先按“6 mm/1044”估算的 5.75 μm（或组织内 4.16 μm）会使厚度达 644/466 μm，不合理
TP_LATERAL_UM = 6000 / 512           # 6 mm / 512 A-scan ≈ 11.72 μm（数据集原文明确）
TP_EVERY_NTH = 4                     # 每隔几张取一张 B-scan
# 文件名只含“眼编号”，不含患者编号：33 例患者 37 只眼，即约 4 例患者双眼入组。
# 默认按眼分组；若作者提供眼—患者对照表，填入下面的 csv（两列：eye_id,patient_id）即可按患者分组。
TP_EYE_PATIENT_CSV = None

# ------------------------------------------------------------------
# 4. Duke DME（Chiu 2015，.mat 格式，Spectralis）
# ------------------------------------------------------------------
DUKE_AXIAL_UM = 3.87                 # Spectralis 轴向像素间距
DUKE_LATERAL_UM = 11.3               # 横向像素间距（μm）：Spectralis 30° 扫描 768 A-scan 的常见值，原文数值待核对【核对 Chiu 2015 原文】

# ------------------------------------------------------------------
# 5. OCTDL（Optovue，JPG）
# ------------------------------------------------------------------
OCTDL_NORMAL_DIR = "NO"              # 正常类所在子文件夹名
OCTDL_SCAN_MM = None                 # 【必改】标称扫描长度（mm），用于换算横向间距；无法确定时设 None 并在方法中说明
OCTDL_AXIAL_UM = None                # 【必改】轴向像素间距（μm）
OCTDL_RESAMPLE = "thickness"         # "thickness"：按正常视网膜厚度推算每幅图的像素间距并等比例缩放；"none"：不缩放（敏感性分析）
OCTDL_NORMAL_THICK_UM = 290          # 正常眼中心凹外 ILM→RPE 厚度假设值（μm），用于上面的推算【可按文献正常值调整】
OCTDL_LATERAL_RATIO = 1.0            # JPG 横向/纵向像素间距之比；1.0 = 假设导出图横纵等比例【查看实际图像后确认】

# ------------------------------------------------------------------
# 6. 本院 Optovue 数据（预处理参数；文章D 中登记为尚未获得的探索性数据集）
# ------------------------------------------------------------------
LOCAL_AXIAL_UM = 2375 / 768         # RTVue XR Raster：深度 2.375 mm / 768 像素 ≈ 3.09 μm（与设备 AScanResolution 3.09208 一致）
LOCAL_RASTER_MM = 12.0               # Raster 扫描宽度 12 mm，1020 个 A-scan → 横向 ≈ 11.76 μm/像素
LOCAL_CROP_MM = 6.0                  # 以中心凹为中心截取的宽度（与 TowardPi 6 mm 视野一致）
LOCAL_SSI_MIN = 35                   # 纳入的最低 SSI（敏感性分析另报 ≥45 的亚组）
# ------------------------------------------------------------------
# 7. nnU-Net 设置
# ------------------------------------------------------------------
DS_TOWARDPI = 101                    # nnU-Net 数据集编号
DS_DUKE_NAME = "Duke"                # 外部测试集只做推理，不需要编号
CONFIG = "2d"
BASE_TRAINER = "nnUNetTrainer_250epochs"       # 基础模型：nnU-Net 自带的 250 epochs 版本（算力充足可改回 "nnUNetTrainer" = 1000 epochs）
PLANS = "nnUNetPlans"

# ------------------------------------------------------------------
# 8. 评估
# ------------------------------------------------------------------
GLOBAL_SEED = 2026
N_BOOT = 2000


# 自测模式（tests/selftest.sh 使用）：填入示例数值，正式运行时不会生效
if os.environ.get("DME_SELFTEST"):
    LOCAL_AXIAL_UM = 3.9
    LOCAL_SSI_MIN = 0
    DUKE_LATERAL_UM = 11.7
    OCTDL_SCAN_MM, OCTDL_AXIAL_UM = None, None


def p(*parts):
    """在 WORK 下拼路径并自动建目录"""
    path = os.path.join(WORK, *parts)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


# ---------------------------------------------------------------- 临床一致性
CENTER_MM = 1.0                      # 中心凹中央区直径（mm）；本院图像以中心凹为中心截取，故取图像中央 1 mm
