#!/usr/bin/env bash
# 文章D：频域模块训练与外部测试（阿里云 PAI-DSW，A10）
# ----------------------------------------------------------------------------------
# 用法（DSW 终端，cd /mnt/workspace/dme_code）：
#   bash gpu/run_freq.sh smoke                       # ① 先跑：自测 + 2 epochs 冒烟训练与推理（约 15 分钟）
#   nohup bash gpu/run_freq.sh train  > /mnt/workspace/freq_train_log.txt 2>&1 &    # ② 训练（主要 + 次要配置）
#   bash gpu/run_freq.sh predict                     # ③ 全部配置训练完、预注册提交后，才对外部测试集推理（只跑一次）
#   查看进度：tail -5 /mnt/workspace/freq_train_log.txt
# 只跑部分配置：FREQ_RUNS="nnUNetTrainer_RAS:1 nnUNetTrainer_RAS:2" bash gpu/run_freq.sh train
#
# 设计（见 文章D_代码说明.md）：所有模型都在 TowardPi 第 0 折（约 80% 数据）上训练，第 0 折验证集给出内部 Dice；
#   基线种子 1 直接复用文章B 已训练好的 nnUNetTrainer_250epochs fold_0（需把模型包解压回 nnUNet_results）。
#   不同种子写入不同的 nnUNet_results 目录：work/nnUNet_results_freq/seed<S>。
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CODE="$ROOT/dme_code"
export DME_WORK="$ROOT/work" PYTHONUNBUFFERED=1 nnUNet_compile=f
export TORCH_HOME="$DME_WORK/torch_home"     # PDU-Net 的 ImageNet 权重缓存在 work/ 下，实例重启后仍在
export nnUNet_raw="$DME_WORK/nnUNet_raw" nnUNet_preprocessed="$DME_WORK/nnUNet_preprocessed"
BASE_RESULTS="$DME_WORK/nnUNet_results"
FREQ_RESULTS="$DME_WORK/nnUNet_results_freq"
PRED="$DME_WORK/pred/freq"
NCPU=$(nproc); export nnUNet_n_proc_DA=$(( NCPU > 4 ? NCPU - 2 : 2 ))
DS=101; TP=Dataset101_TowardPi; CFG=2d; FOLD=0
MODE=${1:-}

# 主要比较（预注册 H1）：RAS vs 基线，各 3 个种子；次要（探索性）：各 1 个种子
# nnUNetTrainer_PDUNet：按开题报告重新实现的 PDU-Net（探索性，与开题衔接；ImageNet 预训练 ResNet-34，见 nnUNetTrainer_PDUNet.py）
# 早期稿件写法（DAN 串行）nnUNetTrainer_PDUNetSerial 默认不跑
DEFAULT_RUNS="nnUNetTrainer_RAS:1 nnUNetTrainer_RAS:2 nnUNetTrainer_RAS:3 \
nnUNetTrainer_FreqBase:2 nnUNetTrainer_FreqBase:3 \
nnUNetTrainer_RASS:1 nnUNetTrainer_WTConv:1 nnUNetTrainer_FFC:1 nnUNetTrainer_RAS_WTConv:1 \
nnUNetTrainer_PDUNet:1"
RUNS=${FREQ_RUNS:-$DEFAULT_RUNS}
# 外部测试集（文件夹存在才推理）：Duke、OCTDL 正常眼、RETOUCH 三家设备、本院 Optovue
TEST_SETS="duke octdl retouch_cirrus retouch_spectralis retouch_topcon local"
# SAP 敏感性分析 S7：Spectralis 不做 1/4 次方显示变换的版本（s13_retouch.py --spectralis-nodisp 生成；
# 原始数据不是线性强度时不会生成，自动跳过）。只对主要比较的两组（RAS、基线）推理。不想跑：FREQ_S7=0
S7_SET="retouch_spectralis_nodisp"
S7_TRAINERS="nnUNetTrainer_RAS nnUNetTrainer_FreqBase nnUNetTrainer_250epochs"
[ "${FREQ_S7:-1}" = "1" ] && TEST_SETS="$TEST_SETS $S7_SET"

python -c "import nnunetv2" 2>/dev/null || python -m pip install -q "nnunetv2>=2.5,<2.7"
python -m pip install -q pandas scipy statsmodels pillow tifffile >/dev/null
NNDIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp "$CODE/nnunet_trainers/nnUNetTrainer_Freq.py" "$CODE/nnunet_trainers/nnUNetTrainer_PDUNet.py" \
   "$CODE/nnunet_trainers/nnUNetTrainer_FT100.py" "$NNDIR/training/nnUNetTrainer/"
python -c "import torch, importlib.metadata as m; print('PyTorch', torch.__version__, '| nnunetv2', m.version('nnunetv2'), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

fetch_resnet34() {  # PDU-Net 的 ImageNet 权重（torchvision IMAGENET1K_V1）；已设 PDU_RESNET34_WEIGHTS 时只校验该文件
  python - <<'PY'
import os, hashlib, torch
p = os.environ.get("PDU_RESNET34_WEIGHTS", "")
if not p:
    torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet34-b627a593.pth", check_hash=True, progress=False)
    p = os.path.join(torch.hub.get_dir(), "checkpoints", "resnet34-b627a593.pth")
sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
assert sha.startswith("b627a593"), f"哈希不符：{sha}"
print("ResNet-34 ImageNet 权重：", p, "sha256", sha)
PY
}

preprocess() {
  N_IMG=$(ls "$nnUNet_raw/$TP/imagesTr" | wc -l)
  N_PRE=$(ls "$nnUNet_preprocessed/$TP/nnUNetPlans_$CFG" 2>/dev/null | grep -c '[.]pkl$' || true)
  if [ -f "$nnUNet_preprocessed/$TP/nnUNetPlans.json" ] && [ "$N_PRE" -ge "$N_IMG" ]; then
    echo "预处理已完成（$N_PRE 例），跳过"
  else
    nnUNetv2_plan_and_preprocess -d $DS -c $CFG --verify_dataset_integrity
  fi
  cp "$DME_WORK/splits/towardpi_splits_final.json" "$nnUNet_preprocessed/$TP/splits_final.json"
}

results_dir() {  # $1 trainer  $2 seed → 该模型所在的 nnUNet_results 根目录
  if [ "$1" = "nnUNetTrainer_250epochs" ]; then echo "$BASE_RESULTS"; else echo "$FREQ_RESULTS/seed$2"; fi
}

train_one() {  # $1 trainer  $2 seed
  local R; R=$(results_dir "$1" "$2")
  local D="$R/$TP/${1}__nnUNetPlans__$CFG/fold_$FOLD"
  if [ -f "$D/checkpoint_final.pth" ] && [ -f "$D/validation/summary.json" ]; then echo "$1 种子 $2 已完成，跳过"; return; fi
  local C=""; [ -f "$D/checkpoint_latest.pth" ] && C="--c"
  echo "== 训练 $1 种子 $2 $(date '+%F %T') =="
  nnUNet_results="$R" DME_SEED="$2" nnUNetv2_train $DS $CFG $FOLD -tr "$1" $C
}

predict_one() {  # $1 trainer  $2 seed
  local R; R=$(results_dir "$1" "$2")
  for SET in $TEST_SETS; do
    [ -d "$DME_WORK/$SET/images" ] || continue
    if [ "$SET" = "$S7_SET" ] && ! echo " $S7_TRAINERS " | grep -q " $1 "; then continue; fi   # S7 只比较 RAS 与基线
    local O="$PRED/${1}_s$2/$SET"
    [ -f "$O/.done" ] && continue
    nnUNet_results="$R" nnUNetv2_predict -i "$DME_WORK/$SET/images" -o "$O" -d $DS -c $CFG -f $FOLD -tr "$1" -chk checkpoint_final.pth
    touch "$O/.done"
  done
}

case "$MODE" in
  smoke)
    python "$CODE/tests/test_freq.py"
    python "$CODE/tests/test_pdunet.py"
    fetch_resnet34
    preprocess
    rm -rf "$FREQ_RESULTS/smoke" "$PRED/smoke"
    nnUNet_results="$FREQ_RESULTS/smoke" DME_SEED=1 nnUNetv2_train $DS $CFG $FOLD -tr nnUNetTrainer_FreqSmoke
    mkdir -p "$PRED/smoke/in"; ls "$DME_WORK/duke/images" | head -3 | while read f; do cp "$DME_WORK/duke/images/$f" "$PRED/smoke/in/"; done
    nnUNet_results="$FREQ_RESULTS/smoke" nnUNetv2_predict -i "$PRED/smoke/in" -o "$PRED/smoke/out" -d $DS -c $CFG -f $FOLD \
        -tr nnUNetTrainer_FreqSmoke -chk checkpoint_final.pth
    ls "$PRED/smoke/out"
    nnUNet_results="$FREQ_RESULTS/smoke" DME_SEED=1 nnUNetv2_train $DS $CFG $FOLD -tr nnUNetTrainer_PDUNetSmoke
    grep -h "PDU-Net" "$FREQ_RESULTS/smoke/$TP/nnUNetTrainer_PDUNetSmoke__nnUNetPlans__$CFG/fold_$FOLD"/training_log_*.txt | tail -2
    nnUNet_results="$FREQ_RESULTS/smoke" nnUNetv2_predict -i "$PRED/smoke/in" -o "$PRED/smoke/out_pdu" -d $DS -c $CFG -f $FOLD \
        -tr nnUNetTrainer_PDUNetSmoke -chk checkpoint_final.pth
    ls "$PRED/smoke/out_pdu"
    echo "== 冒烟测试通过：训练、验证、推理全流程可运行。可以开始 train =="
    ;;
  train)
    echo " $RUNS " | grep -q "PDUNet" && fetch_resnet34
    preprocess
    B="$BASE_RESULTS/$TP/nnUNetTrainer_250epochs__nnUNetPlans__$CFG/fold_$FOLD/checkpoint_final.pth"
    [ -f "$B" ] || echo "[提示] 没找到文章B 基线 fold_0（$B）：请把模型包解压到 work/ 下，或在 FREQ_RUNS 中加入 nnUNetTrainer_FreqBase:1"
    for RS in $RUNS; do train_one "${RS%%:*}" "${RS##*:}"; done
    STAMP=$(date +%Y%m%d_%H%M)
    tar czf "$ROOT/freq_train_$STAMP.tar.gz" --exclude='*.pth' --exclude='*.npz' -C "$DME_WORK" nnUNet_results_freq 2>/dev/null || true
    echo "== 训练完成 $(date '+%F %T')：内部验证结果已打包 freq_train_$STAMP.tar.gz（不含权重）=="
    ;;
  predict)
    echo "外部测试集只应推理一次。确认：全部配置已训练完、预注册/分析计划已定稿？输入 yes 继续"; read -r ok
    [ "$ok" = "yes" ] || { echo "已取消"; exit 1; }
    for RS in nnUNetTrainer_250epochs:1 $RUNS; do predict_one "${RS%%:*}" "${RS##*:}"; done
    python "$CODE/s12_freq_eval.py"
    STAMP=$(date +%Y%m%d_%H%M)
    tar czf "$ROOT/freq_results_$STAMP.tar.gz" --exclude='*.npz' --exclude='smoke' -C "$DME_WORK" \
        results/freq pred/freq 2>/dev/null || true
    echo "== 外部测试完成：请下载 freq_results_$STAMP.tar.gz，然后停止实例 =="
    ;;
  *) sed -n 2,15p "$0" ;;
esac
