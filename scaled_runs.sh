#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scaled_runs.sh  --- 一键调用 TACF 生产级（scale-up）训练的 bash 包装器。
#
# 2026-09-12 v2  按用户新指令：
#   - GPU tier 只保留 24G 和 8G 两档（去掉 16G/12G）。
#   - **每个数据集 d_model, n_layers, agg_d_model, agg_n_layers 完全锁定，
#     24G 和 8G 版本一模一样；只改 physical BS 和 grad-accum_steps**。
#     保证同一数据集不同显卡的训练动力学（BN、梯度噪声）完全等价。
#   - traffic 默认从跑计划里移除：没传 --force-traffic 时 die 报错，提示暂不推荐；
#     但 src/configs/scales/traffic_{24g,8g}.yaml 仍保留。
#   - grad_accum 核心等式 ：
#         BS_eff = BS_phys × accum_steps    (每档统一拉到 40~64)
#         VRAM   ~ BS_phys 线性             (跟 BS_eff 无关)
#
# 使用例子：
#   bash scaled_runs.sh --help
#   bash scaled_runs.sh --dataset ETTh1         --gpu-tier 24G
#   bash scaled_runs.sh --dataset weather       --gpu-tier 8G
#   bash scaled_runs.sh --dataset electricity   --gpu-tier 24G
#   bash scaled_runs.sh --dataset exchange_rate --gpu-tier 8G
#   bash scaled_runs.sh --dataset traffic       --gpu-tier 24G --force-traffic   # 强制跑 traffic
#   # 临时手动覆盖 grad-accum:
#   bash scaled_runs.sh --dataset ETTh1 --gpu-tier 8G --extra "--grad-accum 20"
#   # 关闭 estimate_vram_fast 检查：
#   bash scaled_runs.sh --dataset ETTh1 --gpu-tier 24G --no-vram-check
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

die() { echo "ERROR: $*" >&2 ; exit 2 ; }

# ---------------------------------------------------------------- defaults
DATASET=""
GPU_TIER=""
NO_VRAM_CHECK=0
FORCE_TRAFFIC=0
EXTRA_ARGS=""
PY="${PYTHON:-python}"
# ---------------------------------------------------------------- parser
usage() {
  sed -n '2,120p' "${BASH_SOURCE[0]}" | grep -v '^#' | sed 's/^# \{0,1\}//'
  echo "
Script options:
  --dataset <DS>        (必填) ETTh1|ETTh2|ETTm1|ETTm2|weather|electricity|exchange_rate|traffic
  --gpu-tier <24G|8G>   GPU 显存等级（只有两档，不同档只改 physical BS 和 grad-accum）
  --no-vram-check       跳过 estimate_vram_fast.py 打印
  --force-traffic       traffic 默认不推荐在 early iteration 跑；传此 flag 才放行
  --extra '...'         额外透传给 src.experiments.main 的参数
  --py <bin>            Python 解释器，默认 \$PYTHON 或 python
  -h / --help           Show this help
"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)          usage; exit 0 ;;
    --dataset)          DATASET="$2"; shift 2 ;;
    --gpu-tier)         GPU_TIER="$2"; shift 2 ;;
    --no-vram-check)    NO_VRAM_CHECK=1; shift ;;
    --force-traffic)    FORCE_TRAFFIC=1; shift ;;
    --extra)            EXTRA_ARGS="$2"; shift 2 ;;
    --py)               PY="$2"; shift 2 ;;
    --)                 shift; EXTRA_ARGS="$EXTRA_ARGS $*"; break ;;
    *)                  die "Unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "$DATASET"   ]] || die "--dataset 未传。支持: ETTh1 ETTh2 ETTm1 ETTm2 weather electricity exchange_rate traffic"
[[ -n "$GPU_TIER"  ]] || die "--gpu-tier 未传。仅支持: 24G, 8G"
[[ "$GPU_TIER" == "24G" || "$GPU_TIER" == "8G" ]] || die "--gpu-tier 当前只支持 24G 或 8G (去掉中间档，d_model/layers 锁死只改 grad_accum)"

# -------------------------------------------------------------- traffic gate
if [[ "$DATASET" == "traffic" && "$FORCE_TRAFFIC" -eq 0 ]]; then
  die "traffic (D=862) 数据集太大，early iteration 暂不建议跑。配置文件保留: src/configs/scales/traffic_{24g,8g}.yaml。
      如果确需在 traffic 上试验，请传 --force-traffic 绕过此检查。"
fi

# ------------------------------------------------------------  超参数映射
# 注意：同一数据集 24G 和 8G 的 d_model / n_layers 完全相同（用户要求）。
#       两档的唯一区别就是 --batch-size (physical) 和 --grad-accum。
#       BS_eff = BS_phys × grad_accum 在同一数据集内保持相同。

# ---------- stage 长度 / 学习率调度公共块（按数据集调，不按 tier） ----------
ETT_STAGE_COMMON="--d-model 512 --agg-d-model 256 --n-layers 4 \
 --max-epochs-stage1 40 --max-epochs-stage2 15 --max-epochs-stage3 15 --max-epochs-stage4 30 \
 --early-stop 10 --stage4-lr-mult 0.03 --stage4-early-stop 5 --amp --lr 1e-3 \
 --seq-len 336 --pred-len 168 --label-len 168"

WEATHER_STAGE_COMMON="--d-model 512 --agg-d-model 256 --n-layers 4 \
 --max-epochs-stage1 40 --max-epochs-stage2 15 --max-epochs-stage3 15 --max-epochs-stage4 35 \
 --early-stop 10 --stage4-lr-mult 0.03 --stage4-early-stop 5 --amp --lr 8e-4 \
 --seq-len 336 --pred-len 168 --label-len 168"

ELEC_STAGE_COMMON="--d-model 384 --agg-d-model 192 --n-layers 3 \
 --max-epochs-stage1 40 --max-epochs-stage2 15 --max-epochs-stage3 15 --max-epochs-stage4 30 \
 --early-stop 10 --stage4-lr-mult 0.03 --stage4-early-stop 5 --amp --lr 6e-4 \
 --seq-len 336 --pred-len 168 --label-len 168"

EXCHANGE_STAGE_COMMON="--d-model 512 --agg-d-model 256 --n-layers 4 \
 --max-epochs-stage1 40 --max-epochs-stage2 15 --max-epochs-stage3 15 --max-epochs-stage4 30 \
 --early-stop 10 --stage4-lr-mult 0.03 --stage4-early-stop 5 --amp --lr 1e-3 \
 --seq-len 336 --pred-len 168 --label-len 168"

TRAFFIC_STAGE_COMMON="--d-model 384 --agg-d-model 192 --n-layers 3 \
 --max-epochs-stage1 60 --max-epochs-stage2 20 --max-epochs-stage3 20 --max-epochs-stage4 30 \
 --early-stop 15 --stage4-lr-mult 0.03 --stage4-early-stop 5 --amp --lr 6e-4 \
 --seq-len 336 --pred-len 168 --label-len 168"

case "${DATASET}:${GPU_TIER}" in
  # ===== ETT family (ETTh1/ETTh2/ETTm1/ETTm2) —— 24G: BS=16×accum4=64; 8G: BS=4×accum16=64 =====
  ETTh1:24G|ETTh2:24G|ETTm1:24G|ETTm2:24G)
    MAIN_ARGS="${ETT_STAGE_COMMON} --batch-size 16 --grad-accum 4"
    EFF_BS=$((16 * 4))
    ;;
  ETTh1:8G|ETTh2:8G|ETTm1:8G|ETTm2:8G)
    MAIN_ARGS="${ETT_STAGE_COMMON} --batch-size 4 --grad-accum 16"
    EFF_BS=$((4 * 16))
    ;;

  # ===== Weather D=21 —— 24G: BS=12×4=48; 8G: BS=3×16=48 =====
  weather:24G)
    MAIN_ARGS="${WEATHER_STAGE_COMMON} --batch-size 12 --grad-accum 4"
    EFF_BS=$((12 * 4))
    ;;
  weather:8G)
    MAIN_ARGS="${WEATHER_STAGE_COMMON} --batch-size 3 --grad-accum 16"
    EFF_BS=$((3 * 16))
    ;;

  # ===== Electricity D=321 —— 24G: BS=10×4=40; 8G: BS=2×20=40 =====
  electricity:24G)
    MAIN_ARGS="${ELEC_STAGE_COMMON} --batch-size 10 --grad-accum 4"
    EFF_BS=$((10 * 4))
    ;;
  electricity:8G)
    MAIN_ARGS="${ELEC_STAGE_COMMON} --batch-size 2 --grad-accum 20"
    EFF_BS=$((2 * 20))
    ;;

  # ===== Exchange_rate D=8 —— 24G: BS=32×2=64; 8G: BS=8×8=64 =====
  exchange_rate:24G)
    MAIN_ARGS="${EXCHANGE_STAGE_COMMON} --batch-size 32 --grad-accum 2"
    EFF_BS=$((32 * 2))
    ;;
  exchange_rate:8G)
    MAIN_ARGS="${EXCHANGE_STAGE_COMMON} --batch-size 8 --grad-accum 8"
    EFF_BS=$((8 * 8))
    ;;

  # ===== Traffic D=862 —— ONLY runs if --force-traffic was passed =====
  # 24G: BS=2 × 32 = 64 (VRAM ≈ 11GB)
  # 8G : BS=1 × 64 = 64 (borderline, may OOM)
  traffic:24G)
    MAIN_ARGS="${TRAFFIC_STAGE_COMMON} --batch-size 2 --grad-accum 32"
    EFF_BS=$((2 * 32))
    ;;
  traffic:8G)
    MAIN_ARGS="${TRAFFIC_STAGE_COMMON} --batch-size 1 --grad-accum 64"
    EFF_BS=$((1 * 64))
    echo "[!] traffic × 8G 非常激进，大概率 OOM。如果 OOM 请降 seq-len 或等后续蒸馏版本。" >&2
    ;;

  *)
    die "未知组合: dataset=${DATASET}, tier=${GPU_TIER}  （只支持 24G/8G 两档；traffic 需要传 --force-traffic）"
    ;;
esac

# 从 MAIN_ARGS 自动抽几个关键值给 VRAM pre-check 打印
EXTRACT() {
  local key="--$1"
  local nxt=0
  for tok in $MAIN_ARGS; do
    [[ $nxt -eq 1 ]] && { echo "$tok"; return; }
    [[ "$tok" == "$key" ]] && nxt=1
  done
}
SEQ_LEN=$(EXTRACT seq-len); SEQ_LEN=${SEQ_LEN:-336}
PRED_LEN=$(EXTRACT pred-len); PRED_LEN=${PRED_LEN:-168}
D_MODEL=$(EXTRACT d-model); D_MODEL=${D_MODEL:-512}
N_LAYERS=$(EXTRACT n-layers); N_LAYERS=${N_LAYERS:-4}
PHYS_BS=$(EXTRACT batch-size); PHYS_BS=${PHYS_BS:-32}
GRAD_ACC=$(EXTRACT grad-accum); GRAD_ACC=${GRAD_ACC:-1}

# ---------------------------------------------------------------- preamble
echo "========================================================"
echo " TACF scaled run: dataset=${DATASET}  tier=${GPU_TIER}  (MODEL LOCKED, ONLY grad-accum adaptive)"
echo "========================================================"
echo "  d_model / n_layers / agg_d_model     :  锁定（24G 和 8G 完全相同）"
echo "  物理 batch size (→ 决定 VRAM)        :  $PHYS_BS"
echo "  梯度累积 accum_steps                 :  $GRAD_ACC"
echo "  等效 batch size (→ 决定梯度噪声/BN)  :  $EFF_BS"
echo "  main.py args (不含 --dataset & --extra):"
echo "    ${MAIN_ARGS}"
[[ -n "${EXTRA_ARGS}" ]] && echo "  extra_args:  ${EXTRA_ARGS}"
echo

# ---------------------------------------------------------  VRAM sanity check
if [[ "${NO_VRAM_CHECK}" -eq 0 && -f estimate_vram_fast.py ]]; then
  echo "--- estimate_vram_fast.py VRAM 预检 (按 **physical batch size=${PHYS_BS}** 估算) ---"
  ${PY} estimate_vram_fast.py --datasets "${DATASET}" \
    --seq-len "${SEQ_LEN}" \
    --pred-len "${PRED_LEN}" \
    --d-model "${D_MODEL}" \
    --n-layers "${N_LAYERS}" \
    --batch-size "${PHYS_BS}" \
    --amp \
    || echo "[!] VRAM 预估失败 (非致命，继续训练)"
  case "${GPU_TIER}" in
    24G) CAP="21.6" ;;
    8G)  CAP="7.2"  ;;
    *)   CAP="?" ;;
  esac
  echo "— GPU tier ${GPU_TIER} 安全显存上限 ≈ 0.9 × tier = ${CAP}GB。若上表数值超过此上限，Ctrl+C 中断调小 --batch-size / 调大 --grad-accum 重来。 —"
  echo
fi

# ------------------------------------------------------------------- GO!
echo "🚀 启动训练:  $PY -m src.experiments.main --dataset ${DATASET} ${MAIN_ARGS} ${EXTRA_ARGS}"
echo
exec ${PY} -u -m src.experiments.main --dataset "${DATASET}" ${MAIN_ARGS} ${EXTRA_ARGS}
