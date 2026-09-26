#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scaled_runs.sh  --- 一键调用 TACF 生产级（SOTA-对齐）训练的 bash 包装器。
#
# 2026-09-26 v4  SOTA 全量多数据集一键排程：
#   - 新增 --all：一键按 D 小到大顺序跑 7 个标准数据集
#       [ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather, electricity]
#       Traffic (D=862) 仍需 --force-traffic，且不包含在 --all 中。
#   - 新增 --datasets "DS1 DS2 ..."：自定义要跑的数据集列表。
#   - 新增 --continue-on-fail：某个 dataset 失败后继续跑下一个，不中断 --all。
#   - 新增 --tag-suffix / --run-tag-suffix：透传到 main.py --tag-suffix，
#       区分 ablation/seeds。
#   - 所有 run 结束后自动写出 <logs>/all_runs_<ts>/summary.json，
#       汇总 7 个 dataset 的 { seq, pred, label, log_dir, exit_code, dataset }。
#   - 日志目录命名规则升级（main.py PROJECT_NAME）：
#       <ds>_s<seq>_p<pred>_l<label>[_suffix]_<YYYYMMDD_HHMMSS>
#       原 tacf_YYYYMMDD_HHMMSS 已废弃，不再出现。
#
# 使用例子：
#   bash scaled_runs.sh --help
#   bash scaled_runs.sh --dataset ETTh1         --gpu-tier 24G           # SOTA: seq=336 pred=96
#   bash scaled_runs.sh --dataset weather       --gpu-tier 8G  --pred-len 336
#   bash scaled_runs.sh --dataset electricity   --gpu-tier 24G           # Electricity 顶会最常用 336→96
#   bash scaled_runs.sh --dataset exchange_rate --gpu-tier 8G  --seq-len 512 --pred-len 720
#   bash scaled_runs.sh --dataset traffic       --gpu-tier 24G --force-traffic
#   # ===== 一键跑 7 个标准数据集（顶会最常用 336→96 默认组合，D 升序排）=====
#   bash scaled_runs.sh --all --gpu-tier 24G
#   # 自定义数据集子集：
#   bash scaled_runs.sh --datasets "ETTh1 electricity exchange_rate" --gpu-tier 8G
#   # 打印将执行的命令（不实际训练）：
#   bash scaled_runs.sh --dataset ETTh1 --gpu-tier 8G --dry-run
#   bash scaled_runs.sh --all --gpu-tier 24G --dry-run
#   # 关闭 SOTA 开关，回退 baseline 跑法：
#   bash scaled_runs.sh --dataset ETTh1 --gpu-tier 8G --no-sota-flags
#   # 额外透传 main.py 参数：
#   bash scaled_runs.sh --dataset ETTh1 --gpu-tier 24G --extra "--seed 123 --tag my_run"
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

die() { echo "ERROR: $*" >&2 ; exit 2 ; }

# ---------------------------------------------------------------- defaults
DATASET=""
MULTI_MODE=""            # "all" | "list" | "" (single)
DATASETS_LIST=""
CONTINUE_ON_FAIL=0
TAG_SUFFIX=""
NO_VRAM_CHECK=0
FORCE_TRAFFIC=0
ALLOW_NONSTANDARD=0
NO_SOTA_FLAGS=0
DRY_RUN=0
EXTRA_ARGS=""
PY="${PYTHON:-python}"
# SOTA 默认 setting: seq_len=336 × pred_len=96（所有 DLinear/iTransformer 论文
# 在 ETT / Weather / Electricity / Exchange 四大家族的最常用长 horizon 组合）
SEQ_LEN_DEF=336
PRED_LEN_DEF=96
SEQ_LEN=""
PRED_LEN=""
LABEL_LEN=""
# 7 个标准数据集：按 D（feature dim）升序排，跑 --all 时顺序执行：
#   ETTh1/2=7, ETTm1/2=7, exchange=8, weather=21, electricity=321, traffic=862(excl)
ALL_7_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather electricity"
LOG_ROOT="${SCRIPT_DIR}/logs"

# ---------------------------------------------------------------- parser
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo
  # print header from file docstring, skip shebang
  awk 'NR>=3 && NR<=60 && /^#/ { sub("^# ?",""); print }' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Script options (SINGLE-dataset mode OR MULTI-dataset mode, pick one):
  Single-dataset (classic):
    --dataset <DS>             ETTh1|ETTh2|ETTm1|ETTm2|weather|electricity|exchange_rate|traffic

  Multi-dataset (either, not both):
    --all                      Run all 7 standard datasets in D-asc order:
                                 ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather electricity
                               (traffic excluded; use --force-traffic --dataset traffic to run it)
    --datasets "DS1 DS2 ..."   Run a user-defined list of datasets (same ordering as given).

  Common:
    --gpu-tier <24G|8G>        (必填) GPU 显存等级（两档锁模型，唯 BS×accum 调）
    --seq-len  <96|336|512>    标准 lookback，默认 336（SOTA 最常用）
    --pred-len <96|192|336|720> 标准 horizon，默认 96（SOTA 最常用基准档）
    --label-len <N>            decoder start token，默认 pred_len//2（TS-Lib 惯例）

    --no-vram-check            跳过 estimate_vram_fast.py VRAM 预检
    --force-traffic            traffic 默认不推荐，传此 flag 才放行
    --allow-nonstandard-setting  允许 seq/pred 非顶会标准 12 对（会触发 main.py stderr WARNING）
    --no-sota-flags            关闭 RevIN / DLinear-trend-head / σ_init=1.0 等 SOTA 升级
    --dry-run                  只打印 main.py 命令，不执行（用于参数审查）；多数据集模式下打印所有 run。
    --continue-on-fail         多数据集模式：某个 dataset 失败，继续跑下一个。默认: 第一个失败立即停。
    --extra '...'              额外透传给 src.experiments.main 的参数（所有子 run 共用）
    --tag-suffix '...'         透传给 main.py --tag-suffix，所有子 run 共用（区分 ablation/seeds）
    --py <bin>                 Python 解释器，默认 $PYTHON 或 python
    -h / --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)                   usage; exit 0 ;;
    --dataset)                   DATASET="$2"; shift 2 ;;
    --all)                       MULTI_MODE="all"; shift ;;
    --datasets)                  MULTI_MODE="list"; DATASETS_LIST="$2"; shift 2 ;;
    --gpu-tier)                  GPU_TIER="$2"; shift 2 ;;
    --seq-len)                   SEQ_LEN="$2"; shift 2 ;;
    --pred-len)                  PRED_LEN="$2"; shift 2 ;;
    --label-len)                 LABEL_LEN="$2"; shift 2 ;;
    --no-vram-check)             NO_VRAM_CHECK=1; shift ;;
    --force-traffic)             FORCE_TRAFFIC=1; shift ;;
    --allow-nonstandard-setting) ALLOW_NONSTANDARD=1; shift ;;
    --no-sota-flags)             NO_SOTA_FLAGS=1; shift ;;
    --dry-run)                   DRY_RUN=1; shift ;;
    --continue-on-fail)          CONTINUE_ON_FAIL=1; shift ;;
    --extra)                     EXTRA_ARGS="$2"; shift 2 ;;
    --tag-suffix)                TAG_SUFFIX="$2"; shift 2 ;;
    --run-tag-suffix)            TAG_SUFFIX="$2"; shift 2 ;;   # alias
    --py)                        PY="$2"; shift 2 ;;
    --)                          shift; EXTRA_ARGS="$EXTRA_ARGS $*"; break ;;
    *)                           die "Unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "$GPU_TIER"  ]] || die "--gpu-tier 未传。仅支持: 24G, 8G"
[[ "$GPU_TIER" == "24G" || "$GPU_TIER" == "8G" ]] || die "--gpu-tier 当前只支持 24G 或 8G (去掉中间档，d_model/layers 锁死只改 grad_accum)"

# --- MULTI mode entrypoint ------------------------------------------------
if [[ "$MULTI_MODE" == "all" || "$MULTI_MODE" == "list" ]]; then
  if [[ -n "$DATASET" ]]; then
    die "multi dataset mode (--all / --datasets) 与 --dataset 互斥，只能二选一。"
  fi
  if [[ "$MULTI_MODE" == "all" ]]; then
    DATASETS_LIST="$ALL_7_DATASETS"
  fi
  # split DATASETS_LIST into array
  read -r -a DS_ARRAY <<<"$DATASETS_LIST"
  if [[ "${#DS_ARRAY[@]}" -eq 0 ]]; then
    die "--datasets 列表为空。"
  fi

  # 批 run summary 目录
  BATCH_TS=$(date +%Y%m%d_%H%M%S)
  BATCH_DIR="${LOG_ROOT}/all_runs_${BATCH_TS}"
  mkdir -p "${BATCH_DIR}"
  SUMMARY_JSON="${BATCH_DIR}/summary.json"

  echo "================================================================="
  echo " TACF multi-dataset BATCH RUN  (${#DS_ARRAY[@]} datasets)"
  echo "   mode     : ${MULTI_MODE}"
  echo "   datasets : ${DS_ARRAY[*]}"
  echo "   tier     : ${GPU_TIER}"
  echo "   setting  : seq=${SEQ_LEN_DEF} pred=${PRED_LEN_DEF} (per-run CLI 覆盖生效)"
  echo "   tag-suf  : ${TAG_SUFFIX:-<none>}"
  echo "   extra    : ${EXTRA_ARGS:-<none>}"
  echo "   continue : $([[ $CONTINUE_ON_FAIL -eq 1 ]] && echo 'ON (失败继续)' || echo 'OFF (首个失败即停)')"
  echo "   batch dir: ${BATCH_DIR}"
  echo "================================================================="

  # 先把 summary header 写好
  cat >"$SUMMARY_JSON" <<EOF
{
  "batch_ts": "${BATCH_TS}",
  "gpu_tier": "${GPU_TIER}",
  "multi_mode": "${MULTI_MODE}",
  "datasets": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${DS_ARRAY[@]}"),
  "seq_len_default": ${SEQ_LEN_DEF},
  "pred_len_default": ${PRED_LEN_DEF},
  "tag_suffix": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1] if len(sys.argv)>1 else None))' "$TAG_SUFFIX"),
  "run_order": [],
  "runs": {}
}
EOF

  EXIT_TOTAL=0
  RUN_IDX=0
  for DS in "${DS_ARRAY[@]}"; do
    RUN_IDX=$((RUN_IDX+1))
    RUN_ID=$(printf "%02d_%s" "$RUN_IDX" "$DS")
    echo
    echo "#################################################################"
    echo " #${RUN_IDX}/${#DS_ARRAY[@]}  START  dataset=${DS}"
    echo "#################################################################"
    # 启动脚本本身（递归），把除 --all/--datasets 以外的参数原样透传；
    # 子进程通过 --dataset "$DS" 指定单 dataset mode。
    # 子进程的 stdout/stderr 继承当前进程（直接看）。
    SUB_ARGS=( --dataset "$DS" --gpu-tier "$GPU_TIER" )
    [[ -n "$SEQ_LEN"       ]] && SUB_ARGS+=( --seq-len       "$SEQ_LEN" )
    [[ -n "$PRED_LEN"      ]] && SUB_ARGS+=( --pred-len      "$PRED_LEN" )
    [[ -n "$LABEL_LEN"     ]] && SUB_ARGS+=( --label-len     "$LABEL_LEN" )
    [[ $NO_VRAM_CHECK      -eq 1 ]] && SUB_ARGS+=( --no-vram-check )
    [[ $FORCE_TRAFFIC      -eq 1 ]] && SUB_ARGS+=( --force-traffic )
    [[ $ALLOW_NONSTANDARD  -eq 1 ]] && SUB_ARGS+=( --allow-nonstandard-setting )
    [[ $NO_SOTA_FLAGS      -eq 1 ]] && SUB_ARGS+=( --no-sota-flags )
    [[ $DRY_RUN            -eq 1 ]] && SUB_ARGS+=( --dry-run )
    [[ -n "$EXTRA_ARGS"    ]] && SUB_ARGS+=( --extra "$EXTRA_ARGS" )
    [[ -n "$TAG_SUFFIX"    ]] && SUB_ARGS+=( --tag-suffix "$TAG_SUFFIX" )
    [[ -n "${PY:-}"        ]] && SUB_ARGS+=( --py "$PY" )
    set +e
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run][$DS] 子命令: bash $0 ${SUB_ARGS[*]}"
      SUB_EXIT=0
    else
      bash "$0" "${SUB_ARGS[@]}"
      SUB_EXIT=$?
    fi
    set -e
    # 记录 exit code
    if [[ $SUB_EXIT -ne 0 ]]; then
      EXIT_TOTAL=$(( EXIT_TOTAL + 1 ))
      echo "⚠️   dataset=${DS}  exit=${SUB_EXIT}" >&2
      if [[ $CONTINUE_ON_FAIL -eq 0 ]]; then
        echo "已停止后续数据集（传 --continue-on-fail 可忽略错误继续跑）。" >&2
        # 补完 summary
        "${PY:-python3}" - "$SUMMARY_JSON" "$DS" "$SUB_EXIT" "ABORTED" <<'PY'
import json, sys, pathlib
p, ds, rc, status = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
obj = json.loads(p.read_text())
obj["runs"][ds] = {"dataset": ds, "exit_code": rc, "status": status}
obj.setdefault("run_order", []).append(ds)
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
PY
        exit $EXIT_TOTAL
      fi
    else
      echo "✅   dataset=${DS}  exit=0"
    fi
    # 更新 summary.json：run_order + runs[ds] entry（哪怕 exit !=0 也要记）
    "${PY:-python3}" - "$SUMMARY_JSON" "$DS" "$SUB_EXIT" <<'PY'
import json, sys, pathlib
p, ds, rc = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
obj = json.loads(p.read_text())
obj.setdefault("run_order", []).append(ds)
obj["runs"][ds] = {
  "dataset": ds,
  "exit_code": rc,
  "status": "OK" if rc == 0 else "FAILED",
}
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
PY
  done

  echo
  echo "================================================================="
  echo " 多数据集批跑完成：成功=$(( ${#DS_ARRAY[@]} - EXIT_TOTAL ))/ 失败=${EXIT_TOTAL} 共 ${#DS_ARRAY[@]}"
  echo " summary: ${SUMMARY_JSON}"
  echo "================================================================="
  exit $EXIT_TOTAL
fi
# --- END MULTI mode entrypoint; continue to SINGLE-dataset mode below -----

[[ -n "$DATASET"   ]] || die "请传 --dataset <DS> 或 --all / --datasets \"DS1 DS2 ...\"。支持: $ALL_7_DATASETS traffic"

SEQ_LEN="${SEQ_LEN:-$SEQ_LEN_DEF}"
PRED_LEN="${PRED_LEN:-$PRED_LEN_DEF}"
# label_len 惯例：TS-Lib Autoformer/PatchTST 都用 label_len = pred_len//2；
# DLinear/iTransformer 等纯 encoder 模型不使用此参数，但保持此惯例兼容性更好。
LABEL_LEN="${LABEL_LEN:-$(( PRED_LEN / 2 ))}"

# ------------------------------------------------------------ setting 校验
STANDARD_SEQS=" 96 336 512 "
STANDARD_PREDS=" 96 192 336 720 "
is_in() { [[ " $1 " == *" $2 "* ]] ; }
SEQ_OK=0; PRED_OK=0
is_in "$STANDARD_SEQS"  "$SEQ_LEN"  && SEQ_OK=1
is_in "$STANDARD_PREDS" "$PRED_LEN" && PRED_OK=1
if [[ $SEQ_OK -eq 0 || $PRED_OK -eq 0 ]]; then
  if [[ $ALLOW_NONSTANDARD -eq 1 ]]; then
    echo "[!] allow-nonstandard-setting: seq_len=${SEQ_LEN} pred_len=${PRED_LEN} 非顶会标准 12 对。已放行，但与 DLinear/TEFN/iTransformer 对比不公平。" >&2
  else
    die "seq_len=${SEQ_LEN} pred_len=${PRED_LEN} 不属于顶会标准组合（3×4=12对）。
     标准 lookback ∈ {96, 336, 512}, horizon ∈ {96, 192, 336, 720}.
     （DLinear AAAI23 / iTransformer ICLR24 / PatchTST ICLR23 / Bi-Mamba4TS / Mamba 所有 TS-Lib 脚本通用。）
     常见公平对比 setting:
       seq=336 pred=96       [顶会最常用 baseline]
       seq=336 pred=192/336/720   [四档 horizon sweep]
       seq=512 pred=96/192/336/720 [长 lookback sweep]
       seq=96  pred=96            [短 lookback baseline]
     坚持跑自定义 setting 请传 --allow-nonstandard-setting。"
  fi
fi

# -------------------------------------------------------------- traffic gate
if [[ "$DATASET" == "traffic" && "$FORCE_TRAFFIC" -eq 0 ]]; then
  die "traffic (D=862) 数据集太大，early iteration 暂不建议跑。配置文件保留: src/configs/scales/traffic_{24g,8g}.yaml。
      如果确需在 traffic 上试验，请传 --force-traffic 绕过此检查。"
fi

# ------------------------------------------------------------  超参数映射
# 说明：同一数据集 24G 和 8G 的 d_model / n_layers 完全相同（锁模型）。
#       两档唯一区别是 --batch-size (physical) 和 --grad-accum。
#       BS_eff = BS_phys × grad_accum 在同一数据集内保持相同。
#
# SOTA_FLAGS 公共块：RevIN + DLinear-trend-head + σ_mult_init=1.0（除非 --no-sota-flags）
if [[ "$NO_SOTA_FLAGS" -eq 1 ]]; then
  SOTA_FLAGS=""
else
  SOTA_FLAGS="--use-revin --use-dlinear-trend-head --sigma-global-multiplier-init 1.0"
fi

# ---------- stage 长度 / 容量 / 学习率调度公共块（按数据集单独适配，跨 tier 锁）
# 参考：src/configs/scales/{ett,weather,electricity,exchange}_{24g,8g}.yaml 基线。
# 统一：RevIN + DLinear trend head + σ_mult_init=1.0（除非 --no-sota-flags）
#       S1/S2/S3/S4 epochs 拉到 SOTA 级；S4 stage4_lr_mult/early_stop 统一升级；
#       seq_len / pred_len / label_len 已由 CLI 处理。

# ===== ETT family ×4 (ETTh1/ETTh2/ETTm1/ETTm2) — D=7 小维度 —— SOTA: 768/6/384 =====
#   scales YAML baseline: d=512, L=4, agg=256, lr=1e-3, BS_eff=64
#   升级：d×1.5, L×1.5, agg×1.5 → 容量 ×2.3 匹配 Bi-Mamba4TS / iTransformer-6 水平
ETT_STAGE_COMMON="--d-model 768 --agg-d-model 384 --n-layers 6 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 1e-3 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"

# ===== Weather D=21 —— 中等维度 —— SOTA: 768/6/384 =====
#   scales YAML baseline: d=512, L=4, agg=256, lr=8e-4, BS_eff=48
#   升级：容量 ×2.3 同上；Weather 信号略不稳，lr 保持 8e-4 基线不升
WEATHER_STAGE_COMMON="--d-model 768 --agg-d-model 384 --n-layers 6 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 8e-4 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"

# ===== Electricity D=321 —— 高维 —— SOTA: 576/4/288 =====
#   scales YAML baseline: d=384, L=3, agg=192, lr=6e-4, BS_eff=40
#   升级：容量 ×1.5 (384→576, 3→4, 192→288)。注意：D=321 × d=576 输入投影参数量 ≈ 321*576*4≈2.3M，
#   远大于 ETT 的 7*768*4≈0.24M；强行到 768/6 会 OOM 24G@phys BS>4 还慢 2×。
#   学习率保持基线 6e-4（高维稳定性优先）。BS_eff 提升到 50 梯度质量更稳。
ELEC_STAGE_COMMON="--d-model 576 --agg-d-model 288 --n-layers 4 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 6e-4 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"

# ===== Exchange_rate D=8 —— 小维度 + daily 噪声大 —— SOTA: 768/6/384 =====
#   scales YAML baseline: d=512, L=4, agg=256, lr=1e-3, BS_eff=64
#   升级：容量 ×2.3，BS_eff 升到 96（D 特小，phys BS 非常大不 OOM）
EXCHANGE_STAGE_COMMON="--d-model 768 --agg-d-model 384 --n-layers 6 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 1e-3 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"

# ===== Traffic D=862 —— 极大维，用户说"可以不改"；保留 scales YAML baseline 但 SOTA flags 仍然打开 =====
TRAFFIC_STAGE_COMMON="--d-model 384 --agg-d-model 192 --n-layers 3 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 15 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 6e-4 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"

case "${DATASET}:${GPU_TIER}" in
  # ===== ETT family ×4 —— 24G: BS=16×accum4=64 (原基线 eff=64, 保持一致); 8G: BS=4×accum16=64 =====
  #   基线 scales/ett_24g.yaml BS_eff=64，phys BS=16×accum4 完全一致，只是模型从 512/4→768/6
  ETTh1:24G|ETTh2:24G|ETTm1:24G|ETTm2:24G)
    MAIN_ARGS="${ETT_STAGE_COMMON} --batch-size 16 --grad-accum 4"
    EFF_BS=$((16 * 4))
    ;;
  ETTh1:8G|ETTh2:8G|ETTm1:8G|ETTm2:8G)
    MAIN_ARGS="${ETT_STAGE_COMMON} --batch-size 4 --grad-accum 16"
    EFF_BS=$((4 * 16))
    ;;

  # ===== Weather D=21 —— 24G: BS=12×4=48 (scales YAML 基线值); 8G: BS=3×16=48 =====
  #   基线 scales/weather_24g.yaml BS_eff=48，完全保持一致，容量 512/4→768/6
  weather:24G)
    MAIN_ARGS="${WEATHER_STAGE_COMMON} --batch-size 12 --grad-accum 4"
    EFF_BS=$((12 * 4))
    ;;
  weather:8G)
    MAIN_ARGS="${WEATHER_STAGE_COMMON} --batch-size 3 --grad-accum 16"
    EFF_BS=$((3 * 16))
    ;;

  # ===== Electricity D=321 —— 24G: BS=5×accum10=50; 8G: BS=2×accum25=50 =====
  #   基线 scales/electricity_24g.yaml BS_eff=40 → 升级到 50 (+25%)
  #   容量 384/3/192→576/4/288（×1.5，D=321 不升到 768 防 OOM，速度还能接受）
  #   24G BS=5 估算 VRAM ≈11G AMP（d=576, 321×seq=336），安全；8G BS=2 估算≈5.3G
  electricity:24G)
    MAIN_ARGS="${ELEC_STAGE_COMMON} --batch-size 5 --grad-accum 10"
    EFF_BS=$((5 * 10))
    ;;
  electricity:8G)
    MAIN_ARGS="${ELEC_STAGE_COMMON} --batch-size 2 --grad-accum 25"
    EFF_BS=$((2 * 25))
    ;;

  # ===== Exchange_rate D=8 —— 24G: BS=32×accum2=64; 8G: BS=8×accum8=64 =====
  #   基线 BS_eff=64（scales/exchange_24g.yaml），保持一致（D 小 phys BS 想拉很大都行，
  #   但 eff 太大 BN 不稳，保持基线 64 更安全）
  exchange_rate:24G)
    MAIN_ARGS="${EXCHANGE_STAGE_COMMON} --batch-size 32 --grad-accum 2"
    EFF_BS=$((32 * 2))
    ;;
  exchange_rate:8G)
    MAIN_ARGS="${EXCHANGE_STAGE_COMMON} --batch-size 8 --grad-accum 8"
    EFF_BS=$((8 * 8))
    ;;

  # ===== Traffic D=862 —— 按用户指示"保留不改"。容量 baseline d384/3/agg192。
  # 24G: BS=2×32=64; 8G: BS=1×64=64 (borderline OOM) =====
  traffic:24G)
    MAIN_ARGS="${TRAFFIC_STAGE_COMMON} --batch-size 2 --grad-accum 32"
    EFF_BS=$((2 * 32))
    ;;
  traffic:8G)
    MAIN_ARGS="${TRAFFIC_STAGE_COMMON} --batch-size 1 --grad-accum 64"
    EFF_BS=$((1 * 64))
    echo "[!] traffic × 8G 非常激进，大概率 OOM。如果 OOM 请调小 seq-len 或等后续蒸馏版本。" >&2
    ;;

  *)
    die "未知组合: dataset=${DATASET}, tier=${GPU_TIER}  （只支持 24G/8G 两档；traffic 需要传 --force-traffic）"
    ;;
esac

# 从 MAIN_ARGS 自动抽关键值，用于 VRAM 预检/打印
EXTRACT() {
  local key="--$1"
  local nxt=0
  for tok in $MAIN_ARGS; do
    [[ $nxt -eq 1 ]] && { echo "$tok"; return; }
    [[ "$tok" == "$key" ]] && nxt=1
  done
}
D_MODEL=$(EXTRACT d-model); D_MODEL=${D_MODEL:-768}
N_LAYERS=$(EXTRACT n-layers); N_LAYERS=${N_LAYERS:-6}
PHYS_BS=$(EXTRACT batch-size); PHYS_BS=${PHYS_BS:-16}
GRAD_ACC=$(EXTRACT grad-accum); GRAD_ACC=${GRAD_ACC:-1}

# ---------------------------------------------------------------- preamble
STANDARDITY_MSG="STANDARD (SOTA-公平对比可用)"
if [[ $SEQ_OK -eq 0 || $PRED_OK -eq 0 ]]; then
  STANDARDITY_MSG="NON-STANDARD（已通过 --allow-nonstandard-setting 放行，SOTA 对比不公平）"
fi
SOTA_STATE="ON (RevIN + DLinear trend + σ_init=1.0 + new-stage-hparams)"
[[ "$NO_SOTA_FLAGS" -eq 1 ]] && SOTA_STATE="OFF（baseline 回退，不含上述 SOTA 升级）"

echo "========================================================"
echo " TACF SOTA-对齐 scaled run: dataset=${DATASET}  tier=${GPU_TIER}"
echo "========================================================"
echo "  setting (seq → pred)                    :  ${SEQ_LEN} → ${PRED_LEN}   [${STANDARDITY_MSG}]"
echo "  label_len                               :  ${LABEL_LEN}  (惯例: pred_len/2)"
echo "  SOTA 升级开关                           :  ${SOTA_STATE}"
echo "  d_model / n_layers / agg_d_model        :  ${D_MODEL} / ${N_LAYERS} / $(EXTRACT agg-d-model)"
echo "  物理 batch size (→ VRAM)                :  ${PHYS_BS}"
echo "  梯度累积 accum_steps                    :  ${GRAD_ACC}"
echo "  等效 batch size (→ 梯度噪声/BN)         :  ${EFF_BS}"
echo "  main.py 超参数 (不含 --dataset/extra):"
echo "    ${MAIN_ARGS}"
[[ -n "${EXTRA_ARGS}" ]] && echo "  extra_args:  ${EXTRA_ARGS}"
echo

# ---------------------------------------------------------  VRAM sanity check
if [[ "${NO_VRAM_CHECK}" -eq 0 && -f estimate_vram_fast.py ]]; then
  echo "--- estimate_vram_fast.py VRAM 预检 (按 physical batch size=${PHYS_BS} 估算) ---"
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
  echo "— GPU tier ${GPU_TIER} 安全显存上限 ≈ 0.9 × tier = ${CAP}GB。若上表超过此上限，Ctrl+C 中断调小 --batch-size / 调大 --grad-accum 重来。 —"
  echo
fi

# ------------------------------------------------------------------- GO!
# 自动透传 TAG_SUFFIX 到 main.py --tag-suffix（命名规则：{ds}_s{seq}_p{pred}_l{label}_{suffix}_{ts}）
TAG_SUFFIX_MAIN_ARGS=""
if [[ -n "$TAG_SUFFIX" ]] && [[ "${TAG_SUFFIX}" != '""' ]]; then
  TAG_SUFFIX_MAIN_ARGS=" --tag-suffix ${TAG_SUFFIX}"
fi

CMD="${PY} -u -m src.experiments.main --dataset ${DATASET} ${MAIN_ARGS}${TAG_SUFFIX_MAIN_ARGS} ${EXTRA_ARGS}"
echo "🚀 启动训练: ${CMD}"
echo

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] 实际将执行（已跳过）:" >&2
  echo "  cd ${SCRIPT_DIR} && ${CMD}" >&2
  exit 0
fi

exec ${PY} -u -m src.experiments.main --dataset "${DATASET}" ${MAIN_ARGS}${TAG_SUFFIX_MAIN_ARGS} ${EXTRA_ARGS}
