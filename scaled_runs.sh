set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

die() { echo "ERROR: $*" >&2 ; exit 2 ; }

DATASET=""
MULTI_MODE=""            
DATASETS_LIST=""
RUN_MODE="full"           
                        
CONTINUE_ON_FAIL=1      
                      
RERUN=0                  
REBUILD_DONE=0           
DRY_RUN=0
NO_VRAM_CHECK=0
FORCE_TRAFFIC=0
ALLOW_NONSTANDARD=0
NO_SOTA_FLAGS=0
TAG_SUFFIX=""
EXTRA_ARGS=""
PY=""
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PY="${SCRIPT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
fi
if [[ -z "${PY}" ]] || ! command -v "${PY}" >/dev/null 2>&1; then
  cat >&2 <<'PYFATAL'
PYFATAL
  exit 127
fi
PY_CMD="${PY}"
export PY_CMD
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER=1
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
SEQ_LEN_DEF=336
PRED_LEN_DEF=96
SEQ_LEN=""
PRED_LEN=""
LABEL_LEN=""
ALL_DATASETS="ETTh1 ETTh2 weather exchange_rate"
LOG_ROOT="${SCRIPT_DIR}/logs"
FULL_SEQS="96 336 512"
FULL_PREDS="96 192 336 720"

_build_sota_flags() {
  local NOSF=$1
  if [[ "$NOSF" -eq 1 ]]; then echo "";
  else echo "--use-revin --use-dlinear-trend-head --sigma-global-multiplier-init 1.0"; fi
}

_extract_flag() {
  local tokens="$1"
  local key="--$2"
  local nxt=0
  local tok
  for tok in $tokens; do
    if [[ $nxt -eq 1 ]]; then echo "$tok"; return; fi
    if [[ "$tok" == "$key" ]]; then nxt=1; fi
  done
  echo "$3"
}

_build_hypers() {
  local DS=$1 TIER=$2 SEQ=$3 PRED=$4 LAB=$5 NOSF=$6
  local SOTA_F
  SOTA_F=$(_build_sota_flags "$NOSF")

  local EPOCH_COMMON="--max-epochs-stage1 60 --max-epochs-stage2 50 --max-epochs-stage3 30 --max-epochs-stage4 40 --early-stop 10 --stage2-lr-mult 2.0 --stage3-lr-mult 2.0 --stage2-early-stop 15 --stage4-lr-mult 0.10 --stage4-early-stop 12"
  local AMP_STR="--amp"
  local CAPACITY=""
  if [[ $PRED -ge 336 ]]; then
    AMP_STR=""    
    case "$DS" in
      ETTh1|ETTh2|ETTm1|ETTm2|weather|exchange_rate)
        CAPACITY="--d-model 576 --agg-d-model 288 --n-layers 4 --lr 1e-3"   
        ;;
      electricity)
        CAPACITY="--d-model 576 --agg-d-model 288 --n-layers 4 --lr 6e-4"  
        ;;
      traffic)
        CAPACITY="--d-model 384 --agg-d-model 192 --n-layers 3 --lr 6e-4"  
        ;;
      *)
        die "_build_hypers: 未知 dataset=$DS"
        ;;
    esac
  else
    case "$DS" in
      ETTh1|ETTh2|ETTm1|ETTm2)
        CAPACITY="--d-model 768 --agg-d-model 384 --n-layers 6 --lr 1e-3"
        ;;
      weather)
        CAPACITY="--d-model 768 --agg-d-model 384 --n-layers 6 --lr 8e-4"
        ;;
      electricity)
        CAPACITY="--d-model 576 --agg-d-model 288 --n-layers 4 --lr 6e-4"
        ;;
      exchange_rate)
        CAPACITY="--d-model 768 --agg-d-model 384 --n-layers 6 --lr 1e-3"
        ;;
      traffic)
        CAPACITY="--d-model 384 --agg-d-model 192 --n-layers 3 --lr 6e-4"
        ;;
      *)
        die "_build_hypers: 未知 dataset=$DS"
        ;;
    esac
  fi
  echo "${CAPACITY} ${EPOCH_COMMON} ${AMP_STR} --seq-len ${SEQ} --pred-len ${PRED} --label-len ${LAB} ${SOTA_F}"
}

_build_phys_bs() {
  local DS=$1 TIER=$2 PRED=$3 SEQ=$4
  if [[ -z "$DS" || -z "$TIER" ]]; then
    echo "--batch-size 16 --grad-accum 4"; return 0
  fi
  if [[ -z "${PRED:-}" ]] || ! [[ "$PRED" =~ ^[0-9]+$ ]]; then
    echo "[warn] _build_phys_bs: PRED=$PRED 非数字，fallback 到 PRED=96（2^0 档）" >&2
    PRED=96
  fi
  if [[ -n "${SEQ:-}" ]] && ! [[ "$SEQ" =~ ^[0-9]+$ ]]; then
    SEQ=""
  fi
  local BS=1 ACC=1
  case "${DS}:${TIER}" in
    ETTh1:32G|ETTh2:32G|ETTm1:32G|ETTm2:32G) BS=32; ACC=2 ;;
    weather:32G)        BS=32; ACC=2 ;;
    electricity:32G)    BS=16; ACC=2 ;;
    exchange_rate:32G)  BS=32; ACC=1 ;;
    traffic:32G)        BS=8;  ACC=4 ;;
    ETTh1:24G|ETTh2:24G|ETTm1:24G|ETTm2:24G) BS=16; ACC=4 ;;
    weather:24G)        BS=8;  ACC=4 ;;
    electricity:24G)    BS=4;  ACC=8 ;;
    exchange_rate:24G)  BS=16; ACC=2 ;;
    traffic:24G)        BS=2;  ACC=16 ;;
    ETTh1:8G|ETTh2:8G|ETTm1:8G|ETTm2:8G) BS=4;  ACC=16 ;;
    weather:8G)         BS=2;  ACC=16 ;;
    electricity:8G)     BS=1;  ACC=32 ;;
    exchange_rate:8G)   BS=4;  ACC=8 ;;
    traffic:8G)         BS=1;  ACC=32 ;;
    *)
      echo "[warn] _build_phys_bs: 未知 (dataset=$DS, tier=$TIER)，fallback ETT 24G 保守档 BS=16 ACC=4" >&2
      BS=16; ACC=4 ;;
  esac
  local P720_NOSECOND_CUT=0
  case "${DS}:${TIER}" in
    exchange_rate:32G|weather:32G) P720_NOSECOND_CUT=1 ;;
  esac
  if [[ $PRED -ge 336 ]]; then BS=$(( BS / 2 )); fi
  if [[ $PRED -ge 720 && $P720_NOSECOND_CUT -ne 1 ]]; then BS=$(( BS / 2 )); fi
  if [[ "$TIER" == "32G" && "${SEQ:-0}" -ge 512 && $PRED -lt 336 ]]; then
    case "$DS" in
      ETTh1|ETTh2|ETTm1|ETTm2|weather|exchange_rate)
        BS=$(( BS / 2 ))
        ACC=$(( ACC * 2 ))
        ;;
    esac
  fi
  if [[ -n "${SEQ:-}" && $SEQ -ge 1024 ]]; then BS=$(( BS / 2 )); fi
  if [[ $BS -lt 1 ]]; then BS=1; fi
  echo "--batch-size $BS --grad-accum $ACC"
}

_vram_check_wrapper() {
  local DS=$1 TIER=$2 NO_VRAM=$3 FORCE_TR=$4 PHYS_BS_STR=$5
  local D_MODEL_ N_LAYERS_ PHYS_BS_ GRAD_ACC_ EFF_BS_
  if [[ "$NO_VRAM" -eq 1 ]]; then return 0; fi
  if [[ ! -f estimate_vram_fast.py ]]; then return 0; fi
  D_MODEL_=$(_extract_flag "$PHYS_BS_STR:dummy" d-model 768)  
  PHYS_BS_=$(_extract_flag "$PHYS_BS_STR" batch-size 16)
  N_LAYERS_="6"
  case "$DS" in
    ETTh1|ETTh2|ETTm1|ETTm2|weather|exchange_rate) D_MODEL_="768" ;;
    electricity) D_MODEL_="576" ;;
    traffic) D_MODEL_="384" ;;
  esac
  case "$DS" in
    ETTh1|ETTh2|ETTm1|ETTm2|weather|exchange_rate) N_LAYERS_="6" ;;
    electricity) N_LAYERS_="4" ;;
    traffic) N_LAYERS_="3" ;;
  esac
  echo "--- estimate_vram_fast.py VRAM 预检 (dataset=${DS}, batch-size=${PHYS_BS_}, d=${D_MODEL_}, L=${N_LAYERS_}) ---"
  set +e
  "${PY_CMD}" estimate_vram_fast.py \
    --datasets "$DS" \
    --seq-len "${SEQ_LEN:-336}" --pred-len "${PRED_LEN:-96}" \
    --d-model "$D_MODEL_" --n-layers "$N_LAYERS_" \
    --batch-size "$PHYS_BS_" --amp
  set -e
  local CAP=""
  case "$TIER" in
    32G) CAP="28.0" ;;   
    24G) CAP="21.6" ;;
    8G)  CAP="7.2"  ;;
    *)   CAP="?" ;;
  esac
  echo "— GPU tier ${TIER} 安全上限（日常使用）= ${CAP}GB。超了就 Ctrl+C 中断，调小 --batch-size/调大 --grad-accum。 —"
  echo
}

_print_trainset_slogan() {
  local DS=$1 TIER=$2 SEQ=$3 PRED=$4 LAB=$5 PHYS_BS_STR=$6 NOSF=$7 NO_VRAM=$8 FORCE_TR=$9
  local HYP
  HYP=$(_build_hypers "$DS" "$TIER" "$SEQ" "$PRED" "$LAB" "$NOSF")
  local D_MODEL_ AGG_D_ N_LAYERS_ PHYS_BS_ GRAD_ACC_ LR_
  D_MODEL_=$(_extract_flag  "$HYP" d-model       "768")
  AGG_D_=$(_extract_flag    "$HYP" agg-d-model   "384")
  N_LAYERS_=$(_extract_flag "$HYP" n-layers      "6")
  LR_=$(_extract_flag       "$HYP" lr            "1e-3")
  PHYS_BS_=$(_extract_flag  "$PHYS_BS_STR" batch-size "16")
  GRAD_ACC_=$(_extract_flag "$PHYS_BS_STR" grad-accum "1")
  local EFF_BS_=$(( PHYS_BS_ * GRAD_ACC_ ))

  local SEQ_OK=0 PRED_OK=0
  case " $FULL_SEQS "  in *" $SEQ "*)  SEQ_OK=1 ;; esac
  case " $FULL_PREDS " in *" $PRED "*) PRED_OK=1 ;; esac
  local STD_MSG="STANDARD (SOTA-公平对比可用)"
  if [[ $SEQ_OK -eq 0 || $PRED_OK -eq 0 ]]; then STD_MSG="NON-STANDARD（SOTA 对比不公平）"; fi
  local SOTA_STATE="ON (RevIN + DLinear trend + σ_init=1.0 + new-stage-hparams)"
  if [[ $NOSF -eq 1 ]]; then SOTA_STATE="OFF（baseline 回退，不含 SOTA 升级）"; fi

  cat <<EOF
========================================================
 TACF scaled run: dataset=${DS}  tier=${TIER}
========================================================
  setting (seq → pred)                    :  ${SEQ} → ${PRED}   [${STD_MSG}]
  label_len                               :  ${LAB}  (惯例: pred_len/2)
  SOTA 升级开关                           :  ${SOTA_STATE}
  d_model / n_layers / agg_d_model        :  ${D_MODEL_} / ${N_LAYERS_} / ${AGG_D_}
  物理 batch size (→ VRAM)                :  ${PHYS_BS_}
  梯度累积 accum_steps                    :  ${GRAD_ACC_}
  等效 batch size (→ 梯度噪声/BN)         :  ${EFF_BS_}
  learning rate                           :  ${LR_}
EOF
}

_cleanup_before_run() {
  local DRY="${1:-0}"
  local PY="${2:-python}"
  echo "  ---------------------------------------------------------------"
  echo "  [pre-run cleanup] 清理 GPU/HOST 残留（非致命，失败不中断训练）"
  if [[ "$DRY" -eq 1 ]]; then
    echo "    [dry-run] skip: gc.collect + torch.cuda.empty_cache ×3 + drop_caches try + orphan python kill"
    echo "  ---------------------------------------------------------------"
    return 0
  fi
  set +e
  if command -v "$PY" >/dev/null 2>&1; then
    "$PY" - <<'PYEOF'
import gc, sys
gc.collect()
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        print(f"    [cuda] empty_cache done. device={torch.cuda.get_device_name(None) if torch.cuda.device_count() else 'n/a'}")
    else:
        print("    [cuda] torch available but no CUDA device; skip.")
except Exception as exc:
    print(f"    [cuda] skipped ({exc}).", file=sys.stderr)
PYEOF
  else
    echo "    [gc/cuda] PY_CMD not available, skip."
  fi
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && echo "    [host] drop_caches (pagecache/dentries/inodes): OK"
  else
    echo "    [host] drop_caches skipped (no write perms to /proc/sys/vm/drop_caches)."
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local MY_UID
    MY_UID=$(id -u 2>/dev/null || echo "")
    if [[ -n "$MY_UID" ]]; then
      nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
      awk -F',' '{gsub(/ /,"",$1); if($1+0>0) print $1}' | \
      while read -r _PID; do
        [[ -z "$_PID" ]] && continue
        [[ -r "/proc/$_PID/status" ]] || continue
        local _PPID _EUID _CMDLINE
        _PPID=$(awk '/^PPid:/{print $2}' "/proc/$_PID/status" 2>/dev/null || echo 0)
        _EUID=$(awk '/^Uid:/{print $3}' "/proc/$_PID/status" 2>/dev/null || echo "")
        _CMDLINE=$(tr '\0' ' ' < "/proc/$_PID/cmdline" 2>/dev/null || echo "")
        if [[ "$_PPID" == "1" && "$_EUID" == "$MY_UID" && "$_CMDLINE" == *python* && "$_PID" != "$$" && "$_PID" != "$PPID" ]]; then
          echo "    [orphan] KILL orphan CUDA python pid=$_PID  ($_CMDLINE)"
          kill -9 "$_PID" 2>/dev/null || true
        fi
      done
    fi
  fi
  set -e
  echo "  ---------------------------------------------------------------"
}

_rebuild_done_from_logs() {
  local _BATCH=$1 _PY=$2 _DRY=$3
  local _DONE="${_BATCH}/done"
  echo
  echo "[--rebuild-done] 扫描 LOG_ROOT=${LOG_ROOT} 下完成的项目，重建 DONE 标记到：${_DONE}/"
  mkdir -p "${_DONE}" 2>/dev/null || true
  set +e
  "$_PY" - "$LOG_ROOT" "$_DONE" "$_DRY" <<'PYEOF'
import json, sys, re, pathlib, datetime
log_root, done_dir, dry_str = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
dry = dry_str == "1"
PAT = re.compile(r"^(ETTh1|ETTh2|ETTm1|ETTm2|exchange_rate|weather|electricity|traffic)_s(\d+)_p(\d+)_l(\d+)(?:_20\d{6}_\d{6}|)$")
seen = set()
restored = 0
skipped_exists = 0
skipped_incomplete = 0
if not log_root.exists():
    print(f"  [!] LOG_ROOT 不存在: {log_root}"); sys.exit(0)
# 一级子目录优先（logs/<DS_s_p_l_TIMESTAMP>），其次 logs/<batch>/<DS_s_p_l_*>
candidates = []
for p in sorted(log_root.iterdir()):
    if p.is_dir() and not p.name.startswith('.'):
        candidates.append(p)
# 还加一层（logs/all_runs_*/ 批目录下的子目录）
for p in sorted(log_root.glob("all_runs_*/*")):
    if p.is_dir():
        candidates.append(p)
for d in candidates:
    name = d.name
    m = PAT.match(name)
    if not m:
        continue
    ds, s, p, l = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    run_key = f"{ds}_s{s}_p{p}_l{l}"
    rj = d / "results.json"
    if not rj.exists():
        continue
    # results.json 完成性检查：有 final_best 或 全 4 stages 完成 (stage4 的 best_ckpt_path 存在)
    try:
        obj = json.loads(rj.read_text())
    except Exception:
        skipped_incomplete += 1; continue
    # Do not reuse checkpoints trained with the old mixed-loss protocol.
    mse_only = True
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        cfg = (obj.get(stage) or {}).get("trainer_config") or {}
        if (
            cfg.get("monitor") != "val_mse"
            or cfg.get("lambda_mse") != 1.0
            or any(
                cfg.get(key, 0.0) != 0.0
                for key in (
                    "lambda_nll", "lambda_consensus", "lambda_reject",
                    "lambda_orthogonality", "lambda_agent", "lambda_cov_penalty",
                    "sigma_reg_weight", "lambda_ece", "lambda_crps",
                    "lambda_reject_dist",
                )
            )
        ):
            mse_only = False
            break
    if not mse_only:
        skipped_incomplete += 1
        continue
    complete = False
    if "final_best" in obj and isinstance(obj["final_best"], dict) and obj["final_best"]:
        # final_best 字典非空 → 4 stages 跑完
        complete = True
    else:
        # 退而求其次：4 个 stage 都有记录，且每个 stage 含 best_monitor
        st_ok = 0
        for k in ("stage1", "stage2", "stage3", "stage4"):
            s_obj = obj.get(k) or {}
            if isinstance(s_obj, dict) and s_obj.get("best_ckpt_path"):
                st_ok += 1
        if st_ok == 4:
            complete = True
    if not complete:
        skipped_incomplete += 1
        continue
    if run_key in seen:
        continue
    seen.add(run_key)
    target = done_dir / f"{run_key}.DONE"
    if target.exists():
        skipped_exists += 1
        continue
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if dry:
        print(f"  [dry-run] 会重建：{target.name}  <- {d.name}")
    else:
        target.write_text(f"REBUILT:{ts}\nPROJECT:{d}\n")
        restored += 1
        print(f"  ✅ 重建 DONE：{run_key}.DONE  (from {d.name})")
print(f"  summary: restored={restored}  already_exist={skipped_exists}  incomplete/not_json={skipped_incomplete}")
PYEOF
  set -e
  echo "[--rebuild-done] 完成。"
  echo
}

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo
  awk 'NR>=3 && NR<=60 && /^#/ { sub("^# ?",""); print }' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Script options:  run modes (pick exactly one of: --dataset X | --all | --datasets "...")
  Single-dataset (classic):
    --dataset <DS>             ETTh1|ETTh2|ETTm1|ETTm2|weather|electricity|exchange_rate|traffic
                                 + 配 --mode single: DS 只跑 1 个 setting (seq_len × pred_len)
                                 + 默认 配 --mode full: DS 循环 12 个标准 setting = {96,336,512} × {96,192,336,720}
                                                     中途 Ctrl+C 随时停；下次重传 SAME cli 自动 resume 跳过已完成的 run。

  Multi-dataset (either, not both):
    --all                      Run all 4 standard datasets with MSE-only training:
                                 ETTh1 ETTh2 weather exchange_rate
                               (配 --mode single: 4×1=4 run；配 --mode full: 4×12=48 run)
                               (traffic excluded; use --force-traffic --dataset traffic to run it)
    --datasets "DS1 DS2 ..."   Run a user-defined list of datasets (--mode single/full 同上).

  Run-mode flag (applies to all datasets):
    --mode single|full         single=1 setting per DS; full=12 顶会标准组合 per DS (默认).
                               建议：想随时可续跑请配 --mode full，哪怕单个 dataset。

  Common / setting flags:
    --gpu-tier <24G|8G>        (必填) GPU 显存等级（两档锁模型，唯 BS×accum 调）
    --seq-len  <96|336|512>    single-mode lookback；full-mode 覆盖 FULL_SEQS，传了会 WARNING 忽略
    --pred-len <96|192|336|720> single-mode horizon；full-mode 覆盖 FULL_PREDS，传了会 WARNING 忽略
    --label-len <N>            decoder start token，默认 pred_len//2（TS-Lib 惯例）

  Execution flags:
    --no-vram-check            跳过 estimate_vram_fast.py VRAM 预检
    --force-traffic            traffic 默认不推荐，传此 flag 才放行
    --allow-nonstandard-setting  single-mode: 允许 seq/pred 非顶会标准 12 对（会触发 main.py stderr WARNING）
    --no-sota-flags            关闭 RevIN / DLinear-trend-head / σ_init=1.0 等 SOTA 升级
    --dry-run                  只打印命令，不执行（用于参数审查）；full-mode 每个数据集 12 run，--all 共 48 run。
    --continue-on-fail         某 run/dataset 失败后继续跑后续 run/dataset（默认 ON）。
    --stop-on-fail             严格模式：首个失败即停（覆盖默认 ON 行为，等价旧 CONTINUE_ON_FAIL=0）。
    --rerun                    默认 resume：batch dir 里已写 <key>.DONE 的 run 会 SKIP。传 --rerun 强制重跑全部。
    --rebuild-done             启动训练前，扫描 logs/ 下所有 <DS>_s<SEQ>_p<PRED>_l<LAB>_* 项目目录，
                                 找到 results.json 存在且含 final_best (4 stages 全跑完 exit=0) 的项目，
                                 自动重建 <BATCH_DIR>/done/<RUN_KEY>.DONE，让断点续训恢复 SKIP 能力。
                                 用于 done 被误删后恢复（比如之前 --rerun --dry-run 真删了 done）。
    --extra '...'              额外透传给 src.experiments.main 的参数（所有子 run 共用）
    --tag-suffix '...'         透传给 main.py --tag-suffix，所有子 run 共用（区分 ablation/seeds；也作为 resume batch key 的一部分）
    --py <bin>                 Python 解释器，默认：$PYTHON 或项目 .venv/bin/python 或 PATH python3 (当前=${PY_CMD})
    -h / --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)                   usage; exit 0 ;;
    --dataset)                   DATASET="$2"; shift 2 ;;
    --all)                       MULTI_MODE="all"; shift ;;
    --datasets)                  MULTI_MODE="list"; DATASETS_LIST="$2"; shift 2 ;;
    --mode)
      case "$2" in
        single|full) RUN_MODE="$2" ;;
        *) die "--mode 只能是 single 或 full，收到 $2" ;;
      esac; shift 2 ;;
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
    --stop-on-fail)              CONTINUE_ON_FAIL=0; shift ;;
    --rerun)                     RERUN=1; shift ;;
    --rebuild-done)              REBUILD_DONE=1; shift ;;
    --extra)                     EXTRA_ARGS="$2"; shift 2 ;;
    --tag-suffix)                TAG_SUFFIX="$2"; shift 2 ;;
    --run-tag-suffix)            TAG_SUFFIX="$2"; shift 2 ;;   # alias
    --py)                        PY="$2"; shift 2 ;;
    --_batch-dir)                INTERNAL_BATCH_DIR="$2"; shift 2 ;;
    --)                          shift; EXTRA_ARGS="$EXTRA_ARGS $*"; break ;;
    *)                           die "Unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "$GPU_TIER"  ]] || die "--gpu-tier 未传。仅支持: 32G, 24G, 8G"
case "$GPU_TIER" in
  32G|24G|8G) ;;
  *) die "--gpu-tier 当前只支持 32G / 24G / 8G。你日常 32G 显存用 --gpu-tier 32G（VRAM 预检 CAP=28.0G，日常顶到 28G 不超 32G）。" ;;
esac

if [[ "$RUN_MODE" == "full" ]]; then
  if [[ -n "$SEQ_LEN" || -n "$PRED_LEN" ]]; then
    echo "[WARNING --mode full] --seq-len / --pred-len 被忽略（full-mode 自动循环 FULL_SEQS $FULL_SEQS × FULL_PREDS $FULL_PREDS 共 12 setting）。" >&2
    SEQ_LEN=""; PRED_LEN=""
  fi
fi

if [[ "$MULTI_MODE" == "all" || "$MULTI_MODE" == "list" ]]; then
  if [[ -n "$DATASET" ]]; then
    die "multi dataset mode (--all / --datasets) 与 --dataset 互斥，只能二选一。"
  fi
  if [[ "$MULTI_MODE" == "all" ]]; then
    DATASETS_LIST="$ALL_DATASETS"
  fi
  read -r -a DS_ARRAY <<<"$DATASETS_LIST"
  if [[ "${#DS_ARRAY[@]}" -eq 0 ]]; then
    die "--datasets 列表为空。"
  fi

  if [[ -n "$TAG_SUFFIX" ]]; then
    SUFFIX_KEY="_${TAG_SUFFIX}"
  else
    SUFFIX_KEY=""
  fi
  BATCH_DIR_NAME="all_runs_${MULTI_MODE}_${RUN_MODE}_${GPU_TIER}_mse${SUFFIX_KEY}"
  BATCH_DIR="${LOG_ROOT}/${BATCH_DIR_NAME}"
  DONE_DIR="${BATCH_DIR}/done"
  mkdir -p "${BATCH_DIR}" "${DONE_DIR}"
  SUMMARY_JSON="${BATCH_DIR}/summary.json"
  if [[ ! -f "$SUMMARY_JSON" ]]; then
    cat >"$SUMMARY_JSON" <<EOF
{
  "batch_dir": "${BATCH_DIR_NAME}",
  "gpu_tier": "${GPU_TIER}",
  "multi_mode": "${MULTI_MODE}",
  "run_mode": "${RUN_MODE}",
  "datasets": $("${PY_CMD}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${DS_ARRAY[@]}"),
  "tag_suffix": $("${PY_CMD}" -c 'import json,sys; print(json.dumps(sys.argv[1] if len(sys.argv)>1 else None))' "$TAG_SUFFIX"),
  "run_order": [],
  "runs": {}
}
EOF
  fi

  echo "================================================================="
  echo " TACF multi-dataset BATCH RUN  (${#DS_ARRAY[@]} datasets)  run_mode=${RUN_MODE}"
  echo "   mode     : ${MULTI_MODE}"
  echo "   run_mode : ${RUN_MODE}  (single=1 setting/DS, full=12 settings/DS)"
  echo "   datasets : ${DS_ARRAY[*]}"
  echo "   tier     : ${GPU_TIER}"
  echo "   setting  : seq=${SEQ_LEN_DEF} pred=${PRED_LEN_DEF} (per-run CLI 覆盖生效)"
  echo "   tag-suf  : ${TAG_SUFFIX:-<none>}"
  echo "   extra    : ${EXTRA_ARGS:-<none>}"
  echo "   rerun    : $([[ $RERUN -eq 1 ]] && echo 'FORCE RE-RUN（删 done/*.DONE 重跑全部）' || echo 'RESUME 默认：已有 DONE 标记的 run SKIP')"
  echo "   continue : $([[ $CONTINUE_ON_FAIL -eq 1 ]] && echo 'ON (失败继续)' || echo 'OFF (首个失败即停)')"
  echo "   batch dir: ${BATCH_DIR}"
  [[ "$RUN_MODE" == "full" ]] && \
  echo "   total runs: ${#DS_ARRAY[@]} datasets × 12 settings = $(( ${#DS_ARRAY[@]} * 12 )) runs" || \
  echo "   total runs: ${#DS_ARRAY[@]} datasets × 1 setting  = ${#DS_ARRAY[@]} runs"
  echo "   [断点续跑] 重跑同一条命令 = 自动跳过已有 DONE 标记的 run；清缓存 rm -rf ${DONE_DIR}/* 或传 --rerun"
  echo "================================================================="

  if [[ $RERUN -eq 1 ]]; then
    echo "[--rerun] 清理已有 DONE 标记：${DONE_DIR}/*.DONE"
    if [[ $DRY_RUN -ne 1 ]]; then
      rm -f "${DONE_DIR}"/*.DONE 2>/dev/null || true
    else
      echo "  [dry-run] skip rm -f ${DONE_DIR}/*.DONE（dry-run 不删任何文件）"
    fi
  fi

  if [[ $REBUILD_DONE -eq 1 ]]; then
    _rebuild_done_from_logs "${BATCH_DIR}" "${PY_CMD}" "$DRY_RUN"
  fi

  EXIT_TOTAL=0
  DS_RUN_IDX=0
  for DS in "${DS_ARRAY[@]}"; do
    DS_RUN_IDX=$((DS_RUN_IDX+1))
    echo
    echo "#################################################################"
    echo " #${DS_RUN_IDX}/${#DS_ARRAY[@]}  START  dataset=${DS}"
    echo "#################################################################"
    SUB_ARGS=( --dataset "$DS" --gpu-tier "$GPU_TIER" --mode "$RUN_MODE" )
    [[ -n "$SEQ_LEN"       ]] && SUB_ARGS+=( --seq-len       "$SEQ_LEN" )
    [[ -n "$PRED_LEN"      ]] && SUB_ARGS+=( --pred-len      "$PRED_LEN" )
    [[ -n "$LABEL_LEN"     ]] && SUB_ARGS+=( --label-len     "$LABEL_LEN" )
    [[ $NO_VRAM_CHECK      -eq 1 ]] && SUB_ARGS+=( --no-vram-check )
    [[ $FORCE_TRAFFIC      -eq 1 ]] && SUB_ARGS+=( --force-traffic )
    [[ $ALLOW_NONSTANDARD  -eq 1 ]] && SUB_ARGS+=( --allow-nonstandard-setting )
    [[ $NO_SOTA_FLAGS      -eq 1 ]] && SUB_ARGS+=( --no-sota-flags )
    [[ $DRY_RUN            -eq 1 ]] && SUB_ARGS+=( --dry-run )
    [[ $CONTINUE_ON_FAIL   -eq 1 ]] && SUB_ARGS+=( --continue-on-fail )
    [[ $RERUN              -eq 1 ]] && SUB_ARGS+=( --rerun )
    [[ -n "$EXTRA_ARGS"    ]] && SUB_ARGS+=( --extra "$EXTRA_ARGS" )
    [[ -n "$TAG_SUFFIX"    ]] && SUB_ARGS+=( --tag-suffix "$TAG_SUFFIX" )
    [[ -n "${PY_CMD}" ]] && SUB_ARGS+=( --py "${PY_CMD}" )
    SUB_ARGS+=( --_batch-dir "${BATCH_DIR}" )
    set +e
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run][$DS] 子命令: bash $0 ${SUB_ARGS[*]}"
      SUB_EXIT=0
    else
      bash "$0" "${SUB_ARGS[@]}"
      SUB_EXIT=$?
    fi
    set -e
    if [[ $SUB_EXIT -ne 0 ]]; then
      EXIT_TOTAL=$(( EXIT_TOTAL + 1 ))
      echo "⚠️   dataset=${DS}  exit=${SUB_EXIT}" >&2
      if [[ $CONTINUE_ON_FAIL -eq 0 ]]; then
        echo "已停止后续数据集（传 --continue-on-fail 可忽略错误继续跑）。" >&2
        "${PY_CMD}" - "$SUMMARY_JSON" "$DS" "$SUB_EXIT" "ABORTED" <<'PY'
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
    "${PY_CMD}" - "$SUMMARY_JSON" "$DS" "$SUB_EXIT" <<'PY'
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
  echo " 断点续跑提示：再次运行 SAME CLI = 自动跳过已 <done/*DONE> 的 run；传 --rerun 清所有 DONE 从头来"
  echo "================================================================="
  exit $EXIT_TOTAL
fi
_BATCH_DIR=""
RUN_EXIT_TOTAL=0
if [[ -n "${INTERNAL_BATCH_DIR:-}" ]]; then
  _BATCH_DIR="${INTERNAL_BATCH_DIR}"
fi

[[ -n "$DATASET"   ]] || die "请传 --dataset <DS> 或 --all / --datasets \"DS1 DS2 ...\"。全量数据集: $ALL_DATASETS"
if [[ -z "$_BATCH_DIR" ]]; then
  if [[ -n "$TAG_SUFFIX" ]]; then SUFFIX_KEY="_${TAG_SUFFIX}"; else SUFFIX_KEY=""; fi
  if [[ "${RUN_MODE:-full}" == "full" ]]; then
    BATCH_DIR_NAME="all_runs_all_${RUN_MODE}_${GPU_TIER}_mse${SUFFIX_KEY}"
  else
    _SINGLE_S="${SEQ_LEN:-$SEQ_LEN_DEF}"
    _SINGLE_P="${PRED_LEN:-$PRED_LEN_DEF}"
    BATCH_DIR_NAME="one_ds_${DATASET}_${RUN_MODE}_${GPU_TIER}_s${_SINGLE_S}_p${_SINGLE_P}_mse${SUFFIX_KEY}"
  fi
  _BATCH_DIR="${LOG_ROOT}/${BATCH_DIR_NAME}"
fi
_DONE_DIR="${_BATCH_DIR}/done"
mkdir -p "${_BATCH_DIR}" "${_DONE_DIR}"
_ONE_SUMMARY="${_BATCH_DIR}/summary_one.json"
if [[ ! -f "$_ONE_SUMMARY" ]]; then
  cat >"$_ONE_SUMMARY" <<EOF
{
  "dataset": "${DATASET}",
  "gpu_tier": "${GPU_TIER}",
  "run_mode": "${RUN_MODE}",
  "tag_suffix": $("${PY_CMD}" -c 'import json,sys; print(json.dumps(sys.argv[1] if len(sys.argv)>1 else None))' "$TAG_SUFFIX"),
  "run_order": [],
  "runs": {}
}
EOF
fi

echo
echo "================================================================="
echo " [batch dir: ${_BATCH_DIR}]"
echo " dataset=${DATASET}  run_mode=${RUN_MODE}  tier=${GPU_TIER}"
echo " continue : $([[ $CONTINUE_ON_FAIL -eq 1 ]] && echo 'ON (失败继续)' || echo 'OFF (首个失败即停)')"
echo " resume default: SKIP <done/*.DONE> 已存在的 run (传 --rerun 强制重跑)"
[[ "$RUN_MODE" == "full" ]] && echo " full settings: FULL_SEQS=$FULL_SEQS  FULL_PREDS=$FULL_PREDS  (12 runs/DS)"
[[ "$RUN_MODE" == "single" ]] && echo " single setting: seq=${SEQ_LEN_DEF} pred=${PRED_LEN_DEF} (用户 CLI 覆盖优先)"
echo "================================================================="

if [[ $RERUN -eq 1 ]]; then
  echo "[--rerun] 清理已有 DONE 标记：${_DONE_DIR}/*.DONE"
  if [[ $DRY_RUN -ne 1 ]]; then
    rm -f "${_DONE_DIR}"/*.DONE 2>/dev/null || true
  else
    echo "  [dry-run] skip rm -f ${_DONE_DIR}/*.DONE（dry-run 不删任何文件）"
  fi
fi

if [[ $REBUILD_DONE -eq 1 ]]; then
  _rebuild_done_from_logs "${_BATCH_DIR}" "${PY_CMD}" "$DRY_RUN"
fi

declare -a RUN_JOBS=()
if [[ "$RUN_MODE" == "single" ]]; then
  _S="${SEQ_LEN:-$SEQ_LEN_DEF}"
  _P="${PRED_LEN:-$PRED_LEN_DEF}"
  _L="${LABEL_LEN:-$(( _P / 2 ))}"
  RUN_JOBS+=( "S:${_S}:${_P}:${_L}" )
else
  for _S in $FULL_SEQS; do
    for _P in $FULL_PREDS; do
      _L="${LABEL_LEN:-$(( _P / 2 ))}"
      RUN_JOBS+=( "S:${_S}:${_P}:${_L}" )
    done
  done
fi

echo
echo ">>>>>> 本 dataset 待跑 run 数: ${#RUN_JOBS[@]}（Ctrl+C 可中断；重传 SAME CLI = 自动 resume 跳过 DONE）"
echo

RUNC=0
for JOB in "${RUN_JOBS[@]}"; do
  IFS=':' read -r _T _S _P _L <<<"$JOB"
  RUNC=$((RUNC+1))
  RUN_KEY="${DATASET}_s${_S}_p${_P}_l${_L}"
  if [[ -n "$TAG_SUFFIX" ]]; then
    RUN_KEY="${RUN_KEY}_${TAG_SUFFIX}"
  fi
  DONE_FILE="${_DONE_DIR}/${RUN_KEY}.DONE"

  echo
  echo " -----------------------------------------------------------------"
  echo "  [${RUNC}/${#RUN_JOBS[@]}] run ${RUN_KEY}"
  echo "  -----------------------------------------------------------------"
  if [[ -f "$DONE_FILE" ]]; then
    echo "  ✅ SKIP （已找到 DONE 标记: ${DONE_FILE}）。想重跑传 --rerun 或 rm -f ${DONE_FILE}"
    "${PY_CMD}" - "$_ONE_SUMMARY" "$RUN_KEY" 0 <<'PY'
import json, sys, pathlib
p, rk, rc = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
obj = json.loads(p.read_text())
obj.setdefault("run_order", [])
if rk not in obj["run_order"]: obj["run_order"].append(rk)
st = obj["runs"].get(rk, {})
if not st:
    obj["runs"][rk] = {"run_key": rk, "exit_code": 0, "status": "DONE (resume-skip)"}
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
PY
    continue
  fi

  _SOTA_FLAGS_SINGLE=""
  _HYPERS_SINGLE=""
  _VRAMBS_SINGLE=""
  set +eu
  _SOTA_FLAGS_SINGLE=$(_build_sota_flags "$NO_SOTA_FLAGS")
  _HYPERS_SINGLE=$(_build_hypers "$DATASET" "$GPU_TIER" "$_S" "$_P" "$_L" "$NO_SOTA_FLAGS")
  _VRAMBS_SINGLE=$(_build_phys_bs "$DATASET" "$GPU_TIER" "$_P" "$_S")
  set -eu
  _PY_CMD_SINGLE="${PY_CMD}"

  if [[ $DRY_RUN -ne 1 ]]; then
    _vram_check_wrapper "$DATASET" "$GPU_TIER" "$NO_VRAM_CHECK" "$FORCE_TRAFFIC" "$_VRAMBS_SINGLE"
  fi

  _SLOGAN_BAR_SINGLE=""
  _SLOGAN_BAR_SINGLE=$(_print_trainset_slogan \
      "$DATASET" "$GPU_TIER" "$_S" "$_P" "$_L" \
      "$_VRAMBS_SINGLE" "$NO_SOTA_FLAGS" "$NO_VRAM_CHECK" "$FORCE_TRAFFIC")

  MAIN_ARGS_SINGLE=(
    -u -m src.experiments.main
    --dataset "$DATASET"
  )
  if [[ -n "$_HYPERS_SINGLE" ]]; then
    MAIN_ARGS_SINGLE+=( $_HYPERS_SINGLE )
  fi
  if [[ -n "$_SOTA_FLAGS_SINGLE" ]]; then
    if [[ "$_HYPERS_SINGLE" != *"--use-revin"* ]]; then
      MAIN_ARGS_SINGLE+=( $_SOTA_FLAGS_SINGLE )
    fi
  fi
  if [[ -n "$_VRAMBS_SINGLE" ]]; then
    MAIN_ARGS_SINGLE+=( $_VRAMBS_SINGLE )
  fi
  if [[ -n "$TAG_SUFFIX" ]] && [[ "${TAG_SUFFIX}" != '""' ]]; then
    MAIN_ARGS_SINGLE+=( --tag-suffix "$TAG_SUFFIX" )
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    MAIN_ARGS_SINGLE+=( $EXTRA_ARGS )
  fi

  set +e
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "${_SLOGAN_BAR_SINGLE}"
    _cleanup_before_run 1 "${_PY_CMD_SINGLE}"
    echo "  [dry-run][${RUN_KEY}] main.py 命令: ${_PY_CMD_SINGLE} ${MAIN_ARGS_SINGLE[*]}"
    SUB_EXIT2=0
  else
    echo "${_SLOGAN_BAR_SINGLE}"
    _cleanup_before_run 0 "${_PY_CMD_SINGLE}"
    echo "🚀 启动训练: ${_PY_CMD_SINGLE} ${MAIN_ARGS_SINGLE[*]}"
    echo
    "${_PY_CMD_SINGLE}" "${MAIN_ARGS_SINGLE[@]}"
    SUB_EXIT2=$?
  fi
  set -e

  if [[ $SUB_EXIT2 -eq 0 ]]; then
    if [[ $DRY_RUN -ne 1 ]]; then
      echo "OK:${RUN_KEY}" >"${DONE_FILE}"
    fi
    echo "  ✅ ${RUN_KEY} done (exit=0) → ${DONE_FILE}"
  else
    RUN_EXIT_TOTAL=$((RUN_EXIT_TOTAL + 1))
    echo "  ❌ ${RUN_KEY} FAIL exit=${SUB_EXIT2}" >&2
    if [[ $CONTINUE_ON_FAIL -eq 0 ]]; then
      echo "停止本 dataset 后续 setting（传 --continue-on-fail 继续跑下一个 setting）。" >&2
      "${PY_CMD}" - "$_ONE_SUMMARY" "$RUN_KEY" "$SUB_EXIT2" "ABORTED" <<'PY'
import json, sys, pathlib
p, rk, rc, status = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
obj = json.loads(p.read_text())
obj.setdefault("run_order", [])
if rk not in obj["run_order"]: obj["run_order"].append(rk)
obj["runs"][rk] = {"run_key": rk, "exit_code": rc, "status": status}
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
PY
      exit $RUN_EXIT_TOTAL
    fi
  fi
  set +e
  "${PY_CMD}" - "$_ONE_SUMMARY" "$RUN_KEY" "$SUB_EXIT2" <<'PY'
import json, sys, pathlib
p, rk, rc = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
obj = json.loads(p.read_text())
obj.setdefault("run_order", [])
if rk not in obj["run_order"]: obj["run_order"].append(rk)
if rk not in obj["runs"] or obj["runs"][rk].get("status","") in ("FAILED","ABORTED",""):
    obj["runs"][rk] = {
      "run_key": rk,
      "exit_code": rc,
      "status": "OK" if rc == 0 else ("FAILED" if rc != 0 else "OK"),
    }
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
PY
  _SM=$?
  if [[ $_SM -ne 0 ]]; then
    echo "[summary warning] python3/summary 失败 (exit $_SM，记录 summary 非致命)" >&2
  fi
  set -e
done

echo
echo "================================================================="
echo " dataset=${DATASET} 跑程:  done=$(( ${#RUN_JOBS[@]} - RUN_EXIT_TOTAL )) / fail=${RUN_EXIT_TOTAL} / total=${#RUN_JOBS[@]}"
echo " one-ds summary: ${_ONE_SUMMARY}"
echo "================================================================="

if [[ $RUN_EXIT_TOTAL -ne 0 ]]; then
  echo "共 ${RUN_EXIT_TOTAL} 个 run 失败。" >&2
  exit $RUN_EXIT_TOTAL
fi
exit 0
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

if [[ "$DATASET" == "traffic" && "$FORCE_TRAFFIC" -eq 0 ]]; then
  die "traffic (D=862) 数据集太大，early iteration 暂不建议跑。配置文件保留: src/configs/scales/traffic_{24g,8g}.yaml。
      如果确需在 traffic 上试验，请传 --force-traffic 绕过此检查。"
fi

if [[ "$NO_SOTA_FLAGS" -eq 1 ]]; then
  SOTA_FLAGS=""
else
  SOTA_FLAGS="--use-revin --use-dlinear-trend-head --sigma-global-multiplier-init 1.0"
fi
ETT_STAGE_COMMON="--d-model 768 --agg-d-model 384 --n-layers 6 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 1e-3 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"
WEATHER_STAGE_COMMON="--d-model 768 --agg-d-model 384 --n-layers 6 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 8e-4 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"
ELEC_STAGE_COMMON="--d-model 576 --agg-d-model 288 --n-layers 4 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 6e-4 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"
EXCHANGE_STAGE_COMMON="--d-model 768 --agg-d-model 384 --n-layers 6 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 10 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 1e-3 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"
TRAFFIC_STAGE_COMMON="--d-model 384 --agg-d-model 192 --n-layers 3 \
 --max-epochs-stage1 60 --max-epochs-stage2 25 --max-epochs-stage3 30 --max-epochs-stage4 40 \
 --early-stop 15 --stage4-lr-mult 0.10 --stage4-early-stop 12 --amp --lr 6e-4 \
 --seq-len ${SEQ_LEN} --pred-len ${PRED_LEN} --label-len ${LABEL_LEN} \
 ${SOTA_FLAGS}"

case "${DATASET}:${GPU_TIER}" in
  ETTh1:24G|ETTh2:24G|ETTm1:24G|ETTm2:24G)
    MAIN_ARGS="${ETT_STAGE_COMMON} --batch-size 16 --grad-accum 4"
    EFF_BS=$((16 * 4))
    ;;
  ETTh1:8G|ETTh2:8G|ETTm1:8G|ETTm2:8G)
    MAIN_ARGS="${ETT_STAGE_COMMON} --batch-size 4 --grad-accum 16"
    EFF_BS=$((4 * 16))
    ;;
  weather:24G)
    MAIN_ARGS="${WEATHER_STAGE_COMMON} --batch-size 12 --grad-accum 4"
    EFF_BS=$((12 * 4))
    ;;
  weather:8G)
    MAIN_ARGS="${WEATHER_STAGE_COMMON} --batch-size 3 --grad-accum 16"
    EFF_BS=$((3 * 16))
    ;;
  electricity:24G)
    MAIN_ARGS="${ELEC_STAGE_COMMON} --batch-size 5 --grad-accum 10"
    EFF_BS=$((5 * 10))
    ;;
  electricity:8G)
    MAIN_ARGS="${ELEC_STAGE_COMMON} --batch-size 2 --grad-accum 25"
    EFF_BS=$((2 * 25))
    ;;
  exchange_rate:24G)
    MAIN_ARGS="${EXCHANGE_STAGE_COMMON} --batch-size 32 --grad-accum 2"
    EFF_BS=$((32 * 2))
    ;;
  exchange_rate:8G)
    MAIN_ARGS="${EXCHANGE_STAGE_COMMON} --batch-size 8 --grad-accum 8"
    EFF_BS=$((8 * 8))
    ;;
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
if [[ "${NO_VRAM_CHECK}" -eq 0 && -f estimate_vram_fast.py ]]; then
  echo "--- estimate_vram_fast.py VRAM 预检 (按 physical batch size=${PHYS_BS} 估算) ---"
  "${PY_CMD}" estimate_vram_fast.py --datasets "${DATASET}" \
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
TAG_SUFFIX_MAIN_ARGS=""
if [[ -n "$TAG_SUFFIX" ]] && [[ "${TAG_SUFFIX}" != '""' ]]; then
  TAG_SUFFIX_MAIN_ARGS=" --tag-suffix ${TAG_SUFFIX}"
fi

CMD="${PY_CMD} -u -m src.experiments.main --dataset ${DATASET} ${MAIN_ARGS}${TAG_SUFFIX_MAIN_ARGS} ${EXTRA_ARGS}"
echo "🚀 启动训练: ${CMD}"
echo

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] 实际将执行（已跳过）:" >&2
  echo "  cd ${SCRIPT_DIR} && ${CMD}" >&2
  exit 0
fi

exec "${PY_CMD}" -u -m src.experiments.main --dataset "${DATASET}" ${MAIN_ARGS}${TAG_SUFFIX_MAIN_ARGS} ${EXTRA_ARGS}
