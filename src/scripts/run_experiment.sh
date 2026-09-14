#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# TACF MOA — main experiment launcher
# ----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------------------------------------ configurable defaults
DATASET="${DATASET:-ETTh1}"
SEQ_LEN="${SEQ_LEN:-336}"
PRED_LEN="${PRED_LEN:-168}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
MAX_EPOCHS_S1="${MAX_EPOCHS_S1:-30}"
MAX_EPOCHS_S2="${MAX_EPOCHS_S2:-10}"
MAX_EPOCHS_S3="${MAX_EPOCHS_S3:-10}"
MAX_EPOCHS_S4="${MAX_EPOCHS_S4:-20}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
D_MODEL="${D_MODEL:-512}"
N_LAYERS="${N_LAYERS:-4}"
USE_AMP="${USE_AMP:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python binary not found at: $PYTHON_BIN" >&2
  exit 1
fi

# ----------------------------------------------------------------- print header
echo "================================================================"
echo " TACF Mixture-of-Agents experiment"
echo "================================================================"
echo " dataset        = $DATASET"
echo " seq_len/pred   = $SEQ_LEN -> $PRED_LEN"
echo " batch_size     = $BATCH_SIZE"
echo " lr             = $LR"
echo " d_model/layers = $D_MODEL / $N_LAYERS"
echo " epochs (s1..s4)= $MAX_EPOCHS_S1 / $MAX_EPOCHS_S2 / $MAX_EPOCHS_S3 / $MAX_EPOCHS_S4"
echo " seed           = $SEED"
echo " venv python    = $PYTHON_BIN"
echo " logs           = $LOG_DIR"
echo " amp            = $( (( USE_AMP == 1 )) && echo yes || echo no)"
echo "================================================================"

AMP_FLAG=()
(( USE_AMP == 1 )) && AMP_FLAG=(--amp) || true

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m src.experiments.main \
  --dataset "$DATASET" \
  --seq-len  "$SEQ_LEN"  \
  --pred-len "$PRED_LEN" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR"               \
  --seed "$SEED"           \
  --log-dir "$LOG_DIR"     \
  --max-epochs-stage1 "$MAX_EPOCHS_S1" \
  --max-epochs-stage2 "$MAX_EPOCHS_S2" \
  --max-epochs-stage3 "$MAX_EPOCHS_S3" \
  --max-epochs-stage4 "$MAX_EPOCHS_S4" \
  --d-model "$D_MODEL"     \
  --n-layers "$N_LAYERS"   \
  "${AMP_FLAG[@]}"         \
  $EXTRA_ARGS
