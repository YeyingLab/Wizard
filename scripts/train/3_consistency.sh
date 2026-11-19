#!/bin/bash
# Stage 3: consistency distillation into a 2-step causal student (teacher reads the clean history, student / EMA the self-augmented one).
#   bash scripts/train/3_consistency.sh <causal teacher dit_step11000> <out dir> [max steps]
# Teacher CFG 3, N 48, lr 2e-6, EMA 0.99 from step 200, window 20 + sink 1, 8 GPUs x 1 window; memory candidates past + future as the
# teacher was trained (wizard.distill.consistency_selfaug_future), the world-edit clips (WEDIT=1); SA_P 0.7, sigma <= 0.5 on the revisit groups.
# Ours: 0 -> 2700 (seed 0; only the EMA was saved), then warm-started from ema_step2000 to 10000:
#   STUDENT_INIT=<out>/ema/ema_step2000.safetensors STEP0=2000 SEED=1 bash scripts/train/3_consistency.sh <teacher> <out2>
# (student and EMA both start from that EMA, the teacher stays the causal teacher, the warm-up restarts).
set -e; source "$(dirname "$0")/common.sh"
export WEDIT=1 SA_P=${SA_P:-0.7} SA_MAX=${SA_MAX:-0.5} SA_GROUPS=${SA_GROUPS:-lookback,lookback0,sfly,sfly_replay}
WARM=""; [ -n "$STUDENT_INIT" ] && WARM="--student_init $STUDENT_INIT --step0 ${STEP0:-0}"
launch fsdp_dmd.yaml wizard.distill.consistency_selfaug_future --data $D/windows --ckpt $1 --out $2 --max_steps ${3:-10000} --save_every 1000 \
  --cfg 3 --window 20 --sink 1 --seed ${SEED:-0} $MIX,lookback:4,lookback0:1 --mem_index mem_index_both.pt --extra_rows $ROWS,$LOOKBACK_ROWS,$WEDIT_ROWS \
  --p_trueunk 0 --p_unk2 0.5 $WARM
