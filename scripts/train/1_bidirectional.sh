#!/bin/bash
# Stage 1: bidirectional teacher, from the official Wan2.2-TI2V-5B weights (released as weights/bidirectional/dit_step35000).
#   bash scripts/train/1_bidirectional.sh <out dir>            both phases;  PHASE=1 or PHASE=2 runs one of them
# Batch 8 (ours: 4 GPUs x --grad_accum 2), lr 1e-5 / condition layers 1e-4, seed 0, the same data mixing from step 0.
# Phase 1 (0 -> 3000) without the world-edit clips; phase 2 (3000 -> 35000) restarts from the 3000 checkpoint with WEDIT=1 and the
# world-edit rows added (the clips became available then; the warm-up restarts with the run).
set -e; source "$(dirname "$0")/common.sh"; OUT=$1
ARGS="--data $D/windows --out $OUT --save_every 1000 --num_workers 6 --grad_accum $GRAD_ACCUM $MIX --mem_loss_w 3 --mem_index mem_index_both.pt \
  --p_memunk 0 --p_trueunk 0 --p_unk2 0.5"
if [ "${PHASE:-1}" = 1 ]; then
  launch fsdp.yaml wizard.train.bidirectional $ARGS --max_steps 3000 --extra_rows $ROWS
fi
if [ -z "$PHASE" ] || [ "$PHASE" = 2 ]; then
  WEDIT=1 launch fsdp.yaml wizard.train.bidirectional $ARGS --init $OUT/dits/dit_step3000.safetensors --step0 3000 --max_steps 35000 \
    --extra_rows $ROWS,$WEDIT_ROWS
fi
