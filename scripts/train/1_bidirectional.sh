#!/bin/bash
# Stage 1: bidirectional teacher, from the official Wan2.2-TI2V-5B weights (released as weights/bidirectional/dit_step35000).
#   bash scripts/train/1_bidirectional.sh <out dir> [max steps]
# Batch 8 (GRAD_ACCUM x NGPU), lr 1e-5 / condition layers 1e-4, seed 0, 35k steps, all data including the world-edit clips (WEDIT=1) from step 0.
set -e; source "$(dirname "$0")/common.sh"
WEDIT=1 launch fsdp.yaml wizard.train.bidirectional --data $D/windows --out $1 --max_steps ${2:-35000} --save_every 1000 --num_workers 6 \
  --grad_accum $GRAD_ACCUM $MIX --mem_loss_w 3 --mem_index mem_index_both.pt --p_memunk 0 --p_trueunk 0 --p_unk2 0.5 \
  --extra_rows $ROWS,$WEDIT_ROWS
