#!/bin/bash
# Stage 2: causal teacher, teacher forcing with history self-augmentation, from the bidirectional teacher (released as weights/causal_teacher/dit_step11000).
#   bash scripts/train/2_causal.sh <bidirectional dit_step35000> <out dir> [max steps]
# History window 20 + sink 1; batch 8 (ours: 8 GPUs, from 4000 on 4 GPUs x --grad_accum 2); lr 1e-5 (condition layers 1e-5); seed 0.
# Self-augmentation SA_P 0.5, sigma <= 0.25 on the revisit groups; memory loss weight 3 on cells no clean history frame the noisy frame
# reads has seen (MEMW_HISTW / MEMW_SINK); SceneFly memory from before and after the window (MEM_FUTURE=1); the world-edit clips (WEDIT=1).
# Ours ran to 11000, resumed from its own checkpoints (at 4000; the run was planned for 10000 and extended): to continue,
#   INIT=<out>/dits/dit_step<N>.safetensors STEP0=<N> bash scripts/train/2_causal.sh ...   (seed = N, as ours)
set -e; source "$(dirname "$0")/common.sh"
export WEDIT=1 MEMW_HISTW=20 MEMW_SINK=1 MEM_FUTURE=1 SA_P=${SA_P:-0.5} SA_MAX=${SA_MAX:-0.25} SA_GROUPS=${SA_GROUPS:-lookback,lookback0,sfly,sfly_replay}
STEP0=${STEP0:-0}
launch fsdp.yaml wizard.train.causal_selfaug --data $D/windows --out $2 --init ${INIT:-$1} --step0 $STEP0 --seed $STEP0 --max_steps ${3:-11000} \
  --save_every 500 --num_workers 6 --window 20 --sink 1 $MIX,lookback:4,lookback0:1 --p_memunk 0 --mem_loss_w 3 --mem_index mem_index_both.pt \
  --extra_rows $ROWS,$LOOKBACK_ROWS,$WEDIT_ROWS --p_trueunk 0 --p_unk2 0.5 --log_every 5 --grad_accum $GRAD_ACCUM
