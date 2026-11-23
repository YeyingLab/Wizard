#!/bin/bash
# Stage 4: DMD of the 2-step student; real score and the fake score's start = the bidirectional teacher (CFG 3). Batch = 8 GPUs x accum 4 = 32,
# lr 2e-6 / critic 4e-7, 5 critic updates per student update, levels 1000,500. Memory candidates past + future as the teacher and the
# consistency student (wizard.distill.dmd_future, mem_index_both.pt), the world-edit clips (WEDIT=1, repeated 16x), heavier mixing of the entry /
# revisit groups.
#   bash scripts/train/4_dmd.sh <consistency ema_step10000> <bidirectional dit_step35000> <out dir> [max steps]
# Ours: 500 steps, saving every 50 (an EMA, student_last and fake_last each time). To continue a run from its save at step N:
#   EXTRA="--ema_init <out>/ema/ema_step<N>.safetensors --fake <out>/fake_last.safetensors --step0 <N>" \
#     bash scripts/train/4_dmd.sh <out>/student_last.safetensors ...
set -e; source "$(dirname "$0")/common.sh"
WEDIT=1 launch fsdp_dmd.yaml wizard.distill.dmd_future --data $D/windows --student $1 --real $2 --out $3 --accum 4 --max_steps ${4:-500} --save_every 50 --cfg 3 \
  --levels 1000,500 --num_workers 4 \
  --group_rep e1o:8,e2o:8,e1n:8,e2n:8,e1x:8,turn:16,roam:4,sfly:8,sfly_replay:12,lookback:8,lookback0:2,ww:2,er:2,ue:2,bk:2,wedit:16 --rep_mem 16 \
  --rep_keys $D/roam_memory/rep_keys_both30.json --mem_index mem_index_both.pt --extra_rows $ROWS,$LOOKBACK_ROWS,$WEDIT_ROWS --p_trueunk 0 --p_unk2 0.5 $EXTRA
