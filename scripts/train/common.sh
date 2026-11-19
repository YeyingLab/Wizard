# Shared settings for the training scripts. Override any of them in the environment.
export WAN_CKPT=${WAN_CKPT:-checkpoints/Wan2.2-TI2V-5B}
export WIZARD_DATA=${WIZARD_DATA:-data}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNK_NOCOORD=1 MEMW_COVER=1        # every stage: unknown (dark-green) cells carry no coordinates; "seen" for the memory loss weight by projected coverage
NGPU=${NGPU:-8}
D=$WIZARD_DATA
# every stage has an effective batch of 8 windows per optimizer step (DMD: 32): GRAD_ACCUM defaults to 8 / NGPU
GRAD_ACCUM=${GRAD_ACCUM:-$((8 / NGPU))}
# --group_rep last: stages 2 and 3 append ",lookback:4,lookback0:1" to it
MIX="--rep_mem 8 --rep_keys $D/roam_memory/rep_keys_both30.json --group_rep e1o:2,e2o:2,e1n:2,e2n:2,e1x:2,turn:2,roam:2,sfly:4,sfly_replay:6"
ROWS="$D/scenefly/rows.jsonl,$D/scenefly_replay/rows.jsonl"
LOOKBACK_ROWS="$D/synth_lookback_geo/rows_full.jsonl"
WEDIT_ROWS="$D/worldedit/rows_train.jsonl"      # read with WEDIT=1 (adds worldedit/mem_index.pt and its memory banks)
launch() { accelerate launch --config_file configs/accelerate/$1 --num_processes $NGPU --main_process_port ${PORT:-29500} -m "${@:2}"; }
