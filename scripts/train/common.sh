# Shared settings for the training scripts. Override any of them in the environment.
export WAN_CKPT=${WAN_CKPT:-checkpoints/Wan2.2-TI2V-5B}
export WIZARD_DATA=${WIZARD_DATA:-data}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNK_NOCOORD=1 MEMW_COVER=1        # unknown (dark-green) cells carry no coordinates; memory loss weight only on uncovered cells
NGPU=${NGPU:-8}
D=$WIZARD_DATA
# --group_rep last: scripts 3 and 4 append ",lookback:4,lookback0:1" to it
MIX="--rep_mem 8 --rep_keys $D/roam_memory/rep_keys_both30.json --group_rep e1o:2,e2o:2,e1n:2,e2n:2,e1x:2,turn:2,roam:2,sfly:4,sfly_replay:6"
ROWS="$D/scenefly/rows.jsonl,$D/scenefly_replay/rows.jsonl"
launch() { accelerate launch --config_file configs/accelerate/$1 --num_processes $NGPU --main_process_port ${PORT:-29500} -m "${@:2}"; }
