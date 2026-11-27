#!/bin/bash
# Run the Wizard LTX-2.5 LoRA on one prepared case.
#   bash ltx/run_ltx_case.sh <lora.safetensors> <case dir> <out.mp4>
# env: LTX2_DIR  = a clone of Lightricks/LTX-2 with ltx/ltx2_wizard.patch applied (see ltx/README.md)
#      LTX_CKPT  = local copy of Lightricks/LTX-2.5 (diffusion_models/, text_encoders/, vae/)
#      CUDA_VISIBLE_DEVICES = one GPU with >= 80 GB
set -e
LORA=$(realpath "$1"); C=$(realpath "$2"); OUT=$(realpath -m "$3"); H=$(cd "$(dirname "$0")" && pwd)
: "${LTX2_DIR:?set LTX2_DIR}"; : "${LTX_CKPT:?set LTX_CKPT}"; LTX2_DIR=$(realpath "$LTX2_DIR"); LTX_CKPT=$(realpath "$LTX_CKPT")
W=$(mktemp -d); mkdir -p $W/dummy_precomputed/{conditions,latents,ref1_latents,ref2_latents,reference_latents,slots}
sed -e "s#@LTX_CKPT@#$LTX_CKPT#g" -e "s#@WORK@#$W#g" $H/configs/ltx25_infer.yaml > $W/config.yaml
read FRAMES STEPS SEED SLOTS < <(python -c "import json;c=json.load(open('$C/case.json'));print(c['frames'],c['steps'],c['seed'],c['ref_slots'])")
REF2=$C/ref_b.png; [ -f $REF2 ] || REF2=$C/ref_a.png          # a second reference is only read when its slot is non-zero
export PYTHONPATH=$LTX2_DIR/packages/ltx-trainer/src:$LTX2_DIR/packages/ltx-core/src:$LTX2_DIR/packages/ltx-pipelines/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $LTX2_DIR/packages/ltx-trainer
python $H/infer_ltx.py --config $W/config.yaml --lora $LORA --no-embed --whitebox $C/proxy.mp4 --first-frame $C/first_frame.png \
  --prompt "$(cat $C/prompt.txt)" --width 1024 --height 576 --frames $FRAMES --fps 30 --steps $STEPS --cfg 3.0 --seed $SEED \
  --ref1 $C/ref_a.png --ref2 $REF2 --ref-slots $SLOTS --slots $C/slots.npz --out-dir $W/out
cp "$(ls $W/out/samples/*.mp4 | head -1)" "$OUT" && echo "wrote $OUT"
