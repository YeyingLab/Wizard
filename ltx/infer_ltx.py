#!/usr/bin/env python
"""Wizard on LTX-2.5: run the reference-binding IC-LoRA (weights/ltx25_lora) on one prepared case.

Inputs: the proxy video (--whitebox, IC reference), the first frame, up to two reference images (--ref1/--ref2), each owned by a
skeleton slot via --ref-slots ("2,0": ref1 belongs to slot 2, ref2 unused), and the case's slots.npz (token coverage of each
skeleton). Each reference is encoded at 512x512 as one latent frame and appended as clean tokens whose temporal RoPE position is
shifted by --ref-offset seconds, exactly as in training (FlexibleStrategy, `reference` condition with `time_offset`); the bbox gate
lets a reference be seen only by the video tokens of its slot. Sampling reuses the trainer's ValidationRunner (full-model multi-step
v2v), which the stock ltx-pipelines CLI does not expose. Normally called through ltx/run_ltx_case.sh.

NOTICE: written by the Wizard authors (2026) on top of LTX-2 (Lightricks); distributed under the LTX-2.x Community License.
"""
import argparse, copy, pathlib, sys, torch, yaml
from peft import LoraConfig as PeftLoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="training yaml (supplies model paths + lora spec)")
ap.add_argument("--lora", default=None, help="trained LoRA safetensors; omit to run the bare base model")
ap.add_argument("--whitebox", required=True, help="reference (white-box) video")
ap.add_argument("--first-frame", default=None, help="image or video whose frame 0 anchors appearance")
ap.add_argument("--prompt", required=True)
ap.add_argument("--negative-prompt", default=None)
ap.add_argument("--width", type=int, default=832)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--frames", type=int, default=81)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--stg", type=float, default=1.0)
ap.add_argument("--rescale", type=float, default=0.7)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out-dir", default="infer_out")
ap.add_argument("--ref1", required=True); ap.add_argument("--ref2", required=True); ap.add_argument("--ref-slots", default="2,3")
ap.add_argument("--slots", required=True, help="sample slots.npz (token_cover)"); ap.add_argument("--no-gate", action="store_true"); ap.add_argument("--no-embed", action="store_true")
ap.add_argument("--ref-offset", type=float, default=-2.0); ap.add_argument("--ref-size", type=int, default=512)
a = ap.parse_args()

cfg = LtxTrainerConfig(**yaml.safe_load(open(a.config)))

# ---- reference image: encode it with the runner's own image path, append it after the stock conditionings with a time offset
import ltx_trainer.validation_runner as VR
from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
REF = {}
_enc0 = VR.ValidationRunner._encode_sample_conditions
def _enc(self, sample, dims, vae_encoder, audio_encoder, device):
    media = _enc0(self, sample, dims, vae_encoder, audio_encoder, device)
    REF["latents"] = []
    for p in (a.ref1, a.ref2):
        img = self._load_first_frame(pathlib.Path(p))
        REF["latents"].append(self._encode_image(img, a.ref_size, a.ref_size, vae_encoder, device))
        print("[ref] encoded", p, tuple(REF["latents"][-1].shape), flush=True)
    return media
VR.ValidationRunner._encode_sample_conditions = _enc
class _RefImage(VideoConditionByReferenceLatent):
    def apply_to(self, latent_state, latent_tools):
        n0 = latent_state.positions.shape[2]; st = super().apply_to(latent_state, latent_tools)
        st.positions[:, 0, n0:, :] += a.ref_offset; return st
_app0 = VR.ValidationRunner._apply_video_conditionings
def _app(*args, **kw):
    st = _app0(*args, **kw)
    REF["n_before_refs"] = st.latent.shape[1]
    for lat in REF.get("latents", []):                                        # appended in order: [target][control][ref1][ref2]
        st = _RefImage(latent=lat.to(device=st.latent.device, dtype=torch.bfloat16), downscale_factor=1, temporal_scale_factor=1,
                       strength=1.0).apply_to(st, args[1] if len(args) > 1 else kw["tools"])
    return st
VR.ValidationRunner._apply_video_conditionings = staticmethod(_app)

# One synthetic validation sample carrying the caller's inputs.
sample = {
    "prompt": a.prompt,
    "conditions": [
        {"type": "reference", "video": a.whitebox,
         "downscale_factor": 1, "temporal_scale_factor": 1, "include_in_output": False},
    ],
}
if a.first_frame:
    sample["conditions"].append({"type": "first_frame", "image_or_video": a.first_frame})

val = copy.deepcopy(cfg.validation.model_dump())
val.update(
    samples=[sample],
    video_dims=[a.width, a.height, a.frames],
    frame_rate=a.fps,
    inference_steps=a.steps,
    video_cfg_scale=a.cfg,
    video_stg_scale=a.stg,
    guidance_rescale=a.rescale,
    seed=a.seed,
    interval=1,
    skip_initial_validation=False,
    generate_audio=False,
)
if a.negative_prompt:
    val["negative_prompt"] = a.negative_prompt
val.pop("images", None)
val_cfg = type(cfg.validation)(**val)

# Build the runner first: its init loads Gemma to encode prompts and then releases it.
# Moving the transformer to GPU before that puts two large models on the card at once.
runner = ValidationRunner(
    config=val_cfg,
    model_path=cfg.model.model_path,
    text_encoder_path=cfg.model.text_encoder_path,
    video_vae_path=cfg.model.video_vae_path,
    audio_vae_path=cfg.model.audio_vae_path,
    load_text_encoder_in_8bit=False,
)
torch.cuda.empty_cache()

print(f"[load] transformer {cfg.model.model_path}", flush=True)
transformer = load_transformer(checkpoint_path=cfg.model.model_path, device="cpu", dtype=torch.bfloat16)

SLOT_TABLE = None
if a.lora:
    print(f"[load] lora {a.lora}", flush=True)
    transformer = get_peft_model(
        transformer,
        PeftLoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            init_lora_weights=True,
        ),
    )
    stripped = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(a.lora).items()}
    SLOT_TABLE = stripped.pop("slot_embedding", None)                          # the binding table rides with the LoRA
    result = set_peft_model_state_dict(transformer.get_base_model(), stripped)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise SystemExit(
            f"[lora] {len(unexpected)} keys did not match the model, e.g. {unexpected[:2]}. "
            "Loading would have silently been a no-op."
        )
    print(f"[lora] loaded {len(stripped)} tensors from {pathlib.Path(a.lora).name}", flush=True)
else:
    print("[lora] none — running the bare base model", flush=True)

device = torch.device("cuda")
# ---- slot binding at inference: the trained slot table + per-token slot ids and the bbox gate, exactly as FlexibleStrategy._slot_binding
import dataclasses, types, numpy as np
from ltx_trainer.training_strategies.flexible import FlexibleStrategy, ReferenceConditionConfig
base = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
if a.lora and SLOT_TABLE is not None and not a.no_embed:
    base.enable_slot_embedding(SLOT_TABLE.shape[0] - 1)
    with torch.no_grad(): base.slot_embedding.data.copy_(SLOT_TABLE.to(base.slot_embedding.dtype))
    print(f"[slot] table {tuple(SLOT_TABLE.shape)} loaded, mean |w| {float(SLOT_TABLE.float().abs().mean()):.2e}", flush=True)
elif not a.no_embed:
    print("[slot] WARNING: no slot_embedding in the LoRA file -- running without the embedding", flush=True); a.no_embed = True
SL = np.load(a.slots); COVER = torch.from_numpy(SL["token_cover"].astype(np.float32))[None]
RS = torch.tensor([[int(x) for x in a.ref_slots.split(",")]], dtype=torch.long)
MC = types.SimpleNamespace(slots_dir="slots", slot_cover_thr=0.02, slot_gate_dilate=0.25)
_CTL = ReferenceConditionConfig(latents_dir="c", slot_role="control")
_R = [ReferenceConditionConfig(latents_dir="r1", slot_role="image", ref_index=0), ReferenceConditionConfig(latents_dir="r2", slot_role="image", ref_index=1)]
_BIND = {}
_fwd0 = base.forward
def _fwd(video=None, audio=None, perturbations=None, **kw):
    if video is not None:
        B, T = video.latent.shape[:2]; F_, H_, W_ = COVER.shape[1:4]; n = F_ * H_ * W_
        nb = REF.get("n_before_refs", 0); nr = [l.shape[-1] * l.shape[-2] * l.shape[-3] for l in REF.get("latents", [])]
        if T == nb + sum(nr) and nb >= 2 * n:
            if T not in _BIND:
                segs = [dict(kind="target", n=n, F=F_, H=H_, W=W_), dict(kind="ref", cfg=_CTL, n=n)]
                if nb > 2 * n: segs.append(dict(kind="other", n=nb - 2 * n))
                segs += [dict(kind="ref", cfg=_R[i], n=nr[i]) for i in range(len(nr))]
                ids, mask = FlexibleStrategy._slot_binding(None, MC, {"slots": {"token_cover": COVER, "ref_slots": RS}}, segs, video.latent.device)
                _BIND[T] = (ids, mask)
                print(f"[slot] layout T={T} (target {n}, control {n}, other {nb - 2 * n}, refs {nr}), ref_slots {RS.tolist()[0]}, "
                      f"gate open fraction for ref1 {float(mask[0, :n, nb:nb + nr[0]].mean()):.3f}", flush=True)
            ids, mask = _BIND[T]
            video = dataclasses.replace(video, slot_ids=None if a.no_embed else ids.expand(B, -1),
                                        attention_mask=None if a.no_gate else mask.expand(B, -1, -1))
        elif "warned" not in _BIND:
            _BIND["warned"] = True; print(f"[slot] WARNING: unexpected token count {T} (expected {nb + sum(nr)}); no binding on this call", flush=True)
    return _fwd0(video=video, audio=audio, perturbations=perturbations, **kw)
base.forward = _fwd
transformer = transformer.to(device).eval()

out_dir = pathlib.Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
with TrainingProgress(enabled=False, total_steps=1) as progress, torch.inference_mode():
    saved = runner.run(transformer, step=0, output_dir=out_dir, device=device, progress=progress)
for _, p in saved:
    print("[done]", p, flush=True)
