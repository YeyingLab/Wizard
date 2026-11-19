<div align="center">

# Wizard: Manipulate Any World in Real Time via Agentic Control

**Training and inference code for the Wizard video generator**

[![arXiv](https://img.shields.io/badge/arXiv-2610.40339-b31b1b.svg)](https://arxiv.org/abs/2610.40339)
[![Project Page](https://img.shields.io/badge/Project-Page-4b8bbe.svg)](https://wyzardwu.github.io/wizard/)
[![Weights & Data](https://img.shields.io/badge/%F0%9F%A4%97-Weights%20%26%20Data-ffcc4d.svg)](https://huggingface.co/datasets/yuzhewu207/wizard)
[![Pre-encoded data](https://img.shields.io/badge/%F0%9F%A4%97-Pre--encoded%20data-ffcc4d.svg)](https://huggingface.co/datasets/yuzhewu207/wizard-precoded)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)

<img src="assets/teaser.jpg" width="92%">

</div>

Wizard turns any scene image and any character references into a playable, real-time game world. An agent keeps the
game state (scene geometry, character skeletons, combat logic) and renders it into a **visual proxy**; a video model
built on [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2) turns the proxy into game frames, streaming at
**24 FPS with 2 denoising steps**.

This repository contains the **video generator**: the four-stage training pipeline (bidirectional teacher → causal
teacher → consistency distillation → DMD) and the three inference modes, including online world maintenance.

## Highlights

- **Proxy-driven control.** Scene geometry and articulated skeletons are rendered into one condition video and
  concatenated with the video latents on the channels: precise camera and motion control, zero-shot to new scenes,
  characters and motions.
- **Identity binding.** Each reference image is visible only inside its character's skeleton box.
- **Geometry you can trust.** The proxy marks every pixel as *exact*, *coarse* or *unknown*; the model follows exact
  geometry and invents unknown regions.
- **Spatial memory.** World-coordinate tokens and retrieved memory frames let the model redraw a place it has seen
  when the camera comes back.
- **Real time.** A causal 2-step student with a KV cache (20-frame window + attention sink), distilled with history
  self-augmentation and DMD; the world is maintained online on a second GPU without blocking generation.

## What is new compared with the paper

| | Paper (arXiv v1) | This release |
|---|---|---|
| Visual proxy | height / depth, G = 0 | + per-pixel tier (sky / unknown / coarse / exact) with normal shading, outlined skeletons, weapons, shields |
| World tokens | – | world coordinates + tier, added to every video and memory token |
| Memory | – | per-chunk retrieved memory frames, all characters hidden |
| Causal training | teacher forcing → CD → DMD | + history self-augmentation; memory and world tokens in every stage |
| Beyond the first view | invented terrain | marked *unknown*, then filled online from the generated frames |

<div align="center"><img src="assets/architecture.jpg" width="100%"></div>

## Installation

```bash
git clone https://github.com/wyzardWu/wizard27-clean.git && cd wizard27-clean
conda create -n wizard python=3.12 -y && conda activate wizard
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

`ffmpeg` must be on the `PATH`. Download the official [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
weights (VAE, T5 and tokenizer are used as-is) and point `WAN_CKPT` at the folder.

Online world maintenance additionally needs [MapAnything](https://github.com/facebookresearch/map-anything); it can live
in its own environment (see [docs/INFERENCE.md](docs/INFERENCE.md)).

## Quick start

```bash
export WAN_CKPT=/path/to/Wan2.2-TI2V-5B

# weights and three ready-made cases (first frame, proxy video, reference images, our outputs)
huggingface-cli download yuzhewu207/wizard --repo-type dataset \
    --include "weights/causal_dmd/*" "examples/*" --local-dir hf_wizard
C=hf_wizard/examples/fight_character_strikes_player

# real-time 2-step student
FRAMES=121 bash scripts/infer/run_case.sh causal hf_wizard/weights/causal_dmd/ema_step480.safetensors $C out.mp4 \
    --ref1 $C/ref_a.png --ref2 $C/ref_b.png --ref-slots 2,3

# with an online-maintained 3D world (needs world_px.npz in the case); FAST=1 runs the maintenance on the second GPU
CUDA_VISIBLE_DEVICES=0,1 FAST=1 bash scripts/infer/run_case.sh world <dit> <case> out.mp4
```

Each run writes `out.mp4` (generation | proxy) and `out_gen.mp4` (generation only). The case format is described in
[docs/DATA.md](docs/DATA.md).

## Training

| Stage | Module | Script |
|---|---|---|
| 1. Bidirectional teacher | `wizard.train.bidirectional` | `scripts/train/1_bidirectional.sh` |
| 2. Causal teacher forcing | `wizard.train.causal` | `scripts/train/2_causal.sh` |
| 2b. History self-augmentation | `wizard.train.causal_selfaug` | `scripts/train/3_causal_selfaug.sh` |
| 3. Consistency distillation | `wizard.distill.consistency_selfaug` | `scripts/train/4_consistency.sh` |
| 4. Distribution matching distillation | `wizard.distill.dmd` | `scripts/train/5_dmd.sh` |

Commands, hyper-parameters and the data layout: [docs/TRAINING.md](docs/TRAINING.md) and [docs/DATA.md](docs/DATA.md).

## Checkpoints

All weights are DiT checkpoints fine-tuned from Wan2.2-TI2V-5B (the VAE and T5 are the official ones), in the `weights/` folder of [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard):

| File | Model | Sampling |
|---|---|---|
| `causal_dmd/ema_step480.safetensors` | real-time 2-step student (stage 4, DMD) | `causal` / `world`, 2 steps |
| `causal_cd/ema_step10000.safetensors` | consistency-distilled student (stage 3), the starting point of DMD | `causal` / `world`, 2 steps |
| `causal_teacher/dit_step11000.safetensors` | causal teacher (stage 2) | teacher of stage 3 |
| `bidirectional_main/dit_step35000.safetensors` | bidirectional teacher (stage 1), the model of the paper's main table | `bidirectional`, 30 steps, CFG 3 |
| `bidirectional/dit_step35000.safetensors` | bidirectional teacher trained with the world-edit clips as well | `bidirectional`, 30 steps, CFG 3 |

```bash
huggingface-cli download yuzhewu207/wizard --repo-type dataset --include "weights/causal_dmd/*" --local-dir hf_wizard
```

**Other base models.** The approach is not tied to Wan: the visual proxy, the reference binding and the training recipe carry over to other video backbones. We also trained it on [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (22B) as a LoRA; the weights are in [`weights/ltx25_lora/`](https://huggingface.co/datasets/yuzhewu207/wizard/tree/main/weights/ltx25_lora) (an earlier proxy format without tiers or memory; its inference code, a modified LTX-2 pipeline with the reference-image conditioning, is not part of this repository).

## Data

- **Pre-encoded training data** (the exact files the models were trained on): [`yuzhewu207/wizard-precoded`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded), restored with `scripts/data/restore_precoded.py`; see [docs/DATA.md](docs/DATA.md).
- **Raw training data** (videos, proxy videos, geometry, memory banks, reference images): [`yuzhewu207/wizard`](https://huggingface.co/datasets/yuzhewu207/wizard).
- **WildWorld**: about half of the training windows come from [WildWorld](https://huggingface.co/datasets/AlayaLab/WildWorld), which cannot be redistributed; [`ww_index`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded/tree/main/ww_index) lists the windows and the scripts to rebuild them from the upstream dataset.

## Repository structure

```
wizard/
├── models/       DiT, flow-matching scheduler, proxy and reference injection, world tokens and memory
├── data/         proxy encoding, G-buffer, video I/O
├── memory/       memory retrieval (shared by training and inference)
├── train/        stages 1, 2 and 2b
├── distill/      stages 3 and 4
├── inference/    bidirectional, causal and causal + world maintenance; KV-cache streams
├── world/        online world maintenance (point cache, meshes, reconstruction workers)
└── vendor/wan/   Wan2.2 VAE and T5 encoder
configs/          accelerate / FSDP configs
scripts/          training, inference and data-restore (scripts/data/restore_precoded.py) entry points
tools/            case utilities
docs/             data, training and inference guides
```

## Reproducibility

This code was reorganised from our research code and checked against it on the same inputs and seeds: all three
inference modes produce bit-identical videos, and every training stage produces the same loss on its first step (later
steps differ only by the run-to-run nondeterminism of the GPU backward pass, which the original code shows as well).

## Citation

```bibtex
@article{wu2026wizard,
  title   = {Wizard: Manipulate Any World in Real Time via Agentic Control},
  author  = {Wu, Yuzhe and Lin, Zijun and Jin, Yeying},
  journal = {arXiv preprint arXiv:2610.40339},
  year    = {2026}
}
```

## Acknowledgements

We build on [Wan2.2](https://github.com/Wan-Video/Wan2.2) (the VAE and T5 encoder in `wizard/vendor/wan` are taken
from it) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (flow-matching scheduler). The causal adaptation and distillation follow
Causal Forcing++, [CausVid](https://github.com/tianweiy/CausVid), [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
and [DMD2](https://github.com/tianweiy/DMD2); world maintenance follows [GEN3C](https://github.com/nv-tlabs/GEN3C) and
uses [MapAnything](https://github.com/facebookresearch/map-anything).

## Licence

Non-commercial research use only; see [LICENSE.md](LICENSE.md).
