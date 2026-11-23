# Data

## Training data

Training reads pre-encoded clips (121 frames at 30 fps, 832 × 480, Wan2.2 VAE latents of 31 × 30 × 52) under
`$WIZARD_DATA`:

```
windows/             rows.jsonl (one row per clip: group, key, paths), proxy/ (proxy latents, depth, tiers, cameras),
                     proxy_unk2/ proxy_trueunk/ proxy_memunk/ (the same with more cells unknown), shield/, geometry -> ../geometry
geometry/            per-cell depth and cameras of every window
samples/             clips/ (video latents; links into latents/, synth_*/, roam_*/), text_table.pt (caption embeddings), neg_emb.pt (empty-prompt embedding)
roam_memory/         roaming memory banks, mem_index*.pt, rep_keys_both30.json
scenefly/            revisit sequences (SceneFly), rows.jsonl
scenefly_replay/     revisit sequences replayed backwards, rows.jsonl
synth_lookback_geo/  synthetic look-back sequences (camera turns away and back), rows_full.jsonl
worldedit/           world-edit clips (objects / terrain changing in front of the camera), rows_train.jsonl, mem_index.pt, bank/ (read with WEDIT=1)
```

Each row belongs to a **group**, used for the mixing weights (`--group_rep group:weight,...`):

| Groups | Content |
|---|---|
| `e1o`, `e2o`, `e1n`, `e2n`, `e1x` | synthetic boss entries (one or two characters) |
| `turn` | turning while running |
| `roam` | roaming recordings |
| `sfly`, `sfly_replay`, `lookback`, `lookback0` | revisits: the camera comes back to a place it has seen |
| `ww`, `er`, `ue`, `bk` | duels |
| `wedit` | world edits (stages 1, 2, 3 and 4) |

The pre-encoded data is released in [`yuzhewu207/wizard-precoded`](https://huggingface.co/datasets/yuzhewu207/wizard-precoded) (the exact files the models were trained on, minus the WildWorld windows; see the WildWorld note in the README): group `ww` has no rows). Restore it with

```bash
python scripts/data/restore_precoded.py --root /path/to/wizard_data     # download + extract + rewrite the absolute paths
export WIZARD_DATA=/path/to/wizard_data
```

The data-building pipeline is not part of this repository.

## Visual proxy

One RGB frame per video frame (1024 × 576 before cropping):

| Channel | Meaning |
|---|---|
| R | world height |
| B | depth |
| G | tier: sky 0 · unknown ≈ 38 · coarse ≈ 77 · exact 153–255 (normal shading under a fixed light) |
| skeletons | drawn on top with a 2 px black outline; weapons and shields included |

Tiers come from the geometry's source: engine depth is *exact*, the first frame's MapAnything / MoGe-2 reconstruction is
*coarse*, everything outside the first view is *unknown*. Edits made by the world agent are *exact*.

## Inference case

A case directory holds the inputs of one shot:

| File | Content |
|---|---|
| `proxy.mkv` | the visual proxy (lossless FFV1) |
| `tier.npy` | tier per token cell, `[T, 60, 104]` uint8 |
| `gbuf.npz` | per-frame grid depth, camera position / rotation / intrinsics, scene origin and player height |
| `world_px.npz` | per-pixel depth and tier (world maintenance) |
| `slots.npz` | character slot of every pixel (reference binding) |
| `first_frame.png` | the first frame |
| `prompt.txt` | the text prompt |

`tools/case_to_exact.py <case> <new case>` relabels a case's coarse geometry as exact (same geometry and cameras).
