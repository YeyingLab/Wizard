# Training

All stages run on 8 GPUs with FSDP (`NGPU` changes the count). Set the data and checkpoint locations first:

```bash
export WAN_CKPT=/path/to/Wan2.2-TI2V-5B WIZARD_DATA=/path/to/data
```

## Pipeline

The scripts reproduce the run that produced the released `bidirectional`, `causal_teacher` checkpoints and the current distillation chain:

```bash
bash scripts/train/1_bidirectional.sh runs/bidir                                  # 0 -> 3k, then + world-edit clips 3k -> 35k
bash scripts/train/2_causal.sh runs/bidir/dits/dit_step35000.safetensors runs/causal          # 0 -> 11k
bash scripts/train/3_consistency.sh runs/causal/dits/dit_step11000.safetensors runs/cd      # 0 -> 10k
bash scripts/train/4_dmd.sh runs/cd/ema/ema_step10000.safetensors runs/bidir/dits/dit_step35000.safetensors runs/dmd   # 500
```

Our runs were interrupted and resumed (the scripts' comments say how): stage 2 was resumed from its own checkpoints (seed = resume step); stage 3 ran 0 -> 2.7k
and was then warm-started from its 2k EMA (`STUDENT_INIT`, `STEP0=2000`, `SEED=1`) to 10k.

## Stages

| Stage | Steps | Batch | LR | Notes |
|---|---|---|---|---|
| 1. Bidirectional | 35k | 8 | 1e-5 (condition layers 1e-4) | unknown cells without coordinates, `p_unk2 0.5`, memory loss weight 3; world-edit clips (`WEDIT=1`) |
| 2. Causal teacher | 11k | 8 | 1e-5 (condition layers 1e-5) | history window 20 + sink 1; history self-augmentation `SA_P 0.5 / SA_MAX 0.25` on revisit clips; memory from before and after the window |
| 3. Consistency distillation | 10k | 8 | 2e-6, EMA 0.99 | N = 48 levels, 2 sampling steps (1000, 500), teacher CFG 3, `SA_P 0.7 / SA_MAX 0.5`; memory before and after the window |
| 4. DMD | 500 | 32 (8 × accum 4) | 2e-6 / critic 4e-7 | critic : student = 5 : 1, real score = the stage-1 model with CFG 3; memory from the past only, no world-edit clips |

Hyper-parameters not set in the scripts keep the defaults of each module's argument parser.

**History self-augmentation.** Under teacher forcing the clean history already contains almost everything, so the model
never learns that memory should override a drifting history. With probability `SA_P` the history is replaced by a
one-step estimate from a noised copy (`σ ~ U(0, SA_MAX)`); frame 0, the memory frames and the target stay clean. In
stage 3 the teacher keeps the clean history and only the student / EMA read the augmented one.

**Resuming DMD.** Every save also writes `student_last.safetensors` and `fake_last.safetensors`:
`EXTRA="--fake runs/dmd/fake_last.safetensors --step0 <N>" bash scripts/train/4_dmd.sh runs/dmd/ema/ema_step<N>.safetensors ...`

## Training-time switches (environment)

| Variable | Used by | Meaning |
|---|---|---|
| `UNK_NOCOORD=1` | all | unknown cells carry no world coordinates |
| `MEMW_COVER=1` | all | cells the window's first frame has not seen but a retrieved memory frame has get loss weight `--mem_loss_w`; "seen" by projected coverage |
| `WEDIT=1` | 1, 2, 3 | add the world-edit clips' memory index and banks (`worldedit/`); their rows come with `--extra_rows worldedit/rows_train.jsonl` |
| `MEMW_HISTW`, `MEMW_SINK` | 2 | "seen" also by every clean history frame the noisy frame reads (window + sink), so the weight sits where only memory can tell |
| `MEM_FUTURE=1` | 2 | SceneFly windows may also retrieve frames after the window (stage 3 does the same through `wizard.distill.consistency_selfaug_future`) |
| `SA_P`, `SA_MAX`, `SA_GROUPS` | 2, 3 | history self-augmentation |
