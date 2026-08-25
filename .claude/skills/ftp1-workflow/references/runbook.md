# FTP-1 Operational Runbook

## Training environment

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --dev
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
TRANSFORMERS_DIR=$(
  .venv/bin/python -c \
    'import pathlib, transformers; print(pathlib.Path(transformers.__file__).resolve().parent)'
)
cp -r src/openpi/models_pytorch/transformers_replace/. "$TRANSFORMERS_DIR/"
uv run python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

Verify each selected GPU with a real CUDA tensor operation, not device enumeration alone. On Ampere A100, keep the locked CUDA wheel and do not run the Blackwell override below.

For an NVIDIA Blackwell GPU (`sm_120`), the CUDA 12.6 wheel in the lock file cannot execute kernels. Apply the supported CUDA 12.8 wheel override after the normal sync, then prevent later launcher invocations from reconciling it away:

```bash
bash scripts/install_pytorch_cuda128.sh
export FTP1_UV_NO_SYNC=true
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability())'
```

Keep checkpoints, assets, and the OpenPI cache on a filesystem with adequate free space. Portable launchers default to `.cache/ftp1`, but accept `FTP1_CACHE_ROOT`, `FTP1_CHECKPOINT_BASE_DIR`, `FTP1_ASSETS_BASE_DIR`, and `OPENPI_DATA_HOME`.

## UniVTAC fine-tuning

Prepare a JSON config whose enabled domain is `UniVTAC_lift_bottle` and whose path contains immediate `*.zarr` stores. Then run:

```bash
export FTP1_DATASET_CONFIG=/absolute/path/dataset_univtac_lift_bottle.json
export FTP1_PRETRAINED_CHECKPOINT=/absolute/path/ftp1_pretrain_v0426_50kstep
export FTP1_CACHE_ROOT=/absolute/path/ftp1_cache
export FTP1_REPO_ID=univtac_lift_bottle  # Use exactly the same value for norm stats and training.
export CUDA_VISIBLE_DEVICES=0
export FTP1_UV_NO_SYNC=true  # Required when using the CUDA 12.8 override.

uv run python scripts/ftp1_preflight.py \
  --dataset-config "$FTP1_DATASET_CONFIG" \
  --checkpoint "$FTP1_PRETRAINED_CHECKPOINT"
bash scripts_exp_zarr/univtac/compute_norm_stats_univtac_example.sh

# Smoke test; checkpoints remain compatible with the full run.
FTP1_NUM_TRAIN_STEPS=100 FTP1_SAVE_INTERVAL=100 FTP1_VAL_INTERVAL=50 \
CUDA_VISIBLE_DEVICES=0 bash scripts_exp_zarr/univtac/train_univtac_example.sh

# Full verified one-GPU run on the Precision 7960 reference host.
FTP1_LOCAL_BATCH_SIZE=2 bash scripts_exp_zarr/univtac/train_univtac_example.sh
```

The reference launcher uses a 90/10 episode split, Z-score normalization, channel-wise image-tactile statistics, BF16, a Gemma-small tactile expert, and 20,000 steps. It uses `action_joint_rep=mix`; normalization must be computed with that same representation. It also skips FoundationTactile's T3-large initialization by default because the released FTP-1 checkpoint already includes the matching 768-wide HPT tokenizer, while T3-large is shape-incompatible. Set `FTP1_ENABLE_TRACKING=true` for W&B-compatible logging. Use the SwanLab wrapper only after exporting `SWANLAB_API_KEY`.

The Precision 7960 host has a verified local batch size of 2 on one GPU (one optimizer step in about 27 seconds). Three-rank NCCL `all_reduce` succeeds, but DDP's initial parameter synchronization for the 4.3B model fails with an illegal memory access even with peer-to-peer transport disabled. Do not schedule a long DDP job on this stack until a newer PyTorch/NCCL combination passes a full-model one-step gate.

The parser accepts both historical `UniVTAC/<task>/demo/hdf5/*.hdf5` inputs and the published `UniVTAC/<task>/clean/*.hdf5` layout:

```bash
PATH="$PWD/.venv/bin:$PATH" \
BASE_DIR=/absolute/path/UniVTAC SAVE_DIR=/absolute/path/processed/UniVTAC \
EPISODES_PER_TASK=100 TASK_LIST=lift_bottle \
IMAGE_SIZE=224 \
bash data_processing/parse_data_scripts/parse_data_univtac.sh
```

The wrapper invokes bare `python`, so put `.venv/bin` first on `PATH`. It can also exit successfully when no raw files are discovered; require `lift_bottle_head.zarr`, preflight success, and the expected counts. The published 100-episode subset at the time of the A100 qualification produced exactly 30,432 steps and a 7.6 GiB Zarr store.

Download only the required release artifacts after logging in through the standard user credential stores:

```bash
hf download MJJJJ1064/ftp1_v0426_50kstep --local-dir "$FTP1_PRETRAINED_CHECKPOINT"
hf download byml/UniVTAC --repo-type dataset \
  --include 'lift_bottle/clean/*.hdf5' --local-dir "$RAW_UNIVTAC"
```

Do not place tokens in command arguments, scripts, logs, or repository files. Expect 100 HDF5 files unless upstream changes; independently compute the converted step count as the sum of `embodiment/joint.shape[0] - 1`.

For a two-GPU A100 run, preserve one namespace across normalization and training:

```bash
export FTP1_CACHE_ROOT=/absolute/path/runtime
export FTP1_DATASET_CONFIG=/absolute/path/dataset_univtac_lift_bottle.json
export FTP1_PRETRAINED_CHECKPOINT=/absolute/path/ftp1_pretrain_v0426_50kstep
export FTP1_REPO_ID=univtac_lift_bottle_a100

CUDA_VISIBLE_DEVICES=0 bash scripts_exp_zarr/univtac/compute_norm_stats_univtac_example.sh
```

Require `norm_params_snapshot.json`, `share_norm_stats_all_t0_zscore.json`, and `tactile_input_config.json`. Verify relative pose inputs/actions, absolute proprioceptive joints, `mix` action joints, channel-wise tactile statistics, and the seed-42 episode-level 90/10 split. Run preflight with both the absolute dataset config and pretrained checkpoint before normalization and every training launch.

Qualify DDP with tracking and validation disabled: first set `FTP1_NUM_TRAIN_STEPS=1`, `FTP1_SAVE_INTERVAL=1`, and local batch 1; then use 100 steps, log/save interval 10/100. Keep compile disabled. Require both ranks, clean DDP synchronization, finite logged loss and pre-clipping gradient norm, no NCCL/CUDA/OOM error, symmetric GPU memory, and final optimizer counters of 1 and 100. Gate checkpoints are zero-based directories `0/` and `99/`.

Launch production fresh from `FTP1_PRETRAINED_CHECKPOINT`, never a gate checkpoint. The qualified recipe uses two ranks, local/global batch 1/2, BF16, 20,000 steps, 500-step warmup, peak/end LR `5e-5`/`5e-6`, validation and checkpoint intervals of 2,000, compile off, T3 loading off, and W&B on. Set `USE_SWANLAB=false`, keep credentials out of the tmux command, and require a real HTTPS W&B run URL. Before handoff, wait for both ranks and HPT weights, snapshot match, finite step-0 validation, at least ten finite optimizer steps, active GPUs, and a live tmux session.

The full 3,026-frame validation split takes about 6m50s on the qualified A100 stack and runs at step labels 0, 2000, ..., 18000, plus final label 19999. Do not derive ETA from the first logging window because step-0 validation is included in it; use later training windows plus validation/checkpoint overhead. The PyTorch saver does not prune by `keep_period`.

## Offline and UniVTAC evaluation

Run held-out action evaluation from the training environment:

```bash
uv run python scripts/zarr_eval_ftp1_pytorch.py \
  --checkpoint_dir /absolute/path/to/checkpoint/19999 \
  --device cuda
```

The evaluator writes `evaluation_aggregated_metrics.json` inside the selected step directory. A metric value of `-1` denotes an action group that is inactive for that embodiment (for example, left-arm or Cartesian-pose fields in the UniVTAC right-arm joint-control task); use `rmse_total` and the active right-arm/right-hand fields for comparisons. The entrypoint must remain non-interactive—`scripts/test_zarr_eval_ftp1.py` guards against accidental debugger traps.

Install UniVTAC in its Python 3.10 Isaac Sim environment by following `UniVTAC/Installation_FTP1.md`, then install the inference subset without replacing Isaac's PyTorch:

```bash
conda activate uni
bash UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh
```

Run a ten-episode gate, inspect `infer_input_sample`, then run 100 episodes:

```bash
cd UniVTAC
CHECKPOINT_DIR=/absolute/path/to/checkpoint/19999 \
DOMAIN_NAME=UniVTAC_lift_bottle TOTAL_NUM=10 GPU=0 \
bash scripts/shell/eval_ftp1.sh lift_bottle

CHECKPOINT_DIR=/absolute/path/to/checkpoint/19999 \
DOMAIN_NAME=UniVTAC_lift_bottle TOTAL_NUM=100 GPU=0 \
bash scripts/shell/eval_ftp1.sh lift_bottle
```

Use fixed seeds, `action_rep=auto`, ten diffusion steps, temporal ensembling, and the same chunk prefix for checkpoint comparisons. Results live under `UniVTAC/eval_results/FTP1` with videos and `metadata.json`.

## Released pretraining corpus

The release is roughly 7.5 TB compressed. Provision at least 16 TB before attempting the full corpus. Download individual archives with the ModelScope or Hugging Face CLI rather than cloning everything. Validate and extract one archive without materializing a combined tar:

```bash
bash scripts/stage_ftp1_archive.sh --check-only /archive/download/MotionTrans /data/ftp1/MotionTrans
bash scripts/stage_ftp1_archive.sh /archive/download/MotionTrans /data/ftp1/MotionTrans
```

Point `scripts_exp_zarr/pretrain_small/data_config_pretrain_small.json` at the extracted Zarr domain, then run its normalization and training launchers. Qualify a subset-pretrained checkpoint by fine-tuning and evaluating it on the same UniVTAC split; do not use pretraining loss alone as the transfer result.

## MoT-JEPA pretraining on the H100 SLURM cluster

Self-supervised video-tactile pretraining lives in `src/openpi/mot_jepa/` and
`scripts/mot_jepa_*.py`, entirely separate from the FTP-1 flow-matching path so the baseline
arm stays bit-identical. Launchers are in `scripts_exp_zarr/mot_jepa/` — read its README
before submitting anything long.

### Cluster facts (verified 2026-08-02)

491 H100 nodes, each 8 GPU / **128 CPU** / **2 TB RAM**, with node-local NVMe at `/raid`
(7 TB). Partitions: `batch` 4 h, `batch_long` 8 h, `backfill` 7 d, `batch_large_long` 14 d.
pyxis/enroot available with `nvcr.io` credentials already in `~/.config/enroot/.credentials`.
Lustre quota 250 T.

Request `--cpus-per-task=128` and `--mem=0`, and `export SRUN_CPUS_PER_TASK` before `srun`:
SLURM >= 22.05 stopped propagating `--cpus-per-task` to `srun`, and with `ntasks-per-node=1`
plus torchrun forking 8 ranks, an under-provisioned cgroup pins 8 ranks and ~100 dataloader
workers onto a fraction of a core. It presents as "GPUs at 15% util", not as an error.

### Preemption and requeue

**Verified on this cluster** (`scripts_exp_zarr/mot_jepa/signal_probe.sbatch`, job 6442100):
`--signal=B:USR1@N` reaches the batch shell, the trap fires, `wait` returns 128+signum, the
re-wait loop recovers, the worker sees the flag file and exits, and the script regains
control at rc=0 before walltime.

`--signal=B:USR1@N` delivers **only to the batch shell**, so `srun` must be backgrounded and
`wait`ed or the trap fires only after the walltime kill. The trap touches a flag file, and
that file is the primary mechanism — `scancel --signal` reaches the job step, not torchrun's
worker children, and torchrun has historically not forwarded SIGUSR1. Rank 0 polls it and
broadcasts, so ranks cannot disagree and hang a collective. Requeue keys on an armed flag,
never on the exit code, so a genuine crash cannot loop forever.

### Run continuity across a ~40-job chain

`run_config.json` is frozen with `O_CREAT|O_EXCL` on first launch and re-read afterwards, so
editing a launcher mid-chain cannot silently change hyperparameters; drift is logged and the
frozen value wins. W&B continuity needs `wandb_id.txt` + `resume="must"` **and**
`USE_SWANLAB=false` — the shim defaults to SwanLab and pops `id`/`resume`
(`shared/wandb_compat.py:22,137-145`), which would fork a new run every job. The trainer
asserts on the backend rather than trusting the env.

### Derived clip store

`scripts/mot_jepa_build_clip_zarr.py` rewrites a domain at model resolution with chunks
aligned to exactly one clip. Measured on `RDP_Bimanual`: **4.1x faster reads** (1520 ->
367 ms/clip) and **4.8x smaller** (14 G -> 2.9 G). The win is pre-resizing gel from its
stored 224 to the model's 112, plus chunk alignment — not the compressor, which is already
lz4 in the released stores.

### Known-good commands

```bash
uv run python scripts/mot_jepa_survey_corpus.py --data-root <staged> --assumed-fps 30
uv run python scripts/mot_jepa_build_clip_zarr.py --source <staged> --output <clips>
EXP_NAME=pilot01 NODES=4 STAGE_SOURCE=<clips>/RDP bash scripts_exp_zarr/mot_jepa/submit.sh
uv run python scripts/mot_jepa_analyze.py --checkpoint <ckpt>/<step> --data-glob '<clips>/*/*.zarr' --output reports/<name>
```

### MoT-JEPA transfer to UniVTAC

A MoT-JEPA pretraining snapshot contains only a self-supervised encoder (`teacher_ema.pt`), not
an action policy. Keep these experiments distinct:

- **zero-shot representation:** label-free rank, dispersion, retrieval, and time-shuffle probes;
- **frozen linear probe:** fit ridge weights on UniVTAC training episodes and score held-out
  episodes; this is supervised transfer, not literal zero-shot control;
- **head adaptation:** `scripts/mot_jepa_policy_train.py` trains an action head while the encoder
  remains frozen under `no_grad`;
- **backbone fine-tuning:** `scripts/mot_jepa_policy_train.py` supports `frozen`, `last_blocks`,
  and `full`; the head and backbone share one DDP reducer, every new checkpoint carries
  `backbone.pt`, and evaluation/deployment strictly load it for adapted modes.

Backbone adaptation must start from the canonical completed head
`f100k_univtac_o2a1_h16_flowmatch_s42/checkpoints/20000/student.pt` (SHA-256
`c50f50ffb4eb5299e69c221ecee1ac60e168e78f77e9feaf61a1697d7a191d9e`) and the exact 100k
forecast snapshot, never a reconstructed live preset. Fresh-run validation requires the source
`DONE` and `latest` markers to equal 20,000 and checks the source/target head, horizon, layout,
encoder, predictor, data cadence, holdout, seed, and backbone contracts. It stamps the source
head and source-config digests in every checkpoint. Resume refuses a change in any of those
fields, the trainable-parameter manifest, world size, or source topology before loading weights.

The three trainability arms are:

- `frozen`: source backbone stays bit-identical and supplies the matched extra-update control;
- `last_blocks`: adapt the final `N` encoder blocks plus final norms;
- `full`: adapt every reachable encoder parameter. The two sensor-ID embeddings remain frozen
  because the current `to_inputs` path supplies no sensor IDs, so they are outside the graph.

Run `scripts_exp_zarr/mot_jepa/policy_gate.sbatch` on four H100s before production. First use a
fresh one-step run for every arm; then use a fresh 100-step target with `GATE_PREEMPT_AT=50`, and
resubmit the same run with `ALLOW_RESUME=1`. The gate requires exact optimizer coverage/counters,
finite positive head gradients, finite positive adapted-backbone gradients (exact zero for
`frozen`), selected-only weight changes, strict deployment reload, identical action statistics,
and safe per-rank peak GPU memory. Always pass `--head.horizon 16`: the live preset default is 32,
and the source-contract gate deliberately rejects that mismatch before the first optimizer step.

Verified one-step gates on 2026-08-20 use the same source and four-H100/global-128 topology:

| mode | job | changed backbone tensors | peak allocated GiB/rank | result |
|---|---:|---:|---:|---|
| frozen | 6687944 | 0 (bit-identical) | 1.697 / 1.697 / 1.697 / 1.697 | PASS |
| last two blocks + norms | 6687945 | 60 selected only | 1.712 / 1.712 / 1.712 / 1.712 | PASS |
| full reachable backbone | 6687946 | 348 selected only | 1.916 / 1.916 / 1.916 / 1.916 | PASS |

Each has optimizer counters exactly 1, all 126 head tensors changed, finite gradients, and a strict
backbone reload. Two earlier gate failures are useful safeguards, not qualified runs: jobs
6687760--6687762 were rejected because the unpinned live preset supplied horizon 32, and frozen
job 6687852 exposed an empty-gradient-group CPU-zero/CUDA stacking bug. The fix makes an empty
frozen group report its zero on the training device and is covered by a regression test.

The controlled-preemption qualification also passed for all three modes. These runs used a fresh
100-step target, requested preemption only after a saved checkpoint, and then resumed the same run
exactly once:

| mode | preempt job / step | resume job | changed backbone tensors | final peak GiB/rank | result |
|---|---:|---:|---:|---:|---|
| frozen | 6687986 / 60 | 6688086 | 0 (bit-identical) | 1.584 | PASS |
| last two blocks + norms | 6687987 / 60 | 6688087 | 60 selected only | 1.830 | PASS |
| full reachable backbone | 6687988 / 50 | 6688085 | 348 selected only | 2.352 | PASS |

Every resumed run reached `DONE=latest=100`, reported optimizer counters exactly `{100}`, preserved
the source action statistics, reloaded through the deployment path, and retained finite gradients
on all four ranks. Production jobs 6688213/6688214/6688215 then started fresh from the canonical
20k head for another 20,000 updates in frozen/last-two/full mode respectively; their step 20,000
means **20k additional updates**, not a continuation numbered 40,000. The generic requeue script
contains `#SBATCH --exclusive`; command-line `--oversubscribe` does not override that directive on
this cluster. For a shared four-GPU allocation, submit an outer `sbatch --wrap='bash
mot_jepa_pretrain.sbatch'` with `--gpus-per-node=4` and also export `GPUS_PER_NODE=4`.

All three production jobs completed `0:0` without requeue and wrote `DONE=latest=20000`. Strict
deployment reload and final source comparison passed: frozen changed `0/350` backbone tensors;
last-two changed exactly its 60 selected tensors and no others; full changed all 348 reachable
tensors, leaving only the two declared unreachable sensor embeddings unchanged. Optimizer
counters are exactly 20,000, action-statistics and source-backbone digests match, and the final
frozen/last-two/full losses were `0.08550/0.08328/0.07574`.

The corresponding exact 2,839-clip held-out evaluations completed `0:0`:

| mode | pooled K=1 RMSE | pooled K=16 RMSE | `lift_bottle` K=1 | `lift_bottle` K=16 |
|---|---:|---:|---:|---:|
| frozen extra-20k control | 0.00321473 | 0.00304237 | 0.00092638 | 0.00087980 |
| last two blocks + norms | **0.00305197** | 0.00293142 | 0.00091661 | 0.00087514 |
| full reachable backbone | 0.00311926 | **0.00292840** | **0.00085741** | **0.00081964** |

Last-two is best pooled at K=1; full is narrowly best pooled at K=16 and clearly best on the
target `lift_bottle` domain offline. Closed-loop success, not this open-loop RMSE, is the decision
metric.

The observation and action clocks must not be coupled. The 100k forecast backbone trained on
observation strides `(2, 4)`; UniVTAC deployment uses observation stride `2` but executes and
re-plans every control step, so its action stride is `1`. Every supervised dataset, ridge artifact,
action-statistics artifact, offline evaluator, and deployment wrapper must therefore agree on
`observation/action = 2/1`. Legacy ridge files without `observation_stride`, `action_stride`, and
`horizon` metadata must be refitted rather than relabeled.

At `num_frames=16`, observation/action `2/1`, and index step 4, horizon 32 removes every
`grasp_classify` clip because those episodes are too short. Use horizon 16 for the all-eight-task
arm: the verified episode-mod-10 split contains 26,269 train and 2,839 holdout clips, with
`grasp_classify` contributing 187/20. On eight ranks use local batch 16 (global 128); a global
batch larger than the smallest domain pool makes the domain-pure sampler drop that task. The same
qualified global batch is four ranks at local batch 32, which is the production layout used below.

Run the corrected frozen probe through the GPU-only launcher:

```bash
REPO_ROOT=$PWD \
SNAP=/absolute/path/to/forecast100k_snapshot STEP=100000 \
CLIPS='/absolute/path/to/univtac-clips/*/*.zarr' \
OUT=/absolute/path/to/ridge_obs2_act1_h16.npz \
HORIZON=16 OBSERVATION_STRIDE=2 ACTION_STRIDE=1 INDEX_STEP=4 \
sbatch scripts_exp_zarr/mot_jepa/ridge_policy.sbatch
```

Verified on 2026-08-20 with the 100k forecast/QK-norm snapshot (job 6681966): the corrected
all-eight-task ridge scored pooled held-out RMSE `0.0026487` against a deployable training-mean
constant of `0.00450194`, winning on every task. The exact constant group RMSEs are `0.00475885`
for the right arm and `0.00190077` for the right hand; these use the per-domain training-chunk mean,
never the holdout mean. Mean feature weight share was 35%; `lift_bottle` was
43%, below the preregistered 50% closed-loop gate. Canonical evaluator job 6682069 reproduced the
same pooled RMSE exactly. Pooled arm position error after 16 integrated actions was `0.04314` rad,
close to the fully correlated bound (`0.04489`) rather than the independent-error bound
(`0.01122`), so per-step replanning remains mandatory. The stamped artifact is
`ftp1-runs/eval/ridge_forecast100k_qknorm_s100000_obs2_act1_h16_v2.npz`.

The matched forecast/QK-norm 50k snapshot, rerun with the identical corrected protocol (job
6682161; canonical job 6682288), scored `0.0025282` pooled RMSE with 44% `lift_bottle` and
37% mean feature weight share, versus 100k's `0.0026487`, 43%, and 35%. The 100k row is 4.8%
worse in pooled RMSE. Thus the extra 50k pretraining steps did not improve frozen UniVTAC control
transfer; do not compare either row with the older stride-1/horizon-32 artifacts.

For a fresh frozen-backbone flow-matching head, first run a 100-step gate and then launch
production fresh rather than resuming the gate checkpoint. Use `--data.strides 2`,
`--data.action-stride 1`, `--head.horizon 16`, `--holdout-mod 10`, index step 4, and global batch
128. The 100-step four-H100 gate (job 6682043) completed cleanly; production job 6682146 uses four
H100s, local batch 32, 20,000 steps, warmup 500, peak/end LR `2e-5`/`1e-6`, save interval 1,000,
and seed 42. Policy runs freeze `run_config.json`; `action_stats.npz` is stamped with the same
cadence, horizon, and ordered domain mapping.

Post-adaptation evaluation must load `run_config.json` rather than live preset defaults and must
require `RUN/DONE == selected head step == num_train_steps`; the trainer intentionally exits zero
when preempted, so an `afterok` dependency alone does not prove that the run is final. Reject a
CLI holdout modulus, index step, backbone path/step, action-statistics cadence, horizon, or domain
mapping that differs from the frozen run. For the all-eight-task comparison require exactly 2,839
held-out clips. Use seed 0 and ten Euler steps. Report both one sampled chunk (`K=1`, the actual
stochastic deployment path) and a 16-sample average (`K=16`, a conditional-mean diagnostic that is
comparable in kind to deterministic ridge but costs 16x head inference). Do not describe `K=16` as
deployment-equivalent. Persist the exact store list, SSE/live counts, cadence, head/backbone steps,
sampling seed, and pooled integrated drift in the JSON result, written by atomic replacement.

Verified final result on 2026-08-20: production head-adaptation job 6682146 completed all 20,000
steps in 1:46:24 with `DONE=20000` and a complete final checkpoint. Evaluation jobs 6682676
(`K=1`) and 6682518 (`K=16`) both completed `0:0` on the same 2,839 held-out clips:

| policy | pooled ALL RMSE | arm RMSE | hand RMSE | arm drift at k=16 |
|---|---:|---:|---:|---:|
| training-mean constant | 0.00450194 | 0.00475885 | 0.00190077 | -- |
| frozen 100k ridge | 0.00264866 | 0.00280576 | 0.00100849 | 0.0431445 |
| adapted flow head, `K=1` | 0.00344370 | 0.00366065 | 0.00103438 | 0.0571781 |
| adapted flow head, `K=16` mean | 0.00291463 | 0.00309968 | 0.00083925 | 0.0483224 |

The adapted head beats the deployable constant on all eight tasks: pooled improvement is 23.5%
at `K=1` and 35.3% at `K=16`. This is genuine supervised head learning, unlike the earlier heads
that lost to the constant. It does not beat the frozen ridge overall: `K=1` is 30.0% worse and
`K=16` is 10.0% worse in pooled RMSE; they beat ridge on 2/8 and 3/8 tasks respectively. `K=16`
does beat ridge on right-hand RMSE by 16.8%. Averaging reduces pooled RMSE by 15.4%; under the
descriptive `MSE_K = bias + variance/K` decomposition, sampling contributes about 30.3% of `K=1`
MSE and the estimated infinite-sample floor is 0.00287590, still 8.6% worse than ridge. Integrated
drift remains strongly correlated: at k=16 the `K=1`/`K=16` rows are 32.5%/12.0% worse than ridge.

Therefore do not extend this head blindly to 200,000 steps. Twenty thousand steps already draw
2.56M clips, about 97 effective passes over the 26,269-clip train index. A convergence study needs
a validation split reserved before training and fixed checkpoint evaluations; do not repeatedly
select on the final 2,839-clip holdout. The evidence says optimisation/sampling improved greatly,
but the remaining pooled gap is mostly conditional-mean bias, not variance that more samples or
ten times more head updates can be assumed to remove. The exact post-adaptation artifacts are
`ftp1-runs/eval/f100k_univtac_o2a1_h16_flowmatch_s42_heldout.json` and
`ftp1-runs/eval/f100k_univtac_o2a1_h16_flowmatch_s42_heldout_k16.json`.

Monitor SLURM work from scheduler state (`squeue` while live, then `sacct` state plus exit code)
and require the expected result artifact or DONE marker. Do not use `pgrep -f <script-name>` as a
liveness test: the pattern can match the probe command itself. A waiter also needs a staleness
deadline or heartbeat; silence is not evidence that a slow process is still alive.
On the VSCode dev node, login-shell Conda initialization can itself stall before even `/bin/date`
runs. Execute scheduler monitors as non-login shells and retain an outer timeout; a hung shell
bootstrap is not evidence that Slurm or the watched job is unavailable.

The 100k time-shuffle control was recovered on a real GPU in job 6658717 (`COMPLETED 0:0`). On 12
batches of 32, its displacement/spread ratios were only `0.008--0.034` across sync/final and
pooled/unpooled variants, while shuffling the gel input itself gave `1.021`. The input therefore
contains temporal variation but the encoder is effectively order-blind: the current objective
does not reward temporal correspondence. This affects the explanation of the transfer null, not
the closed-loop gate or success-rate accounting.

The current head has task/domain-specific embeddings and output rows and ignores task text, so it
does not define genuine unseen-task zero-shot control. Closed-loop deployment now supports both
the corrected `RidgeHead` and a completed trained `ActionDiT` plus its checkpointed
`ActionNormalizer`. Treat the run contract as authoritative: require `DONE`, `latest`, metadata,
and configured head step to agree; strict-load the head/normalizer; and load the frozen backbone
path and step from `run_config.json` rather than a live preset.

For UniVTAC deployment use the exact sparse action mask (slots `9:16` and `44`), denormalize the
flow output before masking, integrate the seven arm first differences, pass slot 44 through as an
absolute gripper target, and execute MoT-JEPA chunk index 0. Preserve UniVTAC's native decoded
camera and tactile arrays numerically; do not apply an RGB/BGR reversal. Direct comparison against
the source Zarr and derived clip shows exact equality before reversal and large error after it.
Keep tactile pads in left/thumb then right/index order. Re-plan every control step. Use a dedicated
policy `torch.Generator` reseeded from the
fixed policy seed plus each simulator episode seed, so simulator RNG and policy RNG cannot affect
one another.

Run `scripts_exp_zarr/mot_jepa/closedloop.sbatch` with exactly two GPUs visible to its `srun` step:
Isaac/TacEx on logical `cuda:0`, policy inference on logical `cuda:1`, and literal `--gpu ""` so
the common harness does not hide the second device. On a shared-node smoke the batch allocation
must request exactly two GPUs directly, and the `srun` step must repeat `--gpus-per-task=2`.
Do not use `--exclusive`: a two-GPU exclusive allocation is canceled by the cluster's
allocated-idle-GPU monitor. Do not request all eight GPUs and try to mask six: CUDA honors
`CUDA_VISIBLE_DEVICES`, but Vulkan/Pyxis still enumerates every mounted physical GPU and Isaac
can fail with `ERROR_DEVICE_LOST`. Pin production evaluation to a node that has passed the exact
renderer/evaluator smoke when arbitrary shared nodes prove unreliable. The MoT-JEPA monkeypatch
is intentionally single-worker; both the wrapper and launcher force `--workers 1`. First run
`TOTAL=1` as a smoke.
A result is valid only
when the evaluator and `srun` return zero, the job-owned metadata contains the exact requested
seed set, every seed is `success` or `failed`, root and worker counts agree, there are zero errors,
and artifact/task provenance matches. Never accept a partial result or select outputs by mtime.

For the official FTP1 comparator pin checkpoint `19999`, `action_rep=mix`, executable chunk prefix
20, temporal ensembling with `ensemble_K=0.01`, and ten Euler steps. FTP1 chunk index 0 is the
current-timestamp placeholder, so its first executable index is 1; MoT-JEPA starts at index 0.
Use an episode-local `torch.Generator` seeded from `policy_sample_seed + episode_seed`; never draw
flow noise from the simulator/global RNG. The qualified Python-3.10 overlay lives at
`ftp1-runtime/univtac_ftp1_py310_v1`; it keeps Isaac's torch 2.13/CUDA 13, NumPy 1.26.4, patched
Transformers 4.53.2, JAX 0.5.3 CPU, and the tokenizer digest
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.

Every authoritative closed-loop job now snapshots all participating OpenPI/evaluator/task/TacEx
Python sources before import and rejects live drift at the end. It also validates the 37-GB
container digest, external and local asset-tree digests, policy checkpoint closure, tokenizer,
and (for FTP1) the resolved overlay tree. Save a bounded atomic trajectory NPZ per seed with
`--save_trajectory --trajectory_max_actions 500`; require exact seed coverage, finite 120-D raw
vectors, finite pre/post qpos and sent actions, no truncation, valid termination/result linkage,
and matching trajectory SHA-256 in worker metadata. Source silence is not health: monitor with
`squeue`/`sacct`, evaluator artifacts, and a deadline rather than `pgrep -f`.

Flow smoke job 6686063 verified this path on 2026-08-20: head step 20,000 and backbone step 100,000
loaded, the first `(16, 120)` chunk was finite, mixed-to-absolute debug matched the exact 8-D
command sent to TacEx, and the one-episode verdict was valid (`0/1`; a wiring smoke, not an
accuracy estimate). Production attempt 6686082 then hit an Isaac Vulkan `ERROR_DEVICE_LOST`
during `SimulationApp` startup on `gpu-h100-0338`; it created no episode metadata and the strict
verifier correctly rejected it. This is a node/runtime failure, not a zero-success policy result;
retry on a fresh allocation and exclude the failed node.

Jobs 6686098 (flow) and 6686081 (ridge) completed 0/20 but are superseded and must not be cited:
recovered deployment bytecode proves that both reversed the head camera and both tactile pads,
while the native numeric contract requires no reversal. Their successful exit codes validate only
the old harness mechanics, not current policy accuracy. Rerun flow, ridge, and FTP1 over the same
seed set and frozen source digest before publishing a matched table.

The source-pinned upstream-protocol smokes that supersede the old wiring rows are jobs 6687991
(flow), 6687992 (ridge), and 6687993 (official task-finetuned FTP1). They all completed with an
empty validation-error list, exact seed 1,000,000 accounting, untruncated trajectories, and the
same 367-file source-content SHA-256
`33ff31098a633fb1e9dc0478d6322106f57d68d0b2b10eb391a9d5de5a175a87`. Their direct smoke outcomes
were flow `0/1`, ridge `0/1`, and FTP1 `1/1`; these prove the end-to-end protocol and are not an
accuracy estimate. Keep UniVTAC `livestream=2` even when `HEADLESS=1`: `livestream=0` selects a
different Isaac experience and is not the upstream comparator protocol. The first matched n=20
jobs 6688120/6688121/6688122 preserved the expected source and artifact digests but all lost the
Vulkan device during startup on shared nodes 0088/0132/0135, before episode metadata existed.
They are infrastructure-invalid and have no success rate. The first exclusive retries
6688320/6688321/6688322 requested all eight GPUs and were correctly stopped by preflight because
the unqualified launcher exposed all eight devices; no simulator episode ran. Retries
6688340/6688341/6688342 requested only two GPUs under `--exclusive`, passed simulator startup,
and were then canceled by the cluster's allocated-idle-GPU monitor because six node GPUs were
reserved but unrequested. No episode completed. Full-node preflight 6688589 exposed all eight
devices and stopped before evaluation. Full-node mask smoke 6688624 made Torch see two devices,
but Vulkan still enumerated all eight and Isaac failed with `ERROR_DEVICE_LOST`; therefore the
full-node-plus-mask topology is invalid. Initial exact-two submissions 6688713/6688714 failed the
launcher guard before evaluation because `sbatch --export` parsed the comma in an attempted
escaped `EVAL_CUDA_VISIBLE_DEVICES=0,1` value as a delimiter; 6688715 and dependency-held adapted
jobs 6688728--6688730 were canceled. Omit that export and use the launcher's default, or use an
export file--a backslash in the comma-delimited `--export` string is not sufficient. The
authoritative n=20 retries are exact two-GPU shared allocations pinned to official-smoke-qualified
`gpu-h100-0044`: flow 6688767, ridge 6688768, and FTP1 6688769, with seeds
1,000,000--1,000,019. All three completed `0:0`, strict `ok=true`, zero errors, identical
367-file source digest `26f607c6d61fc6bd9e98313d4864bcc90e8220f7970b8ea6d714daf682c4e151`,
and 20 untruncated trajectory hashes per arm. Direct success was flow `1/20 = 5%` (only seed
1,000,008, after 183 actions), ridge `0/20 = 0%`, and official task-finetuned FTP1
`20/20 = 100%`. Every other flow/ridge episode reached the 500-action limit; every FTP1 episode
terminated in success.

Fine-tuned flow n=20 jobs are frozen 6688770 on `gpu-h100-0063`, last-two blocks 6688771 on
`gpu-h100-0196`, and full backbone 6688772 on `gpu-h100-0044`. Their historical pair/run labels
still contain `node0162`, so use SLURM's actual `NodeList`, not the label, for topology provenance.
All completed `0:0` and passed the 367-source snapshot, 317-artifact dynamic-policy closure,
exact-two-H100 preflight, strict adapted-backbone load, final source/artifact drift checks, exact
20-seed accounting, and untruncated trajectory validation. Direct success was frozen extra-20k
control `1/20 = 5%` (seed 1,000,007), last-two `2/20 = 10%` (seeds 1,000,006 and 1,000,007),
and full `0/20 = 0%`. Thus last-two is numerically best, but its one-extra-success gain over the
source/frozen 5% rows is too small for a strong claim at n=20; full adaptation made offline RMSE
best on `lift_bottle` while making closed-loop success worst. Official FTP1 remains `20/20`.

Use exact 95% Clopper--Pearson intervals when reporting this n=20 panel: `0/20` is
`[0%, 16.84%]`, `1/20` is `[0.13%, 24.87%]`, `2/20` is `[1.23%, 31.70%]`, and `20/20` is
`[83.16%, 100%]`. Exact paired McNemar tests find no supported difference among the source and
adapted MoT-JEPA arms (source/frozen, source/last-two, and source/full all `p=1`; last-two/full
`p=0.5`). The official FTP1 gap is decisive on the paired seeds (`p=3.81e-6` versus source flow
and `p=1.91e-6` versus ridge). Do not call the MoT-JEPA control rows zero-shot: their pretraining
is label-free, but both the flow head and ridge readout use supervised UniVTAC actions. FTP1 is
the official task-fine-tuned comparator.

The six jobs use matched seeds and protocol, but independent Isaac resets are not bitwise-input
matched. The source/full pair ran on the same node yet had the largest source-relative captured
input difference, while the cross-node source/frozen pair was closest. For source/full, pre-policy
capture differences were qpos RMSE `0.002809 rad` (about `0.161 degrees`, maximum `0.004754 rad`),
normalized video RMSE `0.003914` (about half an 8-bit level on average), and gel RMSE `0.004499`;
the low-dimensional schema, seed, channel ordering, and required zero slots were exact. Because
these captures occur before encoding or action sampling, attribute the small differences to Isaac
reset/physics/render nondeterminism (with a smaller node component not excluded), not to policy
output. This does not weaken the 100%-versus-at-most-10% FTP1 gap, but it is another reason not to
over-interpret last-two's single additional success. Describe this panel as matched-seed and
matched-protocol, not bitwise-observation matched.

### MoT-Control V3: task-specific path to the official success level

The low closed-loop rate above is treated as a method failure, not a request for another blind
pretraining extension. MoT-Control V3 replaces the open-loop pooled readout with a state-conditioned
dense decoder: all 16 video/GEL observations remain available as tokens, a 27-D state vector carries
the qpos/command history and presence masks, and 32 learned horizon queries emit the exact next-command
8-D chunk. Auxiliary approach/close/lift/release phase, contact, and temporal-order losses expose the
transition structure that the order-blind self-supervised objective did not learn.

The preregistered production chain is implemented by
`scripts_exp_zarr/mot_jepa/control_v2_submit_pipeline.sh`:

```text
4 independent one-job/one-GPU collection smoke shards -> fail-closed CPU merge
  -> 4 independent 250-episode shards -> exactly 1,000 scripted successful lift_bottle episodes
  -> fail-closed CPU merge with stride-4 seed/topology/content closure
  -> authoritative V3 Zarr preparation + full content digest
  -> frozen-backbone head training, 20,000 updates
  -> final-four-block adaptation, 10,000 updates
  -> n=1 wiring/trace gate
  -> n=20 gate requiring at least 19 successes
  -> paired n=100 student/official evaluation
```

Pipeline `v3_official_20260821T055855Z` is superseded and all jobs 6697991--6698000 were
canceled. Its four-worker smoke put four Isaac applications in one four-GPU Pyxis step: all
started an HTTP server on port 8011, Omniverse's physical enumeration differed from
`CUDA_VISIBLE_DEVICES`, two renderers lost their Vulkan device, and no HDF5 episode was produced.
Four one-GPU tasks inside a single allocation also proved invalid in job 6698189: only one task
could construct a Vulkan device and the other three exited with `ERROR_INCOMPATIBLE_DRIVER` and
segfaulted. A one-GPU job on arbitrary node 0181 (6698267) reached an active renderer but then lost
the device during texture upload. These are infrastructure-invalid startup failures, not 0%
policy or expert success rates.

The qualified collection topology is now four independent Slurm jobs, each owning one GPU, one
Pyxis container, one spawned Isaac process, a job-derived HTTP port, and one seed residue class:
shard rank `r` attempts `2,000,000 + r + 4k`. Failed expert attempts therefore cannot cross into
another shard's seeds. Each shard records its physical GPU UUID, job, node, port, source closure,
exact local quota, and content hashes; a CPU merge independently verifies all four manifests,
requires ranks 0--3 and distinct jobs/ports, rejects duplicate or wrong-residue seeds, and creates
the authoritative parser hardlinks and 4/1,000-episode manifest. Collection is pinned to the
renderer-qualified node used for official evaluation. Standalone topology qualification job
6698419 completed 1/1 scripted success in 877 steps with zero collector errors on node 0044 and
wrote a valid 827,664,634-byte HDF5, post-run runtime qualification, content manifest, and `DONE=1`.
It predates the final schema-3 integrity hardening and is intentionally not reused as official
input: changing the verifier invalidated its frozen source closure as designed.

Official replacement pipeline `v3_official_20260821T071953Z` was submitted as jobs 6698918--6698935.
Jobs 6698918--6698921 are the four fresh one-episode smoke shards and 6698922 is their `afterok:all`
CPU merge. Jobs 6698923--6698926 are the four 250-episode production shards; 6698927 merges the exact
1,000, 6698928 prepares the store, 6698929--6698932 run head/adaptation stats and training, and
6698933/6698934/6698935 are the n=1, n=20, and paired n=100 closed-loop gates. At submission the four
fresh smoke shards were pending for node 0044; every later job was scheduler-held by the expected
`afterok` dependency.

That four-by-250 chain is superseded by the lower-risk one-job/one-GPU batch-20 chain
`v3_official_batch20_20260821T152025Z`. Jobs 6704586--6704605 collect 20 independently committed
50-episode residue shards; 6704606--6704613 are merge, preparation, head statistics/training,
adaptation statistics/training, n=1, and n=20. Paired n=100 shards 6705054--6705058 and CPU merge
6705059 are appended after the n=20 gate. All 20 collectors completed `0:0`; merge 6704606 then
closed and reverified all 1,000 episodes, all 20 ranks/residues, source/runtime hashes, HDF5
payloads, and hardlinks before writing `DONE=1000` and completing `0:0`.

The first preparation job 6704607 failed before conversion because
`control_v2_prepare_data.sbatch` selected the legacy collection verifier. That verifier correctly
remains frozen for the legacy four-shard pipeline but hard-codes `shard_count == 4`; the valid
batch merge has `shard_count=20`. The collection was not corrupt. Do not edit either provenance
module to repair an already frozen merge: both files are hashed into the collection's source
closure. Preparation now accepts an allowlisted `COLLECTION_PROVENANCE_SCRIPT`, the batch launcher
exports `mot_jepa_control_v2_provenance_batch.py`, and prep freezes the selected verifier in its
own source provenance. Its EXIT trap is installed before verification so early failures write
`FAILED.json`. The original empty partial prep root is preserved, and recovery prep 6716916 uses a
fresh job-owned root against immutable merge 6704606. Focused submit/merge/provenance regression
tests passed. Jobs downstream of failed prep 6704607 were dependency-cancelled and must not be
reused; after recovery prep acceptance, submit a fresh uniquely named `PREPARED_STORE` pipeline
and paired n=100 chain. Collection/preparation progress is not a V3 success rate; only the paired
n=100 acceptance artifact may populate that row.

Recovery prep 6716916 completed `0:0` with schema-3 `status=DONE`, 1,000 episodes, 736,486 rows,
97,301 store files, and derived-store content digest
`9b7cae8d0e9caf1bd45be3ea8b94205f6341536d5e982b794abe3eee0e4e386e`. The authoritative store is
the exact path written by its prep-root `DONE` marker under the resolved `/edgeai/projects/...`
run-tree prefix; the shorter `/edgeai/users/chrislin/...` prefix is the same tree through a symlink.
Do not look for provenance inside the Zarr directory: `DONE`, `manifest.json`,
`source_provenance.json`, and `store_content_provenance.json` live at prep root, while the store is
`clips/UniVTAC_lift_bottle/lift_bottle_head.zarr`. The qualified backbone value remains the run root
`.../backbones/forecast100k_qknorm_s100000`; stats/training append `checkpoints/100000` themselves.

The fresh downstream pipeline is
`v3_official_batch20_recovery_prep6716916_20260822T074950Z`. Head stats/train are 6718855/6718856,
adaptation stats/train are 6718857/6718858, and the n=1/n=20 gates are 6718859/6718860. The
authoritative paired n=100 append is five 20-trial shards 6718894--6718898 at starts
1,000,000/1,000,020/1,000,040/1,000,060/1,000,080 plus CPU merge 6718899. Its immutable submission
manifest is `logs/control_v2_paired_n100_v3_official_batch20_recovery_prep6716916_20260822T074950Z.json`;
the shards depend directly on `afterok:6718860`, and the merge depends on all five shards. At
2026-08-22 05:31 PDT, first stats job 6718855 was pending CPU capacity with a scheduler-estimated
start of 18:33 PDT; every later job was correctly dependency-pending. This is a healthy submitted
chain, not a measured success rate. Report V3 and official SR only from merge 6718899's verified
`acceptance.json` after its full marker/hash closure passes.

That submitted-chain status was superseded when head job 6718856 reached all 20,000 updates and
then failed its scientific selection gate. The numerical run did not diverge: held-out action MAE
improved from 0.005442 to 0.000966 and joint-limit violations remained zero, but rate-limit
violations only fell from 29,963 to 9,744 of 81,920 retained values and the worst ratio remained
72.983. No checkpoint wrote `BEST_VALIDATION.json` or `DONE`; Slurm therefore canceled jobs
6718857--6718860 and 6718894--6718899 through their `afterok` dependencies. This is an unknown
closed-loop success rate, not 0%: zero rollout episodes ran.

An independent decode of the exact frozen 512-sample/144-episode validation panel proves the
contract is feasible. Expert targets have 0/81,920 rate violations and maximum ratio 0.791717;
hold targets reproduce step zero's 29,963 violations and 111.153 maximum ratio. Therefore command
`t+h` versus qpos `t+h-1`, physical units, normalization, and fitted limits are correct. The method
failure was an optimization mismatch: the old global-mean hinge supplied no gradient inside the
margin and underweighted rare tails, while selection requires every value to pass.

The recovery method keeps the exact expert targets, physical limits, validation identities,
zero-violation selector, deployment limiter, and closed-loop acceptance unchanged. It adds a
first-20-row safety tube with remaining expert slack
`s = max_delta - |expert_command - reference_qpos|` and normalized error
`|prediction - expert_command| / s`. A dimensionless Smooth-L1 mean plus each example's maximum
tail is weighted by `safety_tube_loss_weight=0.05`; non-positive expert slack fails closed. The
triangle inequality makes tube ratio at most one a sufficient certificate for the unchanged rate
gate. Because the method/config source changed, preserve 6718856 as failed and launch a fresh head
run; never relabel its numeric checkpoint as deployable or weaken the gate.

The provenance-closed recovery chain is `v3_official_safetytube_prep6726442_20260823T093043Z`.
Preparation 6726442 reuses immutable collection merge 6704606 but rebuilds the derived store because
the method/config source changed. Jobs 6726530/6726531 are head stats/train; 6726532/6726533 are
adaptation stats/train; 6726534/6726535 are n=1/n=20. The authoritative paired n=100 append is
6726544--6726548 plus merge 6726549. Every downstream job is `afterok`-gated; no rollout can start
from an unsafe or incomplete checkpoint. The pipeline and paired submission manifests live under
`logs/control_v2_pipeline_v3_official_safetytube_prep6726442_20260823T093043Z.json` and
`logs/control_v2_paired_n100_v3_official_safetytube_prep6726442_20260823T093043Z.json`.

That historical chain did not enter training. Preparation 6726442 and head stats 6726530
completed, but head job 6726531 failed before step 1 because the old validation split hashed the
resolved prepared-store path. Rebuilding byte-identical data under a new job-owned directory
changed the split from 856/144 to 835/165 and placed three non-positive expert-slack values from
two episodes into the frozen panel; every dependent job was then canceled. Never resume or reuse
that run root.

The repaired `source_episode_seed_v1` split hashes the immutable collection episode seed instead
of a path. On collection 6704606 it is exactly 850 train / 150 validation, independent of the prep
directory. An exact CPU audit covered all 110,513 held-out transitions with zero violations,
minimum slack `0.000102410`, and maximum rate ratio `0.923366`; the frozen 512-example panel also
has zero violations, minimum slack `0.000298796`, and maximum ratio `0.819761`. Production data
now carries `meta/source_episode_seed`, data schema v2 and artifact/checkpoint schema 4. Stats
freezes `VALIDATION_SAMPLES.json` and completes this oracle audit before writing `STATS_DONE`, so
the same class of split/safety failure cannot consume a training GPU again.

The fresh formal replacement is
`v3_official_safetytube_splitfix_prep6734747_20260824T015146Z`. Preparation 6734747 rebuilds the
store from immutable collection 6704606; jobs 6734757/6734759 are head stats/train,
6734761/6734766 are adaptation stats/train, and 6734772/6734774 are the n=1/n=20 student gates.
The paired official panel is five 20-seed shards 6734898--6734902 plus CPU merge 6734903. Every
edge is an explicit `afterok`; the paired jobs were submitted atomically and released only after
their immutable append manifest was committed. The pipeline and paired manifests are
`logs/control_v2_pipeline_v3_official_safetytube_splitfix_prep6734747_20260824T015146Z.json` and
`logs/control_v2_paired_n100_v3_official_safetytube_splitfix_prep6734747_20260824T015146Z.json`.

Treat renderer qualification as a physical-GPU property, not merely a node property. Slurm
`--gpu-bind=single:1` binds the GPU selected in the allocation; `ReqNodeList` cannot select a proven
GPU UUID. Before widening a collector's eligible nodes, require successful production-topology
coverage of every allocatable GPU index on every added node. Eight overlapping one-GPU batch-20
jobs proved all eight UUIDs on `gpu-h100-0044`; nodes 0018, 0023, 0024, 0284, 0295, and 0324 had
only one or two proven UUIDs each. Consequently all remaining batch-20 shards were pinned to 0044,
including 6704599 and 6704601, which were moved from partially qualified 0295/0324 while still
pending. A device failure is fail-closed but not automatically recoverable: collectors use
`Requeue=0`, and `afterok` plus `kill_invalid_depend` can cancel the downstream chain.

Do not change a live job's partition from a generic `srun`/`sbatch --test-only` prediction alone.
For these pending collectors, nonallocating tests repeatedly predicted openings hours earlier than
the actual aged-job `squeue --start` estimates. Future estimates also serialized jobs even though
the same eight-resource shape had already packed concurrently on node 0044. More importantly,
`backfill` has `PreemptMode=REQUEUE`, while official collectors intentionally use `Requeue=0`; a
higher-priority preemption can therefore cancel rather than recover a collector and break the
`afterok` chain. Keep official collectors on non-preemptible `batch`. Validate node-placement
optimizations against actual aged jobs, but treat their future start times only as conservative
scheduler hints rather than evidence that packable jobs will execute serially.

Isaac's multiprocessing shutdown can print repeated `Exception ignored` tracebacks ending in
`sem_unlink(...): FileNotFoundError` after `[Final] 50/50`. This is benign cleanup noise only when
the job subsequently reaches `COMPLETED 0:0`, writes exact `DONE=50`, has byte-identical runtime
qualification before/after collection, and passes both source-provenance and full-collection
verification. Never accept it from log context alone, but do not classify the bare word
`Traceback` as a shard failure without checking the terminal state and artifacts.

Schema-3 collection verification is not downgradeable. It binds the canonical in-root runtime
config, source provenance, qualified runtime before/after collection, shard/topology records, and
raw/parser HDF5 direct members. A schema change, relative/traversing pointer, symlink substitution,
source-profile change, unofficial runtime identity, duplicate episode row, missing HDF5 field, or
non-hardlinked parser member fails before merge or preparation.

On the current cluster, CPU-only jobs must use partition `cpu`, whose nodes expose about 246 GiB;
both preparation and stats request 220G. A 500G CPU request is unschedulable even though the GPU
nodes can satisfy it. The pipeline launcher now traps any mid-submission error and cancels every
job ID already accepted, preventing a failed submission from leaving a partial dependency chain.
The collection merge uses the CPU partition at 160G and 24 CPUs; shard collection uses one GPU,
16 CPUs, and 125G per independent job.

Training uses every control anchor (`index_step=1`) and an episode-level train/validation split.
The nominal 512-example validation panel expands if needed to cover every held-out episode, is
balanced across early/middle/late episode time, all four phases, and both contact classes, and
cycles all 32 deployment cold starts (history lengths 1--16 crossed with both stride-2 command
parities). Its structured identities are frozen in `VALIDATION_SAMPLES.json`; an old sampler,
missing stratum, duplicate/tampered source row, or scenario gap fails closed. Step 0 is evaluated
but is selectable only if it is safe. `BEST_VALIDATION.json` first requires zero held-out joint or
rate-limit violations over the first 20 executable horizons, then selects minimum deployed-horizon
action loss with earliest-step tie breaking. Therefore neither 100k nor 200k updates is assumed
sufficient by fiat: the fixed validation curve selects convergence, and the n=20 gate stops an
inadequate policy before the expensive official comparison.

Deployment replans every control step, executes H32 row 1 (row 0 is the current-command
placeholder), ensembles only the first 20 predictions with `K=0.01`, and rejects non-finite,
clamped, reversed, joint-limit, or discontinuous-gripper traces. The final official-level gate is
fixed at paired seeds 1,000,000--1,000,099: the student must succeed on at least 99/100 episodes,
all 100 student traces must qualify, and the exact one-sided 95% Clopper--Pearson upper bound on
official-only discordance must be below 0.05. The verdict is invalid unless it is tied to the
byte-qualified container, external/local assets, FTP1 overlay, checkpoint 19999, READY marker,
tokenizer, policy artifact, validation selection, and immutable source/data provenance.
Raw demonstrations are content-hashed individually and their parser hardlinks are verified on
both sides of conversion. Live head/GEL inputs also reproduce the collector's default JPEG
encode/decode loss before the 224-linear and GEL-112-area resize stages; bypassing JPEG is a
training/deployment preprocessing mismatch. Adaptation freezes the exact prior held-out-selected
checkpoint, and every stage uses an empty job-owned Python bytecode cache so excluded stale `.pyc`
files cannot enter execution.

The controller loss includes a normalized joint/rate safety hinge with weight `0.25` plus the
gate-aligned safety-tube tail objective above with weight `0.05`. The rate
reference for command `t+h` is measured qpos at `t+h-1` (never the previous command), and its
training margin is `0.9` of the train-fitted deployment limit. Exact validation diagnostics use
the real `1.0` limit; counts and maxima are preserved in the selected-checkpoint marker and
independently checked by deployment and acceptance. The limiter also counts its final absolute
clip when the observed current state starts outside a joint bound, so no changed command can hide
behind a zero-clamp verdict.

Do not report a V3 success rate from training loss, offline RMSE, or the n=1 wiring run. Until the
closed-loop jobs finish, its success rate is unknown. Recovery training is intentionally outside
the first chain: only after a valid n=20 run may its actual failed states be expert-corrected and
frozen into a new recovery-data contract.

If a completed numeric training schedule is intentionally evaluated despite failing the held-out
safety selector, keep that path structurally separate from production deployment. Use
`control_v2_submit_diagnostic.sh`: it submits a `DIAGNOSTIC_ONLY / NOT_OFFICIAL` n=1 smoke with an
`afterany` dependency on the originating training job, followed by an n=20 rollout only after the
smoke completes technically. The diagnostic preflight accepts only the exact configured final
numeric checkpoint, `checkpoints/latest == final`, the exact qualification-only `FAILED` markers,
no `DONE`, no `BEST_VALIDATION.json`, `best_validation_step/loss == None`, closed optimizer and
artifact metadata, unchanged training/data provenance, and a final validation row that actually
failed qualification. It never creates formal completion/selection markers, never runs the
official comparator or official seed interval, and writes `diagnostic_result.json` plus
`DIAGNOSTIC_EVAL_COMPLETE` rather than `acceptance.json` or `DONE`. This yields a measured
head-only diagnostic success rate without mislabeling the checkpoint as safe, adapted, or official.

For the split-fixed safety-tube head job 6734759, diagnostic jobs 6739557/6739558 were submitted on
2026-08-24 as smoke seed 900000 and n=20 seeds 900100--900119 on qualified node 0044. Their immutable
manifest is `logs/control_v2_diagnostic_v3_official_safetytube_splitfix_prep6734747_20260824T015146Z.json`.
The original adaptation and formal evaluation chain remains untouched; these diagnostic jobs do
not establish the preregistered V3 or official paired-n100 verdict.

That first diagnostic chain is superseded. Smoke 6739557 passed the host-side training, prepared-
store, checkpoint, runtime, and source-provenance checks but failed before policy construction or
any episode because the inference-only Isaac Python does not install Zarr: importing the runtime
configuration reached the training-only `control_v2_dataset` module. Job 6739558 was consequently
canceled without starting. Do not install the host Python environment into Isaac or edit the
frozen `src/openpi` modules, because either would compromise runtime compatibility or invalidate
the training source closure. The diagnostic evaluator now installs a narrow module stub exposing
only the frozen `CONTROL_V2_SPLIT_IDENTITY=source_episode_seed_v1` constant, and only under both
the explicit unqualified-final opt-in and an outer-store-verification flag. It disables the
in-process Zarr reread while the provenance-closed outer job verifies the prepared store before
launch and again after rollout. A missing-Zarr subprocess regression, the focused diagnostic
suite (7 tests), Ruff, shell syntax, launcher dry run, and Slurm topology test all pass.

Fresh diagnostic-only retry jobs are smoke 6750298 and n=20 6750299, using the same fixed seed
intervals and qualified node 0044 with `afterok:6750298` on the n=20 job. Their new immutable
manifest is
`logs/control_v2_diagnostic_v3_official_safetytube_splitfix_prep6734747_zarrfix_20260825T011427Z.json`;
the failed roots and manifest remain untouched. Both retries completed `0:0` on node 0044:
6750298 ran in 17m30s and 6750299 in 45m28s. The smoke was `0/1`, and the complete diagnostic
panel was `0/20 = 0%` task success (exact two-sided 95% Clopper--Pearson interval
`[0%, 16.84%]`). All 20 requested seeds and trajectories were present, finite, untruncated, and
content-hashed; runtime/source/prepared-store verification passed before and after rollout. Only
3/20 traces passed the no-clamp safety qualification, while 17/20 used the deployment limiter;
there were no non-finite fallbacks. This is a valid negative diagnostic result for the final
unqualified frozen head, not an infrastructure failure, formal V3 result, adapted-policy result,
or official paired comparison.

When training and the n=20 gate are already submitted, append the final paired panel with
`scripts_exp_zarr/mot_jepa/control_v2_submit_paired_n100.sh`; do not resubmit the training chain.
Provide the exact `V2_RUN`, existing `N20_JOB`, dedicated `PAIR_PARENT`, and qualified
`EVAL_NODE=gpu-h100-0044`, then set `SUBMIT=1` only after inspecting its dry run. The launcher
submits five batch-only, four-hour paired student/official shards over starts 1,000,000 through
1,000,080 and one non-requeueable CPU merge. It holds all six jobs until an atomic JSON submission
manifest closes their IDs, roots, dependencies, and launcher/sbatch hashes, then releases them
together; any submission or release failure cancels the accepted jobs. Every shard revalidates
that manifest, the exact evaluator arguments, full hashed source/TacEx profile, and the five
qualified stat-only runtime roots. The merge reconstructs raw n=100 metadata and all 100 student
traces without averaging, invokes the unchanged acceptance program, post-verifies raw/source
provenance, and distinguishes a scientific `REJECT` from an integrity `INVALID` failure.

#### Atomic eight-GPU renderer qualification and pending-job moves

When eight one-GPU collectors must be moved to a new node, qualify all physical GPUs in one
atomic allocation before changing any production job. Use
`control_v2_submit_gpu_coverage_atomic.sh`: it requests exactly 8 GPUs, 128 CPUs, and 1000G,
launches eight concurrent one-GPU/16-CPU/125G collector steps, and records a barrier only after
three stable observations plus a precommit recheck prove eight live numeric Slurm steps, eight
direct topology files, eight distinct UUIDs/ports, and no completion or failure marker. The CPU
gate then requires `ATOMIC_DONE=8`, exact `sacct COMPLETED 0:0`/allocation TRES, all eight full
collection closures, and the independent 50-episode production reference. Node 0222 was verified
by allocation/gate 6710397/6710759; node 0350 was verified without repair by 6710772/6710773;
nodes 0071 and 0277 were verified without repair by 6711115/6711116 and 6711176/6711177; and
newly rebooted node 0026 was verified by 6712327/6712328 in 8m19s plus a 25s gate. All eight 0026
slots committed, used distinct physical UUIDs, and produced canonical `status=VERIFIED` coverage.
The node was immediately allocated to another full-node job, so it was not initially added to
production collectors that already held earlier PLANNED node assignments; do not discard an
earlier scheduler placement merely because an additional qualified node exists. It was added later
only when a fresh exact full-node slot materially improved the then-current collector tail. Node 0237 was subsequently verified
by allocation/gate 6712537/6712538: eight clean commits, eight distinct UUIDs, and canonical
`status=VERIFIED`. Its exact four-hour availability was materially earlier than the then-current
collector plan, so it was added to the aged collector pool while every old/new node was fully
allocated; all eight jobs remained pending/rootless and retained their original eligible time.
Node 0109 then passed allocation/gate 6713098/6713099 with eight clean commits and distinct UUIDs;
its earlier exact full-node slot justified the same guarded pool expansion. Node 0017 passed
allocation/gate 6713393/6713394 with the same eight-slot closure. After verification it transitioned
from `IDLE+PLANNED` to an actual full-node allocation, so it was added with the same guarded,
age-preserving direct expansion.
Never infer node health from a committed concurrency barrier alone: nodes 0002, 0011, 0014, 0016,
0021, 0031, 0068, 0178, 0194, 0249, 0268, 0312, 0325, and 0362 exposed eight distinct UUIDs but one or more Isaac
workers then exited with
SIGSEGV before readiness or any attempt. On 0325, atomic job 6712570 reached a verified concurrent
barrier and then slot 1 committed `FAILED.json` for worker exit `-11`; on 0249, atomic job 6712564
reached the barrier and slots 0 and 2 independently failed with the same `-11` signature; on 0312,
atomic job 6712741 reached the barrier before slot 5 failed identically; and on 0031, atomic job
6712928 failed before barrier commit after slot 3 exited `-11`; on 0268, atomic job 6712959 reached
the barrier before slot 1 failed identically; and on 0011, atomic job 6713027 failed before the
barrier after slots 4 and 5 exited `-11`; on 0362, atomic job 6713052 reached the barrier before
slots 0 and 5 failed identically; and on 0194, atomic job 6713211 reached the barrier before slots
0, 1, 3, and 5 failed identically. The invalid atomic jobs and their gates
6712571/6712565/6712742/6712929/6712960/6713028/6713053/6713212 were canceled or failed to release
the remaining resources. On 0178, atomic job 6713348 failed before barrier commit after slot 0
exited `-11`; gate 6713349 was canceled. Their slot/atomic failure markers correctly prevented the
CPU gates from verifying them. Quarantine those nodes for this workflow unless a fresh full
atomic gate later proves all eight smoke roots.

Three shell details are cluster-critical. A Slurm batch script's `BASH_SOURCE[0]` may be the
controller spool copy, so participating scripts must use the explicit repository path rather
than deriving the repository from `BASH_SOURCE`. Create the parent of a fresh qualification
directory with `mkdir -p` before the strict one-time `mkdir` of the qualification directory.
For inner `srun` steps, `SLURM_EXCLUSIVE=1` and `SLURM_EXCLUSIVE=yes` are invalid parser values on
this Slurm build; use `SLURM_EXCLUSIVE=user` and validate the exact environment/options with
`srun --test-only`. Keep `--cpus-per-task=16`, `--gpus-per-task=1`, 125G, and the concurrency
barrier even though `-c` implies exact CPU allocation.

Lustre exposes the run tree through both `/edgeai/users/chrislin/...` and its resolved
`/edgeai/projects/.../users/chrislin/...` path. Manifest member checks must require an absolute,
traversal-free path, resolve it strictly, compare it to the canonical in-root member, and then
read only the canonical direct regular file. Lexical path equality falsely rejected the accepted
6704586 reference in gate 6710398. The corrected check has positive alias and negative
relative/traversal tests. The immutable GPU allocation did not need rerunning: a fresh CPU-only
repair manifest bound the original manifest/failure hashes, corrected verifier hash, completed
allocation, barrier, and eight roots; repair gate 6710759 then wrote the canonical verified
coverage artifact. Never edit the original submission manifest or reuse it after a verifier hash
changes.

Move aged production jobs only while every target is still `PENDING`, `Requeue=0`, never started,
and its exact output root is absent. Normally hold every job first; verify rank/seed/resources,
empty `NodeList`, and the merge's exact unfulfilled dependency set; update `ReqNodeList` while
held; reverify; then release. Holding and releasing resets `EligibleTime`, however. A direct
per-job update may preserve age only when the old requested node is independently proven fully
allocated through the entire short transaction, so none of the targets can race into execution;
capture and require byte-identical `EligibleTime` before/after, roll back every partial update on
error, and still require pending/rootless state and exact resources for every job. This controller
rejects comma-separated IDs for `scontrol hold/update`, so perform those mutations per job.
Per-job `squeue --start` estimates may serialize jobs that can pack concurrently—the completed
batch-20 history had eight 1-GPU jobs overlap on node 0044. Each collector has
`OverSubscribe=OK`, exact 1 GPU/16 CPU/125G `ReqTRES`, and no exclusive allocation; eight fit in
the 8 GPU/128 CPU/2010G H100-node capacity. Compare candidates with an exact full-node
8-task/8-GPU `srun --test-only`, but still monitor the aged jobs after release and revert only
while all remain pending/rootless. At 2026-08-21 17:58 PDT, a fresh exact
8-GPU/128-CPU/1000G four-hour test showed verified node 0071 materially earlier than node 0277.
Because 0277 was independently proven fully allocated by a running eight-GPU job throughout the
transaction, all eight remaining batch-20 collectors were first moved directly to 0071 without a
hold. They were then generalized, still pending and rootless, to the exact independently verified
pool `gpu-h100-[0071,0222,0277,0350]` so Slurm can choose the earliest safe node without further
placement chasing. After 0237 passed allocation/gate 6712537/6712538 and its exact four-hour test
demonstrated an earlier slot, the pool was safely expanded to
`gpu-h100-[0071,0222,0237,0277,0350]`; verified 0026 was then added after its exact full-node slot
improved the tail. Verified 0109 and 0017 were subsequently added under the same guard. Four
consecutive live forecasts then showed a sustained tail at 03:36--03:37 while exact 0044 capacity
forecast at 01:45--01:56. Node 0044's eight overlapping production collectors covered eight
distinct physical UUIDs with exact 1-GPU/16-CPU/125G topology, schema-3 closure, and `COMPLETED
0:0`; its boot epoch had not changed. This direct production evidence satisfies the same
physical-GPU coverage requirement, so 0044 was added, producing the current pool
`gpu-h100-[0017,0026,0044,0071,0109,0222,0237,0277,0350]`. Their
`EligibleTime=2026-08-21T17:18:42`, IDs, resources, seed residues, and downstream dependencies
remained unchanged, and they stayed on non-preemptible `batch`.
Additional nodes must pass the same atomic eight-GPU gate before entering this pool or receiving a
collector, unless direct overlapping production runs prove all eight physical UUIDs with the same
topology, full closure, clean terminal states, and no intervening reboot.

Keep guarded pool updates low-load. During the 0044 expansion, an O(jobs x nodes) sequence of
repeated `scontrol` reads produced controller responses longer than ten seconds and exposed an
intermediate six-of-eight view while the per-job updates settled. Snapshot all invariant fields
once, recheck only the immediately targeted job and newly added node during mutation, then perform
one full post-snapshot. Never infer rollback or completion from a transient partial read; require
one consistent post-snapshot of every job, output root, eligible time, resource request, and merge
dependency before declaring the transaction committed.

### Isaac Sim closed-loop eval: VERIFIED WORKING on this cluster

`scripts_exp_zarr/mot_jepa/isaac_probe.sbatch`, job 6442243, **PASS**:

```json
{"frame_shape": [240, 320, 4], "frame_dtype": "uint8",
 "nonblack_fraction": 0.9996, "pixel_std": 57.7, "verdict": "PASS"}
```

Isaac Sim 4.5.0 starts headless with `enable_cameras` and the RTX renderer produces a real
frame on an H100 inside `nvcr.io#nvidia/isaac-sim:4.5.0`. H100 having no RT cores is not a
blocker — consistent with UniVTAC's own "A800 headless" workaround at `envs/_base_task.py:114`.

Two things worth knowing:

- **The `sudo apt-get` steps in `Installation_FTP1.md:105,120` are unnecessary in the
  container**, which matters because there is no root on compute nodes. The probe found
  `/usr/share/vulkan/icd.d/` **empty** and no `nvidia_icd.json`, yet rendering works:
  `libnvidia-glcore` and `libnvoptix` are present and pyxis injects the driver, so Isaac
  resolves it without the standard ICD file. Do not "fix" the missing ICD.
- **Verify the frame, not the exit code.** Isaac runs with `--/app/fastShutdown=True`, which
  can `_exit()` and discard buffered stdout. The first probe attempt exited 0 with every
  decisive `print` missing from the log — trivially mistaken for a pass. The probe now writes
  its verdict to `logs/result.json` and the batch script reads the verdict from that file.

Eval Track B is therefore unblocked: fine-tune on `{lift_bottle, lift_can, insert_tube,
put_bottle_in_shelf}` and evaluate held-out on `{grasp_classify, insert_hole, pull_out_key,
insert_HDMI}`. Keep `livestream=2` for the upstream UniVTAC protocol: `livestream=0` loads a
different Isaac headless rendering experience and produced different initial observations and
outcomes for the same seed. A headless result is a new protocol, not a transparent stability
workaround. The `--tactile_mode` ablation injects using `openpi.mot_jepa.ablation`.
