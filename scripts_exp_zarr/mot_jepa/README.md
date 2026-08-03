# MoT-JEPA on the H100 cluster

A long run here is a **chain of 4-hour jobs**, not one process. Everything in this directory
exists to make that chain behave like a single continuous run.

## Smoke ladder — run all four before anything long

Each step answers a question the next one assumes. Skipping them means discovering a
scheduler-shaped problem after burning hundreds of GPU-hours.

```bash
mkdir -p logs
export REPO_ROOT=$PWD

# (a) 3 GPU-minutes. Does --signal=B:USR1@N actually reach the batch shell on THIS cluster?
#     The exact targeting semantics vary by SLURM version and site config.
sbatch scripts_exp_zarr/mot_jepa/signal_probe.sbatch
#     Expect "RESULT: PASS" in logs/motjepa-signalprobe.*.log

# (b) one node, 30 minutes. Does the model train at all on GPU?
EXP_NAME=smoke1n NODES=1 TIME_LIMIT=00:30:00 CONFIG_NAME=mot_jepa_debug \
DATA_GLOB='/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-clips/*/*.zarr' \
bash scripts_exp_zarr/mot_jepa/submit.sh

# (c) two nodes. Does c10d rendezvous, and does each task really see 128 CPUs?
#     Check the log for "cpus=128"; anything less is the throughput cliff.
EXP_NAME=smoke2n NODES=2 TIME_LIMIT=00:30:00 CONFIG_NAME=mot_jepa_pilot \
DATA_GLOB='/lustre/.../ftp1-clips/*/*.zarr' \
bash scripts_exp_zarr/mot_jepa/submit.sh

# (d) the one that matters: kill it mid-run and watch the chain heal.
scancel --signal=USR1 <jobid>
#     Expect, in order: "SIGUSR1 received", "preemption requested at step N;
#     checkpointing", "requeueing job", then "resumed from step N" in the same log file,
#     and one continuous W&B run rather than two.
```

## Production launch

```bash
EXP_NAME=pilot01 NODES=4 CONFIG_NAME=mot_jepa_pilot \
STAGE_SOURCE=/lustre/fsw/portfolios/edgeai/users/chrislin/ftp1-clips/RDP \
bash scripts_exp_zarr/mot_jepa/submit.sh
```

`STAGE_SOURCE` copies the derived store to node-local `/raid` once per node and points the
trainer at the copy. Prefer it over `DATA_GLOB` for multi-node runs: the training read
pattern is thousands of small chunk files, which is precisely what saturates a Lustre MDS
when 32 ranks do it at once.

## Why the header differs from the plain template

| Change | Reason |
|---|---|
| `--cpus-per-task=128` | With `ntasks-per-node=1` and torchrun forking 8 ranks, a 16-CPU cgroup pins 8 ranks plus ~100 dataloader workers onto a fraction of a core. It presents as "GPUs at 15% util", not as an error. |
| `export SRUN_CPUS_PER_TASK` | SLURM ≥ 22.05 stopped propagating `--cpus-per-task` to `srun`. Without this you get the cliff anyway, header notwithstanding. |
| `--mem=0` | Claims all 2 TB. zarr decompression plus page cache wants it; the default limit OOM-kills workers. |
| `--exclusive` | Guarantees `/raid` and the NIC are ours. |

`%j` is right across requeues — `scontrol requeue` preserves the JobID, so with
`--open-mode=append` every attempt appends to one continuous log. Do not switch to `%J`.

## How the preemption path actually works

`--signal=B:USR1@600` delivers **only to the batch shell**, and a bash trap only runs between
commands. So `srun` is backgrounded and `wait`ed: `wait` becomes the interruptible point, the
trap fires, and the script re-waits (an interrupted `wait` returns `128+signum` without the
child having exited).

The trap touches `$RUN_DIR/PREEMPT_REQUEST`. **That file is the primary mechanism**, not the
signal: `scancel --signal` reaches the job step (`bash srun_node.sh`), not torchrun's eight
worker children, and torchrun has historically not forwarded SIGUSR1. Rank 0 polls the path
every few steps and **broadcasts** the decision, so every rank agrees — a per-rank
`exists()` race would let ranks disagree and hang the next collective.

The requeue decision keys on `REQUEUE_ARMED`, **not** the exit code, so a genuine crash does
not enter an infinite loop burning 4 GPU-hours per attempt. Three things stop the chain: the
`DONE` marker, `MAX_REQUEUES`, and a non-signal failure.

## What keeps the run continuous

| Concern | Mechanism |
|---|---|
| One W&B run across ~40 jobs | `run_dir/wandb_id.txt` + `resume="must"`, written before the first checkpoint |
| SwanLab silently breaking that | `USE_SWANLAB=false` **and** a hard assert in the trainer — the shim pops `id`/`resume` |
| Hyperparameters pinned | `run_config.json`, written once with `O_CREAT\|O_EXCL`, re-read every launch; drift is logged, the frozen value wins |
| Same data order after resume | The sampler is a pure function of `global_step` |
| LR and EMA decay after resume | Both are stateless closures over `global_step`; nothing extra is stored |
| No internet on compute nodes | `WANDB_MODE=offline`, then `sync_wandb.sh <run_dir>` from a login node |

## Files

```
submit.sh                 creates the run dir and logs/, then sbatch
mot_jepa_pretrain.sbatch  header, USR1 trap, backgrounded srun, requeue decision
srun_node.sh              per-node: stage, rendezvous, exec torchrun
stage_to_raid.sh          sourced (it exports DATA_GLOB); sentinel makes node reuse free
signal_probe.sbatch       smoke ladder step (a)
sync_wandb.sh             offline -> online from a login node
```
