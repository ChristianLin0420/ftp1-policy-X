# FTP-1 Data and Checkpoint Contract

## Zarr layout

Place one or more immediate `*.zarr` directories under each configured domain path. Each store contains `data/` and `meta/episode_ends`. Every array under `data/` is time-major and its first dimension equals the final episode end.

Require `timestamps`, `sub_task_instruction`, at least one supported RGB key, and state trajectories from which future actions can be derived. Supported RGB keys are `camera_main_rgb`, `camera_ego_rgb`, `right_wrist_camera_rgb`, and `left_wrist_camera_rgb`.

Pair each tactile group exactly:

```text
<side>_tactile_data_<group>:   (T, N, ...)
<side>_tactile_area_<group>:   (T, N)
<side>_tactile_sensor_<group>: (T,)
<side>_tactile_type_<group>:   (T,)
```

Use tactile type `image`, `binary`, or `state`. Image tactile input has shape `(T, N, H, W, 3)`. UniVTAC uses two GelSight Mini pads grouped as `right_tactile_*_gripper`, areas `[0, 1]`, with the thumb pad first.

Pair `left_hand_joints` and `right_hand_joints` with matching `*_hand_joints_idx`. FTP-1 canonical hand slots are `[0, 31]`; UniVTAC maps its scalar gripper to slot 28. Raw Zarr does not need a standalone action array because the loader derives targets from future state trajectories.

The published `lift_bottle` head-camera conversion qualified at revision `fb235e7` contains 100 episodes, 30,432 steps, and these ten arrays: `timestamps`, `camera_ego_rgb`, `right_arm_joints`, `right_hand_joints`, `right_hand_joints_idx`, `right_tactile_data_gripper`, `right_tactile_area_gripper`, `right_tactile_sensor_gripper`, `right_tactile_type_gripper`, and `sub_task_instruction`. Require every array's first dimension to equal the final episode end, all episode lengths to be positive, state values to be finite, hand index 28, areas `[0, 1]`, sensor `GelSightMini`, and tactile type `image`.

Preserve UniVTAC's native decoded image order. The official writer passes simulator arrays directly to `cv2.imencode`, and its loader returns `cv2.imdecode` output directly. The converter's two channel reversals cancel; removing only one changes the dataset contract. If simplifying the code, remove both reversals together and lock the behavior with a synthetic HDF5 regression.

## Reference representations

Use the following together for the released UniVTAC recipe:

- `proprioception_pose_rep=relative`
- `action_pose_rep=relative`
- `proprioception_joint_rep=abs`
- `action_joint_rep=mix`
- `disable_history=true`, so online state and tactile history use `T=1`
- `norm_image_tactile_mode=channel_wise`
- seed-42 deterministic 90/10 split by source episode, never timestep

The current unified action layout is 120 dimensions: two 48-D arm blocks, a 9-D ego/head block, and 15 reserved dimensions. Each arm block contains 9-D wrist pose, 7 arm joints, and 32 hand slots.

## Deployable checkpoint

Require a step directory containing:

```text
model.safetensors
model_config.json
train_config.json
hpt_tokenizer/       # when tactile input is enabled
normalization/
```

The deployment `domain_name` must resolve under `normalization/`. Keep this step directory intact; copying only model weights loses the tactile tokenizer and denormalization contract.

## Measured properties of the released corpus

Verified on the staged `RDP` and `RDP_Bimanual` domains (190 episodes, 86,228 frames) with
`scripts/mot_jepa_survey_corpus.py`. Re-run that tool on new domains rather than assuming
these hold.

**`timestamps` is unusable and must not drive sampling.** It stores absolute Unix epoch
seconds in **float32**. At ~1.74e9 the float32 ulp is 128 s, so every frame in a clip rounds
to the same value and 100% of within-episode inter-frame deltas are exactly 0 or 128.
Deriving a rate from it yields ~0.008 Hz and inflates an 0.8-hour corpus into 3,000 hours.
Index clips by frame, never by time. This is consistent with `disable_history=true` being the
default. Any hour-based corpus figure quoted from this release is unverified.

**A tactile type label is not evidence that a sensor was recording.** In
`RDP_Bimanual/lift_v2`, `left_tactile_data_gripper_gelsightmini` and
`right_tactile_data_gripper_mctac` are identically zero (mean 0, std 0 across space and
time) while their per-hand partners are live — so each hand has exactly one working gel
sensor, and which one differs by hand. Both dead streams are still labelled type `image`.
Feeding a constant stream to a masked-prediction objective is worse than dropping it: the
target is trivially predictable, so reconstruction modes score well while learning nothing,
and it contributes no discriminative signal to a contrastive term. Check content, not just
key presence (`openpi.mot_jepa.clip_dataset.is_degenerate`).

**Shapes and chunking.** RGB and gel are both stored at 224x224x3 uint8, chunked at **14
frames** along time (~2 MB) with **Blosc lz4, NOSHUFFLE** — not the zstd-bitshuffle setting
the parse scripts use elsewhere. A 16-frame clip therefore straddles two chunks and
decompresses ~28 frames to use 16. Measured cost of one 16-frame clip from the source store
is ~682 ms: ~360 ms of zarr reads plus ~320 ms of on-the-fly resize. Decompressed size is
3.61 MB/clip.

**Low-dimensional tactile** appears as `*_tactile_data_gripperforce_flexivgripper`, shape
`(T, 1, 1)`, type `state`.

## MoT-Control V3 authoritative UniVTAC contract

Production V3 data must contain exactly 1,000 successful `lift_bottle` episodes collected by four
independent one-job/one-GPU scripted-expert shards. A multi-worker or multi-task Pyxis step is not
qualified for collection on this cluster. Shard rank `r` owns the unbounded seed stream
`2,000,000 + r + 4k`, so failed attempts cannot overlap another shard. Each shard must record its
job/node/GPU UUID/HTTP port, reach its one-worker startup and participation gates, exit zero, and
commit exactly 250 successful episodes. The CPU merge requires ranks 0--3 exactly once, distinct
jobs and ports, verified source/runtime/sidecar hashes, no duplicate or wrong-residue seed, and an
aggregate of exactly 1,000 successful episodes. Every raw HDF5 episode must contain at least
63 aligned rows, yielding at least 62 parser rows and one T16/stride-2 plus H32/stride-1 sample.
The exact HDF5 file set, bytes, SHA-256 values, and parser hardlink identities are committed and
reverified before and after conversion. Preparation uses an episode-level seeded train/validation
split and writes a full content manifest covering metadata and every Zarr chunk. Preparation must
reject both underfull and overfull collections before conversion and run the parser without an
episode cap; a cap that silently selects the first 1,000 files is not an exact-set validation.
Production verification requires integer schema version 3 and canonical, direct in-root members;
legacy schema fallback, path traversal, symlink substitution, changed source/runtime content IDs,
or a parser file that is not the raw HDF5 inode must fail closed.

Each time-major `data/` group additionally requires:

```text
command8:           (T, 8) float32
command_valid:      (T,)   uint8/bool
phase_id:           (T,)   integer in {0,1,2,3}
contact:            (T,)   finite binary value
contact_valid:      (T,)   uint8/bool
control_step:       (T,)   consecutive integer control index
control_step_valid: (T,)   uint8/bool
```

Each prepared store's `meta/` group additionally requires `source_episode_seed: (E,) int64`.
The values must be globally unique, remain in the exact episode order committed by the collection
manifest, and equal the numeric raw-HDF5 stems. The parser and clip builder must copy this identity
without deriving it from a prepared-store path. The production split is
`sha256(f"{split_seed}\\0episode_seed={source_episode_seed}")`; the first unsigned 64-bit word,
divided by `2**64`, selects validation when it is below the held-out fraction. Store relocation or
a fresh job-owned preparation must therefore leave the episode split unchanged.

All validity values must be true for production. Phase IDs are fixed to `approach`, `close`,
`lift`, and `release`, and both contact classes and every phase must occur. `command8[t]` is the
command produced from observation row `t`. Consequently a sample whose current row is `t` targets
`command8[t+1:t+32]`; H32 row 0 is a placeholder equal to the current command and the first
executable prediction is row 1. Observation history uses 16 frames at stride 2, whereas command
targets and `control_step` advance at stride 1. Both cold-start histories that have the same count
of real observations but differ in whether the oldest command exists must be represented during
training. Build training anchors at `index_step=1`; deployment replans at every control step, so a
coarser clip-index step leaves deployment offsets out of distribution.

The eight command values are seven relative arm-joint deltas plus one absolute gripper command.
The per-history-row state conditioner is 27-D: qpos8, velocity8, previous command8, and three
presence bits for history, velocity, and command. Reject inferred future-state targets or relabeled
legacy stores for production; legacy inference exists only for compatibility tests.

Preserve native decoded numeric channels without a BGR/RGB reversal. The UniVTAC writer applies
OpenCV's default `.jpg` encoding to head and marker images. Live deployment must reproduce that
same default JPEG encode/`IMREAD_COLOR` decode loss before resizing; a raw simulator array is not
training-equivalent. Both paths then resize to 224x224 with `INTER_LINEAR`; GEL follows that first
stage and is reduced from 224x224 to 112x112 with `INTER_AREA`. Keep the two pads in the qualified
thumb/index order. Training, deployment, and acceptance must agree on artifact schema 4 and data
schema `mot_jepa_control_v2_data_v2`, `dense_video_gel`, `qpos8_history_command`,
`next_command_chunk_relative_mix8`, H32, observation / action / control-step cadence `2/1/1`, and
temporal ensemble `K=0.01` over the first 20 predictions.

Checkpoint selection uses the frozen
`all_episode_time_phase_contact_cold_start_stratified_v2` panel. Treat its configured example
count as a floor: expand it to cover every held-out episode and all 32 history scenarios (lengths
1 through 16 crossed with both oldest-command-presence parities), then stratify remaining choices
across episode time, phase, and valid contact class. Each sample's source row, strata, and history
scenario are part of its structured identity and SHA-256 closure; reject a missing stratum,
duplicate/tampered index, scenario gap, episode-count mismatch, or changed frozen manifest.
Selection is lexicographic: require zero exact joint/rate violations on the first 20 executable
rows, then minimize deployed-horizon action loss with earliest-step tie breaking. Train the safety
hinge against `qpos[t:t+20]`, the same preceding-observation reference used by the runtime limiter,
at `0.9 * max_command_delta`; do not substitute preceding commands or include unused rows 21--31
in the safety gate. Before `STATS_DONE` is committed, CPU stats must construct the exact frozen
validation panel and prove that every expert first-20 command has finite, strictly positive
rate-limit slack under the same FP32 reference semantics used by training. A non-positive or
non-finite oracle slack is a data/split failure and must stop the chain before GPU allocation.
