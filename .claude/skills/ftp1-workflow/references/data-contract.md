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

## Reference representations

Use the following together for the released UniVTAC recipe:

- `proprioception_pose_rep=relative`
- `action_pose_rep=relative`
- `proprioception_joint_rep=abs`
- `action_joint_rep=mix`
- `disable_history=true`, so online state and tactile history use `T=1`

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
