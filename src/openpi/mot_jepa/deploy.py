"""Turning a MoT-JEPA action chunk into the absolute qpos targets UniVTAC executes.

This is the one conversion in the deployment path that fails *silently*. Everything else -- a
wrong image size, a missing module, a bad checkpoint -- raises. This does not: feed per-step
deltas through FTP-1's chunk-relative resolver and the commanded target simply never advances
past one step's motion, so the robot under-actuates, the episode fails, and the number looks like
a bad policy rather than a bad unit conversion.

The two conventions, measured from the code rather than assumed:

**MoT-JEPA** (:func:`openpi.mot_jepa.action_parse.states_to_actions_from_mask`) emits
``state[i+1] - state[i]`` -- a per-step FIRST DIFFERENCE -- for every live slot except the
columns in ``ABSOLUTE_COLUMNS`` (44 and 92, the grippers), which carry ``state[i+1]`` directly.

**FTP-1** (``UniVTAC/scripts/eval_ftp1.py:_resolve_univtac_abs_action_from_ftp1``) adds a SINGLE
``qpos8_base`` to every index of the chunk, because its actions are chunk-relative: each entry is
already an offset from the chunk's start, not from its predecessor.

So the arm needs a cumulative sum before re-basing, and the gripper needs neither -- it is already
absolute. That asymmetry is the part worth being careful about: applying the cumsum uniformly
would corrupt the gripper just as surely as omitting it corrupts the arm, and both produce
plausible-looking trajectories.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import torch

from openpi.mot_jepa.action_parse import ABSOLUTE_COLUMNS
from openpi.mot_jepa.action_parse import ACTION_DIM
from openpi.mot_jepa.action_parse import SLICES
from openpi.mot_jepa.model import ClipInputs

#: UniVTAC executes 8 numbers: 7 arm joints then 1 gripper (``eval_ftp1.py:_get_qpos8``).
QPOS8_DIM = 8

#: Where those 8 live in the 120-slot FTP-1 layout. Read from SLICES rather than written as
#: literals so a layout change breaks the import instead of silently shifting the mapping.
ARM_JOINT_SLICE = slice(*SLICES["right-arm-joints"])
GRIPPER_SLOT = ABSOLUTE_COLUMNS[0]


def univtac_action_mask(*, device: torch.device | None = None) -> torch.Tensor:
    """The exact 120-slot action mask used by every current UniVTAC training store."""
    mask = torch.zeros(ACTION_DIM, dtype=torch.float32, device=device)
    mask[ARM_JOINT_SLICE] = 1.0
    mask[GRIPPER_SLOT] = 1.0
    return mask


@dataclasses.dataclass(frozen=True)
class TrainedPolicyArtifacts:
    """A completed supervised head run, loaded without reconstructing live defaults."""

    config: object
    backbone: torch.nn.Module
    head: torch.nn.Module
    normalizer: torch.nn.Module
    domain_names: tuple[str, ...]
    observation_stride: int
    action_stride: int
    head_step: int
    # ``backbone_step`` remains the source EMA step for compatibility with existing deployment
    # logs. The remaining fields say whether and where that source was subsequently adapted.
    backbone_step: int
    backbone_train_mode: str
    backbone_checkpoint: pathlib.Path | None
    adapted_backbone_step: int | None


def load_trained_policy_artifacts(
    run: str | pathlib.Path,
    device: torch.device,
    *,
    step: int | None = None,
) -> TrainedPolicyArtifacts:
    """Strictly load a final head, normalizer, and its exact policy-time backbone.

    Deployment must be driven by the run's saved contract. In particular, a head loaded with a
    live preset can have the right tensor count while silently using the wrong cadence, horizon,
    domain rows, or backbone.
    """
    from openpi.mot_jepa import config as config_module  # noqa: PLC0415
    from openpi.mot_jepa import runtime  # noqa: PLC0415
    from openpi.mot_jepa.action_dit import ActionDiT  # noqa: PLC0415
    from openpi.mot_jepa.action_dit import ActionNormalizer  # noqa: PLC0415
    from openpi.mot_jepa.action_dit import LinearHead  # noqa: PLC0415

    run = pathlib.Path(run)
    config_path = run / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} missing; refuse to reconstruct a trained run")
    config_text = config_path.read_text()
    cfg = config_module.PolicyConfig.from_json(config_text)

    done_path = run / "DONE"
    latest_path = run / "checkpoints" / "latest"
    if not done_path.exists() or not latest_path.exists():
        raise RuntimeError(f"{run} is incomplete: require both DONE and checkpoints/latest")
    done_step = int(done_path.read_text().strip())
    latest_step = int(latest_path.read_text().strip())
    if done_step != cfg.num_train_steps or latest_step != done_step:
        raise ValueError(
            f"completion mismatch: DONE={done_step}, latest={latest_step}, configured={cfg.num_train_steps}"
        )
    if step is not None and step != done_step:
        raise ValueError(f"requested head step {step} is not the completed step {done_step}")
    step = done_step

    checkpoint = run / "checkpoints" / str(step)
    required = ("student.pt", "loss.pt", "metadata.pt", "train_config.json")
    missing = [name for name in required if not (checkpoint / name).exists()]
    if missing:
        raise FileNotFoundError(f"{checkpoint} lacks final artifacts {missing}")
    checkpoint_cfg = json.loads((checkpoint / "train_config.json").read_text())
    if checkpoint_cfg != json.loads(config_text):
        raise ValueError("final checkpoint train_config.json differs from authoritative run_config.json")
    metadata = torch.load(checkpoint / "metadata.pt", map_location="cpu", weights_only=True)
    if int(metadata.get("global_step", -1)) != step:
        raise ValueError(f"checkpoint metadata global_step={metadata.get('global_step')} != {step}")

    stats_path = run / "action_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} missing")
    with np.load(stats_path, allow_pickle=False) as stats:
        required_stats = {
            "mean",
            "scale",
            "domains",
            "observation_strides",
            "action_stride",
            "horizon",
        }
        missing_stats = sorted(required_stats - set(stats.files))
        if missing_stats:
            raise ValueError(f"{stats_path} lacks {missing_stats}")
        mean = np.asarray(stats["mean"], dtype=np.float32)
        scale = np.asarray(stats["scale"], dtype=np.float32)
        domain_names = tuple(str(name) for name in stats["domains"])
        observation_strides = tuple(int(value) for value in np.asarray(stats["observation_strides"]).reshape(-1))
        action_stride = int(np.asarray(stats["action_stride"]).item())
        horizon = int(np.asarray(stats["horizon"]).item())

    if not domain_names or len(set(domain_names)) != len(domain_names):
        raise ValueError(f"invalid or duplicate domain mapping {domain_names}")
    if mean.shape != (len(domain_names), ACTION_DIM) or scale.shape != mean.shape:
        raise ValueError(f"invalid action statistics shapes mean={mean.shape}, scale={scale.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("action statistics must be finite with positive scales")
    if observation_strides != tuple(cfg.data.strides):
        raise ValueError(f"action stats strides {observation_strides} != config {tuple(cfg.data.strides)}")
    if len(observation_strides) != 1:
        raise ValueError(f"closed-loop deployment requires one observation stride, got {observation_strides}")
    if action_stride != cfg.data.action_stride or action_stride != 1:
        raise ValueError(
            f"deployment requires action stride 1, got stats={action_stride}, config={cfg.data.action_stride}"
        )
    if horizon != cfg.head.horizon:
        raise ValueError(f"action stats horizon {horizon} != config {cfg.head.horizon}")
    if not cfg.pretrained_run or cfg.pretrained_step is None:
        raise ValueError("saved policy config does not pin a pretrained run and step")

    head_cls = LinearHead if cfg.head.objective == "linear" else ActionDiT
    head = head_cls(cfg.head, cfg.layout, num_domains=len(domain_names)).to(device)
    head_state = torch.load(checkpoint / "student.pt", map_location=device, weights_only=True)
    head.load_state_dict(head_state, strict=True)
    head.eval()

    normalizer = ActionNormalizer(len(domain_names)).to(device)
    normalizer_state = torch.load(checkpoint / "loss.pt", map_location=device, weights_only=True)
    normalizer.load_state_dict(normalizer_state, strict=True)
    normalizer.eval()
    if not torch.equal(normalizer.mean.detach().cpu(), torch.from_numpy(mean)):
        raise ValueError("checkpoint normalizer mean differs from action_stats.npz")
    if not torch.equal(normalizer.scale.detach().cpu(), torch.from_numpy(scale)):
        raise ValueError("checkpoint normalizer scale differs from action_stats.npz")

    loaded_backbone = runtime.load_policy_backbone(run, step, cfg, device)
    if loaded_backbone.source_step != cfg.pretrained_step:
        raise ValueError(
            f"loaded source backbone step {loaded_backbone.source_step} != configured {cfg.pretrained_step}"
        )
    return TrainedPolicyArtifacts(
        config=cfg,
        backbone=loaded_backbone.backbone,
        head=head,
        normalizer=normalizer,
        domain_names=domain_names,
        observation_stride=observation_strides[0],
        action_stride=action_stride,
        head_step=step,
        backbone_step=loaded_backbone.source_step,
        backbone_train_mode=loaded_backbone.mode,
        backbone_checkpoint=loaded_backbone.checkpoint,
        adapted_backbone_step=loaded_backbone.adapted_step,
    )


def chunk_to_absolute_qpos8(chunk: np.ndarray, qpos8_base: np.ndarray) -> np.ndarray:
    """``(H, 120)`` MoT-JEPA chunk + current ``(8,)`` qpos -> ``(H, 8)`` absolute targets.

    The arm is integrated then re-based; the gripper is passed through. Returns one target per
    horizon step, so the caller can execute the chunk open-loop or re-plan at any index.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    base = np.asarray(qpos8_base, dtype=np.float32).reshape(-1)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"chunk must be (H, {ACTION_DIM}), got {chunk.shape}")
    if base.shape[0] != QPOS8_DIM:
        raise ValueError(f"qpos8_base must be ({QPOS8_DIM},), got {base.shape}")

    out = np.empty((chunk.shape[0], QPOS8_DIM), dtype=np.float32)
    # Integrate: target at step k is the base plus every delta up to and including k. Without the
    # cumsum each target would be base + one step, i.e. the arm would stop after the first move.
    out[:, :7] = base[:7] + np.cumsum(chunk[:, ARM_JOINT_SLICE], axis=0)
    # Already absolute -- ABSOLUTE_COLUMNS exists precisely so the gripper is not differenced.
    out[:, 7] = chunk[:, GRIPPER_SLOT]
    return out


class MotJepaRidgePolicy:
    """The deployable policy: frozen MoT-JEPA encoder + fitted per-domain ridge, wearing the
    interface ``UniVTAC/scripts/eval_ftp1.py`` calls.

    The ridge remains a useful deterministic control: on 699 held-out UniVTAC clips the corrected
    100k ridge scores 0.00330 held-out RMSE against a per-domain constant's 0.00435. The completed
    supervised flow head is deployed by :class:`MotJepaFlowPolicy` below.

    **Re-plans every step.** ``act`` returns only the FIRST target of the predicted chunk.
    Integrated open-loop drift measured on the same holdout is 0.00231 rad at k=1 but 0.101 rad at
    k=32 -- about 61 mm at 0.6 m, and 89% of the way from the independent-error bound to the fully
    correlated one. Executing a whole chunk open-loop is hopeless; re-planning each step is what
    keeps the error at 1.4 mm. eval_ftp1.py:591 already re-infers every step, so this costs
    nothing extra.

    The claim this supports is narrow: how much of a fine-tuned expert's success a FROZEN encoder
    plus a CLOSED-FORM head recovers. Not that it beats FTP-1.
    """

    def __init__(
        self,
        backbone,
        head,
        layout,
        domain_id: int,
        device,
        *,
        num_frames: int = 16,
        frame_stride: int | None = None,
        observation_stride: int | None = None,
        action_stride: int | None = None,
        domain_names: tuple[str, ...] | None = None,
        save_infer_input_dir: str | pathlib.Path | None = None,
    ):
        self.backbone = backbone
        self.head = head
        self.layout = layout
        self.domain_id = int(domain_id)
        self.device = device
        self.num_frames = num_frames
        # COUPLED to DataConfig.strides, which is (2, 4). The encoder must be handed frames at a
        # rate the training set brackets: RoPE `t` is the tubelet index and carries no rate, so an
        # encoder fed a different frame rate cannot tell -- it just sees a slower or faster world
        # than it trained on, with no error anywhere. Buffering every control step (stride 1) while
        # training on (2, 4) would put deployment a factor of two outside the trained range.
        artifact_observation_stride = (
            int(observation_stride) if observation_stride is not None else getattr(head, "observation_stride", None)
        )
        artifact_action_stride = (
            int(action_stride) if action_stride is not None else getattr(head, "action_stride", None)
        )
        if artifact_observation_stride is None or artifact_action_stride is None:
            raise ValueError("policy head has no sampling metadata; refuse to guess deployment cadence")
        if artifact_action_stride != 1:
            raise ValueError(
                f"policy action_stride is {artifact_action_stride}, but this harness executes every control step"
            )
        if frame_stride is None:
            frame_stride = artifact_observation_stride
        if frame_stride != artifact_observation_stride:
            raise ValueError(
                f"frame_stride {frame_stride} disagrees with fitted observation stride {artifact_observation_stride}"
            )
        self.frame_stride = int(frame_stride)
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        # Attributes eval_ftp1.py reads off the policy object.
        # ``act`` returns an already-resolved 8-D absolute target. The raw 120-D head chunk is
        # mixed (arm first differences, absolute gripper), which is tracked separately for debug.
        self.action_rep = "absolute"
        self.chunk_action_rep = "mix"
        self.first_executable_index = 0
        self.action_dim = ACTION_DIM
        self.mapping = None
        self.domain_names = tuple(domain_names) if domain_names is not None else None
        if self.domain_names is not None and not (0 <= self.domain_id < len(self.domain_names)):
            raise ValueError(f"domain_id {self.domain_id} outside domain mapping {self.domain_names}")
        self.save_infer_input_dir = pathlib.Path(save_infer_input_dir) if save_infer_input_dir else None
        self._saved_infer_input = False
        self._chunk_history: list = []
        self._last_debug: dict | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._frames: list = []
        self._chunk_history = []
        self._last_debug = None
        self._episode_seed: int | None = None

    def reset(self, seed: int | None = None) -> None:
        self._reset_state()
        self._episode_seed = seed

    def set_task(self, task_name: str) -> None:
        # Accepted and ignored. MoT-JEPA is not instruction-conditioned -- FTP-1's
        # sub_task_instruction is episode-constant and paraphrased per store, so it carries no
        # signal -- and _get_task_instruction is a bare dict index that KeyErrors on unknown tasks.
        self._task = task_name
        if self.domain_names is not None:
            canonical = str(task_name).replace("UniVTAC_", "")
            if canonical not in self.domain_names:
                raise KeyError(f"task {task_name!r} is absent from trained domains {self.domain_names}")
            self.domain_id = self.domain_names.index(canonical)

    def get_last_action_debug(self) -> dict | None:
        return self._last_debug

    @staticmethod
    def _resize(image: np.ndarray, size: int) -> np.ndarray:
        import cv2  # noqa: PLC0415

        return cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA)

    def observe(self, observation: dict) -> None:
        """Push one frame into the ring buffer, in the layout the encoder trained on."""
        # Preserve UniVTAC's native numeric channel order. The official writer passes simulator
        # arrays directly to cv2.imencode; its parser does BGR->RGB and then reverses once while
        # building Zarr, so those two parser conversions cancel. Byte-level checks against the
        # source HDF5, source Zarr, and derived clips confirm that deployment must not reverse here.
        head = _to_uint8(observation["observation"]["head"]["rgb"])
        tac = observation.get("tactile", {})
        left_key = "left_gsmini" if "left_gsmini" in tac else "left_tactile"
        right_key = "right_gsmini" if "right_gsmini" in tac else "right_tactile"
        pads = [
            self._resize(_to_uint8(tac[left_key]["rgb_marker"]), self.layout.gel_size),
            self._resize(_to_uint8(tac[right_key]["rgb_marker"]), self.layout.gel_size),
        ]
        self._frames.append(
            {
                "video": self._resize(head, self.layout.video_size),
                "gel": np.stack(pads, axis=0),
            }
        )
        # Keep enough CONTROL steps to subsample `num_frames` at `frame_stride`. The policy still
        # acts every control step; only the frames handed to the encoder are decimated.
        if len(self._frames) > (self.num_frames - 1) * self.frame_stride + 1:
            self._frames.pop(0)

    def _clip_inputs(self):
        if not self._frames:
            raise RuntimeError("observe() must be called before act()")
        # Subsample at the trained stride, newest-last. `[::-1][::stride][::-1]` anchors the
        # decimation to the MOST RECENT frame: the policy's action depends on where the motion is
        # now, so the newest observation must always be in the window regardless of buffer length.
        frames = self._frames[::-1][:: self.frame_stride][::-1]
        # Cold start. Training REJECTS short windows rather than padding (clip_dataset.py:9-13),
        # so edge-padding here is a deliberate deviation, recorded as one: until the buffer fills,
        # the oldest frame is repeated rather than the episode skipped.
        pad = [frames[0]] * (self.num_frames - len(frames))
        window = (pad + frames)[-self.num_frames :]
        video = np.stack([f["video"] for f in window])  # (T, H, W, 3)
        gel = np.stack([f["gel"] for f in window])  # (T, N, h, w, 3)

        def prep(arr, axes):
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
            return t.permute(*axes).float().div_(127.5).sub_(1.0).unsqueeze(0)

        return ClipInputs(
            video=prep(video, (0, 3, 1, 2)),
            gel=prep(gel, (0, 1, 4, 2, 3)),
            # UniVTAC populates no lowdim stream, and ~79% of the pretraining corpus zero-fills it
            # too, so zeros are what the encoder saw for most of training rather than a stand-in.
            # Shape must be the layout's (B, T, lowdim_slots, lowdim_channels): the embedder
            # asserts the slot count, so a (1, T, 1, 1) placeholder fails with "expected 8 slots,
            # got 1" rather than being broadcast.
            lowdim=torch.zeros(
                1,
                self.num_frames,
                self.layout.lowdim_slots,
                self.layout.lowdim_channels,
                device=self.device,
            ),
        )

    @torch.no_grad()
    def act(self, observation: dict, prompt: str | None = None) -> np.ndarray:
        del prompt  # accepted and ignored; see set_task
        self.observe(observation)
        # bf16 autocast, as the trainer and the offline evaluator both run this encoder. Deploying
        # it in fp32 was not just slower: Isaac's renderer shares this GPU and died with
        # VkResult ERROR_DEVICE_LOST / "Failure to upload Texture" after 60 s semaphore waits,
        # i.e. the compute was blocking texture uploads. It also keeps the closed-loop numerics
        # identical to the offline RMSE the policy was selected on.
        inputs = self._clip_inputs()
        self._save_inference_input_once(inputs, observation)
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
            encoded = self.backbone.encode_full(inputs)
        encoded = type(encoded)(
            tokens=[t.float() for t in encoded.tokens],
            sync_readout=[t.float() for t in encoded.sync_readout],
            final_readout=[t.float() for t in encoded.final_readout],
        )
        chunk = self._sample_chunk(encoded)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or chunk.shape[0] < 1:
            raise ValueError(f"policy produced invalid chunk shape {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise FloatingPointError("policy produced a non-finite action chunk")

        # Detach BEFORE np.asarray, not after: the observation carries a CUDA tensor and
        # np.asarray() on one raises rather than converting. Mirrors eval_ftp1._get_qpos8, which
        # tests for a tensor first.
        qpos8 = observation["embodiment"]["joint"][:8]
        if hasattr(qpos8, "detach"):
            qpos8 = qpos8.detach().cpu().numpy()
        qpos8 = np.asarray(qpos8, dtype=np.float32).reshape(-1)
        targets = chunk_to_absolute_qpos8(chunk, qpos8)
        if not np.isfinite(targets).all():
            raise FloatingPointError("policy produced a non-finite absolute target")

        self._chunk_history.append((chunk, len(self._chunk_history), qpos8.copy()))
        self._last_debug = {
            "arm_delta_step0": chunk[0, ARM_JOINT_SLICE].tolist(),
            "gripper_step0": float(chunk[0, GRIPPER_SLOT]),
            "target_step0": targets[0].tolist(),
        }
        # FIRST target only: re-plan every step. See the class docstring for the drift measurement
        # that makes this mandatory rather than stylistic.
        return targets[0].astype(np.float32)

    def _sample_chunk(self, encoded) -> np.ndarray:
        """Ridge heads emit raw action units directly."""
        domain = torch.tensor([self.domain_id], device=self.device)
        action_mask = torch.ones(1, ACTION_DIM, device=self.device)
        return self.head.sample(encoded, action_mask, domain).float().cpu().numpy()[0]

    def _save_inference_input_once(self, inputs: ClipInputs, observation: dict) -> None:
        if self.save_infer_input_dir is None or self._saved_infer_input:
            return
        output = self.save_infer_input_dir
        output.mkdir(parents=True, exist_ok=True)
        video = inputs.video.detach().cpu().numpy()
        gel = inputs.gel.detach().cpu().numpy()
        lowdim = inputs.lowdim.detach().cpu().numpy()
        qpos8 = observation["embodiment"]["joint"][:8]
        if hasattr(qpos8, "detach"):
            qpos8 = qpos8.detach().cpu().numpy()
        np.savez_compressed(
            output / "motjepa_input.npz",
            video=video,
            gel=gel,
            lowdim=lowdim,
            qpos8=np.asarray(qpos8, dtype=np.float32),
            episode_seed=np.asarray(-1 if self._episode_seed is None else self._episode_seed, dtype=np.int64),
        )
        metadata = {
            "video_shape": list(video.shape),
            "gel_shape": list(gel.shape),
            "lowdim_shape": list(lowdim.shape),
            "channel_order": "univtac_native_cv2_decoded",
            "tactile_pad_order": ["left/thumb", "right/index"],
            "observation_stride": self.frame_stride,
            "domain_id": self.domain_id,
            "domain": self.domain_names[self.domain_id] if self.domain_names is not None else None,
            "episode_seed": self._episode_seed,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        self._saved_infer_input = True
        print(f"[motjepa.deploy] saved first model input to {output}", flush=True)


class MotJepaFlowPolicy(MotJepaRidgePolicy):
    """Frozen MoT-JEPA encoder plus a supervised, normalized generative action head."""

    def __init__(
        self,
        backbone,
        head,
        normalizer,
        layout,
        domain_id: int,
        device,
        *,
        observation_stride: int,
        action_stride: int,
        domain_names: tuple[str, ...],
        num_inference_steps: int = 10,
        num_samples: int = 1,
        sample_seed: int = 0,
        save_infer_input_dir: str | pathlib.Path | None = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.normalizer = normalizer
        self.num_inference_steps = int(num_inference_steps)
        self.num_samples = int(num_samples)
        self.sample_seed = int(sample_seed)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(self.sample_seed)
        self._action_mask = univtac_action_mask(device=device)
        super().__init__(
            backbone,
            head,
            layout,
            domain_id,
            device,
            observation_stride=observation_stride,
            action_stride=action_stride,
            domain_names=domain_names,
            save_infer_input_dir=save_infer_input_dir,
        )

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        episode_seed = 0 if seed is None else int(seed)
        self._generator.manual_seed((self.sample_seed + episode_seed) % (2**63 - 1))

    def _sample_chunk(self, encoded) -> np.ndarray:
        domain = torch.tensor([self.domain_id], device=self.device)
        action_mask = self._action_mask.unsqueeze(0)
        samples = [
            self.head.sample(
                encoded,
                action_mask,
                domain,
                num_steps=self.num_inference_steps,
                generator=self._generator,
            ).float()
            for _ in range(self.num_samples)
        ]
        normalized = torch.stack(samples).mean(dim=0)
        raw = self.normalizer.denormalize(normalized, domain) * action_mask.unsqueeze(1)
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("flow head produced a non-finite denormalized action")
        return raw.cpu().numpy()[0]


def _to_uint8(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x * (255.0 if x.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return x
