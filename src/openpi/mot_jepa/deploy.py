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

    Deliberately the RIDGE and not a trained head. On 699 held-out UniVTAC clips the ridge scores
    0.00329 held-out RMSE against a per-domain constant's 0.00435, winning on all eight domains,
    while every trained head lands 0.00454-0.00484 -- worse than the constant, on every domain.
    Deploying the head would be deploying the weaker artefact.

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
        self, backbone, head, layout, domain_id: int, device, *, num_frames: int = 16, frame_stride: int = 2
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
        self.frame_stride = int(frame_stride)
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        # Attributes eval_ftp1.py reads off the policy object.
        self.action_rep = "relative"
        self.action_dim = ACTION_DIM
        self.mapping = None
        self._chunk_history: list = []
        self._last_debug: dict | None = None
        self.reset()

    def reset(self) -> None:
        self._frames: list = []
        self._chunk_history = []
        self._last_debug = None

    def set_task(self, task_name: str) -> None:
        # Accepted and ignored. MoT-JEPA is not instruction-conditioned -- FTP-1's
        # sub_task_instruction is episode-constant and paraphrased per store, so it carries no
        # signal -- and _get_task_instruction is a bare dict index that KeyErrors on unknown tasks.
        self._task = task_name

    def get_last_action_debug(self) -> dict | None:
        return self._last_debug

    @staticmethod
    def _resize(image: np.ndarray, size: int) -> np.ndarray:
        import cv2

        return cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA)

    def observe(self, observation: dict) -> None:
        """Push one frame into the ring buffer, in the layout the encoder trained on."""
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
        video = np.stack([f["video"] for f in window])           # (T, H, W, 3)
        gel = np.stack([f["gel"] for f in window])               # (T, N, h, w, 3)

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
                1, self.num_frames, self.layout.lowdim_slots, self.layout.lowdim_channels,
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
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
            encoded = self.backbone.encode_full(self._clip_inputs())
        encoded = type(encoded)(
            tokens=[t.float() for t in encoded.tokens],
            sync_readout=[t.float() for t in encoded.sync_readout],
            final_readout=[t.float() for t in encoded.final_readout],
        )
        domain = torch.tensor([self.domain_id], device=self.device)
        action_mask = torch.ones(1, ACTION_DIM, device=self.device)
        chunk = self.head.sample(encoded, action_mask, domain).float().cpu().numpy()[0]

        # Detach BEFORE np.asarray, not after: the observation carries a CUDA tensor and
        # np.asarray() on one raises rather than converting. Mirrors eval_ftp1._get_qpos8, which
        # tests for a tensor first.
        qpos8 = observation["embodiment"]["joint"][:8]
        if hasattr(qpos8, "detach"):
            qpos8 = qpos8.detach().cpu().numpy()
        qpos8 = np.asarray(qpos8, dtype=np.float32).reshape(-1)
        targets = chunk_to_absolute_qpos8(chunk, qpos8)

        self._chunk_history.append((chunk, len(self._chunk_history), qpos8.copy()))
        self._last_debug = {
            "arm_delta_step0": chunk[0, ARM_JOINT_SLICE].tolist(),
            "gripper_step0": float(chunk[0, GRIPPER_SLOT]),
            "target_step0": targets[0].tolist(),
        }
        # FIRST target only: re-plan every step. See the class docstring for the drift measurement
        # that makes this mandatory rather than stylistic.
        return targets[0].astype(np.float32)


def _to_uint8(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x * (255.0 if x.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return x
