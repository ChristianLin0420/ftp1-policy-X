# MoT-JEPA: architecture from pretraining to action-policy post-training

A Mixture-of-Transformers video–tactile world model, pretrained self-supervised on a 527-store
robot corpus, then frozen and used as the observation encoder for a generative action policy.

All shapes below are the **pilot** preset (`LAYOUT_PILOT`), which is what every run to date used.
The `base` preset scales the same structure to 224px video and 2448 tokens.

---

## 1. Input contract

One *clip* is 16 frames sampled at stride ∈ {1, 2} from inside a single episode. Clips never
cross an episode boundary — they are rejected, never clamped.

| stream | shape | dtype | meaning |
|---|---|---|---|
| `video` | `(B, 16, 3, 112, 112)` | uint8 | third-person / wrist RGB |
| `gel` | `(B, 16, 2, 3, 64, 64)` | uint8 | 2 vision-based tactile pads |
| `lowdim` | `(B, 16, 8, 48)` | float32 | 8 non-image tactile slots × 48 channels |
| `state` | `(B, 16, 120)` | float32 | proprioception in the FTP-1 120-slot layout |
| `action_mask` | `(B, 120)` | float32 | **1 = this store populates the slot, 0 = ABSENT** |

Images are converted to `[-1, 1]` on device (`/127.5 − 1`), never in the dataloader worker.

`action_mask` semantics matter everywhere downstream: a 0 means *this embodiment has no such
joint*, not *this joint read zero*. Every loss weights by it rather than regressing the zeros.

---

## 2. Tokenisation

```
16 frames ── tubelet_t=2 ──► 8 tubelet steps   (the shared time axis)

video   112/16 = 7×7 = 49 patches/step × 8 = 392 tokens   width 384
gel      64/16 = 4×4 = 16 × 2 pads = 32/step × 8 = 256    width 192
lowdim                          8 slots/step × 8 =  64    width 192
                                                   ─────
                                             total  712 tokens
```

Two **experts**: `VIDEO` owns tokens `[0, 392)`; `TACTILE` owns `[392, 712)` (gel + lowdim).
Each expert has its own width, so the streams are never forced into a shared dimensionality
before they need to be.

**3D RoPE** with a shared time axis (`dim_t=32, dim_h=16, dim_w=16`, `theta_t=1e4`,
`theta_hw=100`). The time axis is shared across both experts, which is what lets a video token
and a gel token at the same tubelet step know they are simultaneous.

---

## 3. Pretraining

### 3.1 Model

```
   video (B,16,3,112,112)   gel (B,16,2,3,64,64)   lowdim (B,16,8,48)
            └──────────────────────┴────────────────────┘
                              ▼
                       StreamEmbed                 per-stream patchify + project
                              ▼
        ┌─────────────────────────────────────────────────┐
        │  MoTEncoder — 12 layers, 6 heads, head_dim 64    │   29.9M params
        │                                                   │
        │   layers 0-3   modality-LOCAL                     │
        │       each expert attends only within itself      │
        │       ── sync_readout snapshot taken here ──      │
        │                                                   │
        │   layers 4-11  GLOBAL                             │
        │       one shared attention over all 712 tokens    │
        │       per-expert LN / qkv / out / FFN             │
        └─────────────────────────────────────────────────┘
                              ▼
    tokens        [ (B,392,384) video , (B,320,192) tactile ]   final, LayerNorm'd
    sync_readout  [ (B,  8,384) video , (B,  8,192) tactile ]   per-step, pre-global
```

"Mixture of Transformers" = **one shared attention, per-expert everything else**. The experts
mix only through attention, and only after layer 4.

`sync_readout` is snapshotted *before* the global layers on purpose: after them each modality
already contains the other, so a cross-modal comparison on final tokens would be partly
self-matching.

### 3.2 The JEPA objective

A **predictor** (6 layers, width 192, 5.6M params) receives the encoder's output on *context*
tokens plus learned mask tokens at *target* positions, and predicts the target embeddings.
Targets come from an **EMA teacher** — an fp32 shadow of the encoder, decay `0.998 → 0.99999`
over 30k steps — run on the full unmasked clip.

Targets are LayerNormed (`normalize_targets`), which is parameter-free, so `‖z*‖ = √D` exactly
and magnitude carries zero information. Only direction is learnable.

**Mask modes**, sampled per batch:

| mode | p | what is hidden |
|---|---|---|
| `V` | 0.35 | a video window (88% of video tokens) |
| `T` | 0.20 | a tactile window |
| `T_HARD` | 0.10 | **all** tactile — predict touch from vision alone |
| `V_HARD` | 0.10 | **all** video |
| `X` | 0.25 | both, 75% of video |

`T_HARD` is why pretraining needs `find_unused_parameters=True`: the tactile encoder legitimately
receives no gradient on ~10% of steps.

### 3.3 Losses

```
L = 1.00·L_video  +  1.00·L_tactile  +  0.05·L_lowdim  +  0.20·L_sync
```

- **`L_video` / `L_tactile`** — latent prediction against the EMA target. Selectable objective:
  - `l1` (default): `|ẑ − z*|`
  - `direction`: `(1 − cos(ẑ_centred, z*)) + 0.25·Huber(‖ẑ_centred‖ − ‖z*‖)`
- **`L_lowdim`** — weighted 0.05, gated by a running variance estimate so dead channels
  contribute nothing.
- **`L_sync`** — InfoNCE matching video↔tactile, warmed in over 5000 steps, temperature 0.07,
  through two 256-d projectors. Level A (clip-level, weight 0.8) + Level B (weight 0.2).

### 3.4 Runs completed

| run | tactile objective | steps |
|---|---|---|
| `probe2` (`pilot02`) | `l1` | 50 000 |
| `probe3` | `direction` | 50 000 |

Frozen snapshots live at `ftp1-runs/backbones/{probe2,probe3}_s50000/`, holding
`teacher_ema.pt` — the **EMA teacher**, not the student.

> ⚠️ `EmaTeacher.state_dict` serialises the shadow **positionally** (`shadow.0 … shadow.253`),
> ordered by `backbone.parameters()`. `load_state_dict(..., strict=False)` matches *nothing* and
> silently leaves a random backbone. Always load via `runtime.load_frozen_backbone`, which does a
> positional copy with a hard count check.

---

## 4. What pretraining actually learned — measured

Three findings that shape everything downstream. All are direct measurements, not inferences.

**The encoder encodes state well.** A ridge readout recovers the clip's own proprioception at
**R² 0.92** pooled. State is fed in through the lowdim stream, so this is also the positive
control that validates the estimator.

**It carries real action information.** Held-out R²(future 15-step action chunk ← frozen latent)
is **0.32 / 0.35** pooled for probe2 / probe3, up to **0.81 / 0.82** on the best domain
(ViTaMIn). This is the gate that authorised the policy build.

**It is temporally order-blind.** Scrambling a clip's tactile frames in time moves the latent by
**0.003–0.007** of the between-clip spread, while the input's time-varying component moves
**1.01** — as much as substituting a completely different clip. Root cause is structural: the
sync readout is a **mean over the time axis**, so clip-pooled `L_sync` is order-invariant *by
construction*. Nothing in the loss ever rewarded temporal correspondence.

**The two tactile objectives are indistinguishable.** Three independent measurements —
displacement (0.003 vs 0.003), action readout (0.32 vs 0.35), `timeshuffle_gap` (both ≈ 0) — say
the `direction` objective changed nothing that matters.

> Both checkpoints predate the `051592a` fix, so their `L_sync` projectors reset to random init at
> every requeue (3× per run). Neither trained `L_sync` continuously for more than ~24% of its
> steps. Metrics computed *through* those projectors (retrieval, donor ratio) are contaminated;
> the three findings above are not, because none of them touch the projectors.

---

## 5. Post-training: the action policy

### 5.1 Additional data

The clip dataset gains a **future** action chunk. The existing `action` field is retrospective and
only 8 steps long, so it cannot serve a policy.

```
action_chunk  (B, 32, 120)   derived at the clip's own stride via actions_from_state
chunk_mask    (B, 32, 120)   the per-clip mask, broadcast over the horizon
```

The clip index tightens to `s + (num_frames − 1 + H)·stride < episode_end`, so a chunk can never
splice the next episode's motion onto this one's observation.

Action semantics (`action_parse.py`): first difference by default; the three 9-D pose blocks use
`relative_pose(P_t, P_{t+1}) − identity`; columns 44 and 92 (the grippers) are **absolute**.

### 5.2 Model

```
   ❄️ FROZEN MoT-JEPA backbone (no_grad, eval)  ──► sync_readout
                                                    [(B,8,384), (B,8,192)]
                                                          │
                                   Linear 384→512 ─┬─ Linear 192→512
                                                   ▼
                                        context (B,16,512)
                                          │              │
                            cross-attn memory        mean → cond (B,512)
                                          │              │
   x (B,32,120) ──×mask──► Linear 120→512 │              │  + sinusoid(t)
        + pos_embed (1,32,512)            │              ▼
                    ▼                     ▼          adaLN-Zero
   ┌────────────────────────────────────────────────────────────┐
   │  ActionDiT × 8   width 512, 8 heads      🔥 61.9M params    │
   │    self-attn (32 chunk tokens)                              │
   │    cross-attn (16 context tokens)                           │
   │    SwiGLU MLP                                               │
   │    9 modulation vectors per block, zero-init gates          │
   └────────────────────────────────────────────────────────────┘
                    ▼
         final adaLN → Linear 512→120  (zero-init)
                    ▼
              (B, 32, 120)
```

Two invariants worth stating explicitly:

- **The mask is applied before the input projection.** An absent action group must contribute
  exactly zero, not a learned bias — otherwise the head can read embodiment identity off its own
  input and appear to use the action without doing so.
- **adaLN gates and the output layer are zero-initialised**, so at step 0 the head is a
  pass-through. That matters when the conditioning comes from an encoder the head cannot correct.

### 5.3 Objective A — Drifting (Implicit Drifting Policy)

arXiv 2606.01098, which fixes a failure the plain Drifting objective (arXiv 2602.04770) has on
behaviour cloning. Plain Drifting needs several real samples per condition; a robot observation
has exactly one expert chunk, and its Proposition 3.1 shows the field then evaluates to
`V = a* − a` — the objective collapses into isotropic MSE.

IDP never materialises a field. It estimates the *local geometry* of expert actions under similar
observations and minimises an anisotropic potential directly:

```
h_i    = sg[pooled readout] / ‖·‖                       frozen, detached
w_ij   = softmax_j( standardise( h_i·h_j ) ),  j ≠ i    row-standardisation IS the temperature
v_cond = Σ_j w_ij (a*_j − a*_i)²                        per dimension
v_ref  = Var_i(a*)                                      per dimension, per minibatch
M      = ReLU( ŝ_cond / ŝ_ref − 1 )                     ŝ = (v+ε)^-½, normalised across dims

E(a)   = ½[ ‖a − a*‖²  +  (a − a*)ᵀ M (a − a*) ]        masked
L      = E(y) + 0.1·E(z)
           y = head(noise,           t = 0)             the real prediction
           z = head(a* + 0.05·ε,     t = 0.95)          expert-proximal probe
```

**`ŝ` are precisions, not variances.** This is the detail that decides whether the method works:
a dimension where neighbours disagree (multimodal) has *low* precision → ratio < 1 → ReLU gives 0
→ `(I + M)` is the identity → the weakest pull the objective applies. Extra pull is added **only**
where the observation makes a dimension tighter than the global prior. Implementing this with
variances inverts the anisotropy, collapsing valid modes toward their mean while presenting as
ordinary underfitting.

Batches are **domain-pure** for this reason: both `w_ij` and `v_ref` are computed within a batch,
and mixing embodiments would read a difference in *layout* as a difference in *action*.

**Inference: 1 forward pass.**

### 5.4 Objective B — Flow matching (control)

Mirrors `models_pytorch/ftp1_pytorch.py:435-534`.

```
t   ~ Beta(1.5, 1.0)·0.999 + 0.001
x_t = t·noise + (1 − t)·a*
u_t = noise − a*
L   = Σ (v_θ(x_t, t) − u_t)² · chunk_mask / Σ chunk_mask
```

**Inference: 10 Euler steps.**

This arm exists because the pipeline changes *two* things relative to the FTP-1 policy — the
conditioning encoder and the generative objective. Without an arm that changes only the encoder,
a bad result cannot be attributed to either.

---

## 6. Fine-tuning and evaluation on UniVTAC

1. **Convert** — `parse_data_univtac.py` on the 8 task directories (~100 episodes each) → FTP-1
   zarr → `mot_jepa_build_clips` + `mot_jepa_add_conditioning` → derived clip store.
2. **Fine-tune** the head from the corpus-pretrained checkpoint. Encoder stays frozen.
3. **Offline** — chunk RMSE via `zarr_eval_ftp1_pytorch.eval_loop`, masked per action group.
4. **Closed-loop** — `UniVTAC/scripts/eval_ftp1.py`. UniVTAC is natively 8-D
   (7 Franka joints + gripper); the mapping into the 120-slot layout already exists — arm joints
   at `[9:16]`, gripper at index 44.

> Check the Lustre **inode** budget before converting. It sits at ~24.3M of 26.21M, and it has
> already killed a job mid-write.

---

## 7. File map

| file | role |
|---|---|
| `mot_jepa/layout.py` | token layout, expert slices, the 3D coordinate table |
| `mot_jepa/embed.py` | `StreamEmbed` — per-stream patchify |
| `mot_jepa/mot_encoder.py` | `MoTEncoder`, `EncoderOutput` |
| `mot_jepa/predictor.py` | `MoTPredictor` (pretraining only) |
| `mot_jepa/masking.py` | mask modes, `build_batch_masks` |
| `mot_jepa/losses.py` | `MotJepaLoss` — latent + lowdim + sync |
| `mot_jepa/ema.py` | fp32 shadow teacher |
| `mot_jepa/clip_dataset.py` | clips, conditioning, future action chunks |
| `mot_jepa/action_parse.py` | FTP-1 120-slot action derivation |
| `mot_jepa/action_dit.py` | `ActionDiT`, `flow_matching_loss` |
| `mot_jepa/drifting.py` | IDP geometry and energy |
| `mot_jepa/runtime.py` | DDP, requeue, checkpoints, `load_frozen_backbone` |
| `scripts/mot_jepa_train.py` | pretraining |
| `scripts/mot_jepa_policy_train.py` | policy post-training, both arms |
| `scripts/mot_jepa_action_readout.py` | the viability gate |
| `scripts/mot_jepa_timeshuffle_control.py` | temporal-order control |

---

## 8. Known open issues

1. **Order-blindness (§4)** is the outstanding architectural problem. The targeted fix is a
   time-local `L_sync` — matching video tubelet *t* to tactile tubelet *t* instead of pooling the
   clip. Requires a new pretraining run to evaluate, and a clean baseline with the `051592a`
   checkpoint fix in place.
2. **Thin tactile temporal signal** — only 2.2% of the gel tensor's magnitude varies within a
   clip, so even a corrected objective learns temporal structure from a small fraction of the
   input. Widening the clip's temporal span is the lever; measure variation vs stride before
   rebuilding, and budget inodes.
3. **IDP hyperparameters** (`t_* = 0.95`, `λ_prox = 0.1`) come from the paper's appendix. It does
   **not** publish batch size, learning rate or step count for IDP — it defers to three other
   papers. Those are set from this repo's conventions, not from the paper.
