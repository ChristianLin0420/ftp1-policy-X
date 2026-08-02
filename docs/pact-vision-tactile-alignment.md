# PACT — Pose-Anchored Contact Tokens

A vision–tactile alignment change to FTP-1, motivated by the premise that **language is a
separable modality** (discrete, symbolic, task-specifying) while **vision and touch are the pair
that must be tightly bound** — both are continuous, high-bandwidth channels observing the *same*
contact event.

This document records the design, the defects it targets, what was implemented, and what was
deliberately left out.

---

## 1. The defects, verified in code

| # | Defect | Evidence |
|---|---|---|
| 1 | **A 224×224×3 gel frame is compressed to one 512-d vector.** `SharedImageChunkEncoder` kept `tokens[:, 0, :]` and discarded all 196 patch tokens. | `ftp1_blocks.py` `SharedImageChunkEncoder.forward` |
| 2 | **Vision and tactile never attend to each other, at any layer.** `build_expert_attention_layout` writes a block-diagonal mask: the prefix block is zeroed against tactile columns and vice versa. The two streams meet only as independent K/V for the action expert, after 18 layers of separate processing. | `ftp1_attention_masks.py:49-99` |
| 3 | **No alignment objective exists anywhere.** A grep across `src/openpi/` finds no contrastive, predictive, or cross-modal term. The 99.1 M tactile expert is trained solely by gradient through one cross-attention edge from a masked-MSE flow loss. | `src/openpi/` |
| 4 | **Slot allocation is 24× over-provisioned and identity is a nameplate.** `total_num_tactile_tokens = 2 × 24 = 48`; UniVTAC fills 2. Identity is `nn.Embedding(48, d)` — an index, not a pose. | `ftp1_blocks.py:552-553` |

Measured baseline: **3.6424 B** parameters total (`paligemma_with_expert` 3566.8 M,
`hpt_tactile_encoder` 64.7 M). The "4.3 B" figure in the runbook is wrong.

## 2. Design position

Touch is **not** aligned to vision's *appearance*. The literature is consistent that
appearance-similarity alignment is the wrong instrument: global vision–tactile InfoNCE has been
measured as net-negative, and where it helps, the benefit accrues to the *vision* encoder. Aligning
touch to vision teaches touch to discard exactly what makes it valuable — force, shear, slip,
hardness — none of which vision can see.

Instead:

- **Spatial structure is preserved**, not collapsed to a CLS vector.
- **Routing is asymmetric.** Tactile reads vision; vision never reads tactile. This preserves the
  prefix KV cache and keeps a fully gated batch bit-identical to a vision-language-only policy.
- **Language stays out of the tactile pathway.** Implemented as *absence* — no touch-language
  objective, plus the language columns held closed. Cost zero, and flipping one boolean is the
  falsification test.

## 3. What was implemented

All three changes default to **off**, so the baseline code path is bit-identical to FTP-1.

### 3.1 Spatial tactile tokens

`FTP1TactileTokenizerConfig.tokens_per_area` (default `1`) and `spatial_pool_grid` (default `3`).

- `1` — legacy CLS-only readout.
- `1 + grid²` — CLS ++ adaptive-average-pooled patch grid. With `grid=3` this is 10 tokens per pad.

When `tokens_per_area > 1` the encoder switches from FTP-1's scatter into 48 canonical slots to a
**compact layout** carrying only the areas that exist. Area identity is still supplied by
`func_area_idx_embedding` indexed by the true function-area id, so the pretrained table is reused
unchanged; a small `spatial_token_embedding` distinguishes the slots within an area.

Measured on the UniVTAC config:

| Setting | Tokens | Valid |
|---|---|---|
| `tokens_per_area=1` (FTP-1) | 48 | **2** |
| `tokens_per_area=10` (PACT) | 20 | **20** |
| `tokens_per_area=10` + gating, one pad open | 20 | 11 |

The tactile sequence gets **10× more real content while getting shorter**.

### 3.2 Vision → tactile attention edge

`FTP1ModelConfig.tactile_reads_vision` (default `False`). When enabled, tactile rows may read the
**image** columns of the prefix. Image tokens always precede language tokens in `embed_prefix`, so
the count is tracked there and the slice is exact.

Invariants held by construction and asserted:

- vision never reads tactile (`m[prefix_rows, tactile_cols] == 0`),
- language columns stay closed (`m[tactile_rows, lang_cols] == 0`).

The inference path required a matching change: FTP-1 prefills the tactile expert **separately**
with its own self-attention layout and then concatenates KV. With the edge open, the tactile
prefill instead runs against the VLM cache (`past_key_values=vlm_past_key_values`), which returns
`[prefix ++ tactile]` in the layout the action branch already expects — so the manual merge is
skipped and train/sampling stay consistent.

### 3.3 Contact gating (model side)

`FTP1ModelConfig.contact_gating` (default `False`). An area reported as not-in-contact keeps its
CLS token and has its spatial tokens masked out of attention — "this pad is present and feels
nothing" is informative, so the CLS must survive.

Requires the dataset to emit `Observation.tactile_contact` of shape `(B, N)`. **The dataset side is
not implemented in this arm** (see §5), so the flag stays off and the code path is inert.

### 3.4 Modality-dominance monitor

`compute_modality_grad_norms` logs pre-clipping gradient norms per branch
(`tactile_encoder`, `tactile_expert`, `vision_tower`, `action_expert`) and the ratio
`GradNorm/tactile_over_vision`. If that ratio does not rise during training, the tactile pathway is
being starved by the pretrained vision-language stream and any headline gain is not coming from
touch.

## 4. Running the two arms

The same launcher produces both, so the comparison varies only the architecture — same data, same
seed-42 split, same normalization statistics, same 20 000-step qualified recipe.

```bash
# Baseline (bit-identical to the existing FTP-1 path)
FTP1_TACTILE_TOKENS_PER_AREA=1 FTP1_TACTILE_READS_VISION=false \
  bash scripts_exp_zarr/univtac/train_univtac_example.sh

# PACT
FTP1_TACTILE_TOKENS_PER_AREA=10 FTP1_TACTILE_READS_VISION=true \
  bash scripts_exp_zarr/univtac/train_univtac_example.sh
```

An existing baseline run is already available at
`runtime/checkpoints/ftp1/univtac_lift_bottle_a100_20k/19999`, so the baseline arm need not be
re-run.

## 5. Deliberately not in this arm

Each of these changes the **input distribution**, which forces a normalization recompute and would
confound a controlled A/B. They belong in a separate arm.

- **Reference-difference encoding and T=4 temporal channels.** Requires a `history_mode` refactor
  (`disable_history` currently raises if any per-modality history list is non-empty) plus ViT stem
  inflation 3→12 channels, plus new norm stats.
- **Contact-gate supervision.** The shipped detector, `compute_tactile_delta_score`, is
  `mean|I_t − I_{t−1}|` — a *temporal-change* detector, not a contact detector. A pad in sustained
  static contact scores near zero and is classified no-contact. The repair keeps the whole MAD
  calibration pipeline and changes only the score to `mean|I_t − I_ref|` against a per-episode
  open-gripper reference.
- **Physics supervision (`L_phys`) and future-field prediction (`L_fut`).** The raw HDF5s **do**
  contain the required arrays — `tactile/*/depth (T,240,320)`, `tactile/*/marker (T,2,1200,2)`,
  `tactile/*/pose (T,7)`, `embodiment/ee (T,7)` — and the converter currently reads only
  `rgb_marker`. Recovering them is a converter change over the same files, with no new collection
  and no annotation. Note the marker array is 1200 markers as `[init_xy, curr_xy]`, so shear is
  `marker[:,1] − marker[:,0]`; an earlier draft assumed 63 markers.
- **Image-plane projection of the contact patch.** Dead on the head camera by arithmetic: at
  `fx_224 ≈ 161.7` and ~1.0 m standoff a GelSight Mini pad spans ~4.3 px ≈ 0.31 of one 14-px SigLIP
  patch, and the two pads are ~0.90 patch apart. Gated behind extracting wrist-camera intrinsics
  and validating a reprojection overlay.

## 5.5 Results (20 000 steps, UniVTAC lift_bottle, 2×A100)

Both arms: identical data, seed-42 split, normalization statistics, and recipe; architecture is
the only difference. Baseline W&B `4tdhuz7r`, PACT `2b3520wj`, comparison `dhws8jhv`.

### Offline action error — no significant difference

Paired across 10 validation points, PACT minus baseline:

| metric | mean Δ | SE | t |
|---|---|---|---|
| `rmse_total` | +1.35% | 2.36% | +0.57 |
| `rmse_right-arm-joints` | +1.35% | 2.36% | +0.57 |
| `rmse_right-hand-joint` | +1.47% | 3.27% | +0.45 |
| `mape_total` | +3.46% | 2.44% | +1.42 |
| `jitter_rms_total` | +4.68% | 2.86% | +1.63 |

Every |t| < 2. PACT is indistinguishable from baseline, trending marginally worse. The change is
free in memory (~38.9 GiB/GPU, baseline-identical) and throughput (0.92 s/step), and does not
damage the pretrained prior (step-0 within 0.6%).

### Tactile ablation — the informative result

`rmse_right-arm-joints` under test-time tactile substitution:

| condition | baseline | Δ | PACT | Δ |
|---|---|---|---|---|
| `real` | 0.0136 | — | 0.0143 | — |
| `zero` | 0.0166 | **+22.1%** | 0.0193 | **+35.0%** |
| `noise` | 0.0166 | +22.1% | 0.0193 | +35.0% |
| `shuffle` | 0.0137 | **+0.7%** | 0.0142 | **−0.7%** |

1. **Both models use tactile substantially**, and use its *structure*: `noise` (matched mean/std,
   structure destroyed) is worth exactly as much as `zero`.
2. **PACT increased tactile reliance 1.6×** — 35.0% vs 22.1% degradation when touch is removed.
   The architecture change measurably altered how much the policy leans on that pathway.
3. **Neither model binds tactile to the current observation.** A real gel frame from an unrelated
   timestep performs as well as the correct one on both arms. The tactile stream is a
   distributional anchor, not observation-specific evidence.

### Conclusion

Connectivity is not the bottleneck. Nothing in the flow-matching action loss ever requires the
tactile tokens to describe *this* contact, so the model has no reason to bind them to it — and
does not, even with 10× the tokens and an open vision→tactile edge.

This makes `L_phys` (§5) a measured requirement rather than an argument: it regresses penetration
depth and shear, a target that **cannot be satisfied by a shuffled tactile input**. The falsifiable
prediction is that after `L_phys`, `real`→`shuffle` must open a gap. The numbers above (+0.7% /
−0.7%) are the pre-registered baseline for that test, and
`scripts/zarr_eval_ftp1_pytorch.py --tactile_mode` is the instrument.

## 6. Risks

1. **The evaluation task may be unable to show a gain.** UniVTAC reports `lift_bottle`-family
   tasks moving within noise while tactile gains concentrate in `grasp_classify`. Offline action
   MSE on the held-out split is a diagnostic, not a success rate.
2. **Statistical power.** A 10–15 pp delta over 100 trials near 50 % is inside the confidence
   interval. Closed-loop claims need ≥200 trials per condition.
3. **DDP re-qualification.** The change alters tactile branch shapes and therefore DDP bucket
   structure. The runbook records an illegal-memory-access failure for full-model DDP init on one
   host.
4. **Modality dominance may persist** despite the open edge — monitored by
   `GradNorm/tactile_over_vision`, not assumed away.

## 7. Decisive ablations

| | Ablation | Falsifies |
|---|---|---|
| **B** | Mask flip: `tactile_reads_vision` false vs true, everything else fixed | the peer-attention thesis (one boolean) |
| **G** | Token budget: `tokens_per_area` 1 vs 10 | the CLS-bottleneck claim |
| **E** | Real tactile / zeroed / matched-statistics noise, at fixed parameter count | whether the tactile pathway carries information or is merely capacity |
| **A** | Physics objectives only, no vision edge | whether vision→touch alignment matters at all versus touch-only SSL |

Ablations B and G are the two implemented here and are separable by env var.
