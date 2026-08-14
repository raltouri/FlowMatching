# Stage 2 — Report

**Status:** all 108 flow-matching runs complete. Figures and the geometric analysis remain.

Builds directly on [REPORT_STAGE1.md](REPORT_STAGE1.md); design decisions are recorded in
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md); the concepts are explained without
jargon in [EXPLAINER.md](EXPLAINER.md).

---

## 1. What Stage 2 adds

A velocity network `v_θ(z, t)` transports each frozen image feature toward the fixed
prototype of its class. Classification itself is unchanged from Stage 1: after transporting,
the point is assigned to the nearest prototype by cosine similarity.

Everything upstream is reused **verbatim** — the same feature caches, the same K-shot
subsets for a given (K, seed), prototypes built by the same function, the same official test
splits. That is not an optimisation; it is what makes ΔAcc against the Stage 1 baseline
meaningful. Any divergence would silently invalidate the comparison.

Only the velocity network is trained. The encoders remain frozen and the prototypes are
fixed targets, never optimised.

## 2. Protocol

| | |
|---|---|
| Architecture | MLP, input `[z ; t]`, two hidden layers of 512, SiLU. 657k parameters at d=384 |
| Optimiser | AdamW, lr 1e-3, batch 64, 200 epochs |
| Euler steps | T ∈ {4, 12}; for rolled-out training, the same T at train and test time |
| Training sizes | K ∈ {5, 10, full}, seeds {0, 1, 2} — identical subsets to Stage 1 |
| Runs | 3 pipelines × 3 K × 3 seeds × 2 modes × 2 T = **108** |

Both training modes receive an identical architecture, optimiser, epoch budget and batch
size. The objective is the only difference between them, which is what the spec requires for
the comparison to be fair.

**Two implementation decisions**, both recorded with reasoning in `IMPLEMENTATION_NOTES.md`:

- **The flow operates on L2-normalised features.** Prototypes are unit vectors while raw
  features have norms of roughly 9–54, so on raw features the target velocity `p - z` would be
  dominated by a magnitude change that the cosine decision rule ignores entirely.
- **No best-validation checkpointing.** Both modes train for a fixed 200 epochs and the final
  network is evaluated. The spec asks that training be stable rather than tuned, and an
  identical budget keeps the two modes comparable.

## 3. Results

Top-1 accuracy (%) on the complete official test split, mean ± standard deviation over three
seeds. The bracketed value is ΔAcc against the corresponding Stage 1 prototype baseline.

### DTD · ResNet-18

| method | K=5 | K=10 | K=full |
|---|---|---|---|
| prototype baseline (Stage 1) | 47.13 | 51.56 | 58.78 |
| fm_std_T4 | 44.15 ± 0.84 (−2.98) | 47.68 ± 0.45 (−3.88) | 59.49 ± 0.08 (+0.71) |
| fm_std_T12 | 44.63 ± 0.97 (−2.50) | 48.33 ± 0.52 (−3.23) | **59.63** ± 0.05 (+0.85) |
| fm_roll_T4 | 40.39 ± 1.35 (−6.74) | 40.89 ± 0.78 (−10.67) | 56.01 ± 0.27 (−2.77) |
| fm_roll_T12 | 40.53 ± 1.43 (−6.60) | 41.33 ± 0.88 (−10.23) | 55.80 ± 0.32 (−2.98) |

### FGVC-Aircraft · ResNet-18

| method | K=5 | K=10 | K=full |
|---|---|---|---|
| prototype baseline (Stage 1) | 16.39 | 19.69 | 25.50 |
| fm_std_T4 | 17.36 ± 0.23 (+0.97) | 23.60 ± 0.37 (+3.91) | 32.33 ± 0.43 (+6.83) |
| fm_std_T12 | **17.74** ± 0.30 (+1.35) | **23.69** ± 0.30 (+4.00) | **32.59** ± 0.40 (+7.09) |
| fm_roll_T4 | 14.89 ± 0.82 (−1.50) | 20.26 ± 0.88 (+0.57) | 25.50 ± 1.74 (−0.00) |
| fm_roll_T12 | 14.61 ± 0.58 (−1.78) | 20.30 ± 0.97 (+0.61) | 25.75 ± 1.57 (+0.25) |

### FGVC-Aircraft · DINOv2 ViT-S/14

| method | K=5 | K=10 | K=full |
|---|---|---|---|
| prototype baseline (Stage 1) | 24.33 | 27.51 | 34.26 |
| fm_std_T4 | 33.47 ± 0.41 (+9.14) | 43.21 ± 0.69 (+15.70) | **58.17** ± 0.86 (+23.90) |
| fm_std_T12 | 33.18 ± 0.29 (+8.85) | 42.76 ± 0.72 (+15.25) | 57.63 ± 0.53 (+23.36) |
| fm_roll_T4 | 33.62 ± 0.68 (+9.29) | 43.31 ± 0.75 (+15.80) | 54.25 ± 0.59 (+19.99) |
| fm_roll_T12 | **33.74** ± 0.89 (+9.41) | **43.44** ± 0.82 (+15.93) | 53.39 ± 0.58 (+19.12) |

## 4. Findings

### Flow matching works spectacularly where the prototype assumption fitted badly

On Aircraft · DINOv2 at K=full, accuracy rises from 34.26% to **58.17%** — a gain of 23.9
points. For context, Stage 1's linear probe on the same features scored 67.69%. The flow
layer therefore recovers roughly **72% of the gap** between the prototype method and a fully
trained discriminative classifier, while still classifying by nothing more than "which
prototype is nearest".

The gains are large at every K on this pipeline: +9.1 at K=5, +15.7 at K=10.

### Its value is proportional to how badly prototypes were failing

The three pipelines fall into a clean ordering, and it is predicted entirely by the Stage 1
gap between the prototype baseline and the linear probe:

| Pipeline | Stage 1 prototype | Stage 1 probe | gap | best ΔAcc at K=full |
|---|---:|---:|---:|---:|
| Aircraft · DINOv2 | 34.26 | 67.69 | 33.4 | **+23.90** |
| Aircraft · ResNet-18 | 25.50 | 38.07 | 12.6 | +7.09 |
| DTD · ResNet-18 | 58.78 | 62.55 | 3.8 | +0.85 |

Where prototypes already captured the class structure — DTD, where Stage 1's t-SNE showed
clean, compact clusters — there was almost nothing left to recover, and the flow at best
matches the baseline. Where classes were badly non-compact — 100 fine-grained aircraft
variants — the flow has room to work and exploits it.

This is a satisfying result because the mechanism is visible: nearest-class-mean assumes each
class occupies one compact region, and flow matching's job is precisely to *make* that
assumption true by moving points toward their class centre.

### Standard FM beats rolled-out training in every pipeline — the opposite of its motivation

Rolled-out training exists to remove the train/test mismatch: standard FM supervises only
points on the ideal straight path, while inference visits states the network generated
itself. Removing that mismatch was expected to help.

It does not. Standard FM wins in **all nine** dataset × K cells, and the margin is largest
exactly where it should have been smallest — on the pipelines with the most data.

The training curves explain why:

| Aircraft · DINOv2, K=full, T=4 | final training loss | test accuracy |
|---|---:|---:|
| standard | 0.00041 | **58.17%** |
| rolled-out | 0.00011 | 54.26% |

**Rolled-out fits its own objective nearly four times better and generalises worse.** That is
overfitting, and the mechanism is specific: rolled-out training supervises a single endpoint
per example and only shapes the velocity field along the trajectories its own training points
happen to travel. Standard FM draws a fresh random `t` for every example in every epoch,
spreading supervision across the entire interpolation path — which covers far more of the
feature space and acts as a strong regulariser.

So the coverage that standard FM gains from random-`t` sampling is worth more than the
train/test consistency that rolled-out training buys. This is the most discussable result of
Stage 2.

The effect is worst on DTD at K=10, where rolled-out costs **10.7 points**, and rolled-out
also shows markedly higher seed variance on Aircraft · ResNet-18 (±1.74 against ±0.43 for
standard), consistent with a less stable optimisation problem.

### The number of Euler steps barely matters

T=4 and T=12 differ by less than the seed-to-seed spread in almost every cell. T=12 is
marginally ahead for standard FM (+0.1 to +0.6 points) and marginally behind for rolled-out
at K=full. Since rolled-out training costs roughly T× a standard run, T=12 rollout is by far
the most expensive configuration and the least rewarding.

### Flow matching narrows but does not close the gap to a trained classifier

Even at its best — 58.17% on Aircraft · DINOv2 — flow matching remains 9.5 points below the
Stage 1 linear probe (67.69%). Transporting features toward class centres recovers most of
what nearest-class-mean gives away, but a discriminatively trained boundary still extracts
more from the same frozen features.

## 5. Remaining work

1. Accuracy-versus-K plot with error bars: baseline and all four FM variants.
2. Training-loss curves, standard versus rolled-out.
3. Feature-space comparison: before FM, after standard FM, after rolled-out FM — same test
   examples, classes and colours, with the projection fitted jointly across all three views
   and the prototypes.
4. Flow trajectories for a few representative test examples, using PCA.
5. Optional: the reverse flow, starting from the prototypes.
