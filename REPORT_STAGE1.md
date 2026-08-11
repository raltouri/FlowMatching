# Stage 1 — Report

**Status:** milestones M0–M4 complete. All 48 experimental runs finished; figures and
write-up remain.

Repository: `raltouri/FlowMatching` (private) · environment: Python 3.14, torch 2.13.0,
torchvision 0.28.0, scikit-learn 1.9.0.

---

## 1. Configuration

The spec leaves three choices to the group. Ours, with reasons:

| Choice | Decision | Reason |
|---|---|---|
| Datasets | **DTD** (47 classes, partition 1) and **FGVC-Aircraft** (100 classes, variant level) | Both provide a genuine 5 → 10 → full curve. Flowers-102's official training split holds only 10 images per class, which would collapse K=10 and K=full into the same experiment. |
| Prototype branch | **Option A — image-derived prototypes** | Reuses the feature caches the linear probe already requires, and produces prototypes that vary with K, giving a second curve to compare against the probe. Zero-shot CLIP would yield two numbers and a flat reference line. |
| DINOv2 dataset | **FGVC-Aircraft** | Fine-grained variants are where generic ImageNet features are weakest, so the encoder comparison has the most headroom. |

This yields **six pipelines** (dataset × encoder × head) evaluated at three training-set
sizes — 18 reported cells, from 48 runs.

## 2. Protocol

Official splits only, never merged. Verified counts:

| Dataset | train | val | test | classes | per class |
|---|---:|---:|---:|---:|---|
| DTD (partition 1) | 1,880 | 1,880 | 1,880 | 47 | 40 |
| FGVC-Aircraft (variant) | 3,334 | 3,333 | 3,333 | 100 | 33–34 |

- **K ∈ {5, 10, full}.** Balanced subsets at K=5 and K=10, drawn with seeds {0, 1, 2}.
- **Seeds.** At K=5/10 the seed selects the training subset — the dominant source of
  variance in the few-shot regime. At K=full there is no subset to select, so the same seed
  controls classifier initialisation instead.
- **Model selection** uses the full validation split (highest validation accuracy).
  The test split is evaluated once per run, after training, and is never consulted for any
  decision.
- **Encoders frozen throughout:** `eval()` mode *and* `requires_grad=False`, both asserted at
  extraction time. The first stops BatchNorm updating its running statistics; the second
  stops gradient tracking. Either alone is insufficient.

## 3. Implementation

Features are extracted **once** and cached; every experiment then runs on cached tensors and
never reads an image again. This is the architectural boundary the whole design rests on: it
makes 48 runs cheap, removes data-loading nondeterminism from the comparison, and satisfies
the spec's caching requirement structurally rather than by convention.

```
L0  official split          DTD partition 1 · Aircraft variant level
L1  preprocessing           each checkpoint's own transform
L2  frozen encoder          ResNet-18 → 512-d (pre-fc) · DINOv2 ViT-S/14 → 384-d (CLS)
L3  feature cache  ─────────  boundary: nothing below reads an image
L4  subset selection        K ∈ {5,10,full}, balanced, seeds {0,1,2}
L5  head                    linear s = Wz + b  |  prototypes μ_c
L6  decision                argmax_c s_c       |  argmax_c cos(z, μ_c)
L7  evaluation              top-1 on the full official test split
```

| Module | Role |
|---|---|
| `config.py` | every protocol constant and hyperparameter, in one place |
| `data.py` | official splits, class names, balanced K-shot sampler |
| `encoders.py` | the two frozen encoders and their preprocessing |
| `extract.py` | builds the nine feature caches (45 MB, committed) |
| `heads.py` | `linear_probe` and `build_prototypes` |
| `train.py` | orchestrates all 48 runs, appends to `results/runs.csv` |
| `evaluate.py` | top-1 accuracy, confusion matrices |
| `utils.py` | seeding, cache loading, the results ledger |

**Preprocessing detail.** Every FGVC-Aircraft image carries a copyright banner along the
bottom; a 20-pixel strip is cropped before the encoder's own transform, so the encoder does
not spend part of its representation reading text.

**Results provenance.** Each run appends one row to `results/runs.csv` the moment it finishes.
Every reported number and figure derives from that file, so no value in this report is
transcribed by hand. Per-epoch loss curves and best-validation checkpoints are saved
alongside it.

## 4. Verification

Each milestone was gated before the next began.

| Check | Result |
|---|---|
| Split sizes and class counts vs. published values | pass — confirms DTD partition 1 and Aircraft variant level |
| Sampler determinism, balance, seed-sensitivity, no duplicates | pass |
| Encoders frozen (`eval()` + no `requires_grad`) | pass, asserted in code |
| Cached features aligned with labels, in order | pass, asserted per split |
| Re-extraction bit-identical to cache | pass — max absolute difference 0.0 |
| Feature health (norms, dead rows) | pass — mean norm 23.6, no all-zero rows |
| Re-running experiments is idempotent | pass — ledger unchanged |

The prototype baseline was deliberately run **before** any training code existed. Being
training-free, it validates the entire data → cache → label path in seconds; had the caches
been misaligned, it would have shown accuracy near chance.

## 5. Results

Top-1 accuracy (%) on the complete official test split. Mean ± standard deviation over three
runs; prototypes at K=full are deterministic and require a single run.

| Pipeline | Head | K=5 | K=10 | K=full |
|---|---|---|---|---|
| DTD · ResNet-18 | prototypes | **47.13** ± 0.61 | 51.56 ± 0.92 | 58.78 |
| DTD · ResNet-18 | linear | 46.15 ± 0.71 | **52.09** ± 0.45 | **62.55** ± 0.51 |
| Aircraft · ResNet-18 | prototypes | 16.39 ± 0.26 | 19.69 ± 0.20 | 25.50 |
| Aircraft · ResNet-18 | linear | **20.41** ± 0.35 | **26.90** ± 0.23 | **38.07** ± 0.27 |
| Aircraft · DINOv2 ViT-S/14 | prototypes | 24.33 ± 0.24 | 27.51 ± 1.22 | 34.26 |
| Aircraft · DINOv2 ViT-S/14 | linear | **36.66** ± 0.42 | **50.56** ± 0.12 | **67.69** ± 0.08 |

Chance level is 2.1% on DTD and 1.0% on Aircraft.

### Figures

All regenerate from `results/runs.csv` and the feature caches via `make table figures`.

| File | Content |
|---|---|
| `results/accuracy_table.md` | the table above, generated — 18 cells, 0 missing |
| `figures/stage1/accuracy_vs_k.png` | accuracy vs. K with error bars, both datasets, chance line marked |
| `figures/stage1/loss_curves.png` | train and validation loss, K=10 seed 0, one panel per pipeline |
| `figures/stage1/confusion_dtd.png`, `figures/stage1/confusion_aircraft.png` | row-normalised confusion matrices |
| `results/top_confusions.md` | the largest off-diagonal entries, named |
| `figures/stage1/features_dtd.png`, `figures/stage1/features_aircraft.png` | t-SNE of test features with prototypes |

Encoder is encoded as hue and head as line style, so the two are separable without
relying on four hues. The nine-class projections use three hues x three marker shapes
rather than nine hues, which keeps every pair distinguishable under colour-vision
deficiency; the palette was checked with a validator rather than by eye.

## 6. Findings

**The encoder dominates everything else.** On Aircraft at K=full, swapping ResNet-18 for
DINOv2 ViT-S/14 moves the linear probe from 38.07% to 67.69% — a 29.6-point gain on identical
data, with an identical head and identical training. ImageNet-1K supervised pretraining was
optimised to collapse exactly the within-category distinctions that aircraft variants consist
of; DINOv2's self-supervised objective preserves them.

**Prototypes beat the trained probe in the lowest-data regime — but only where classes are
blob-like.** On DTD at K=5, prototypes reach 47.13% against the probe's 46.15%. The probe must
fit 47 × 512 + 47 ≈ 24,100 parameters from 235 labelled images and overfits; the class mean has
no capacity to overfit and averages sampling noise away. By K=full the ordering reverses and
the probe leads by 3.8 points.

**That crossover does not occur on Aircraft, at either encoder.** Nearest-class-mean
classification assumes each class occupies a compact region around a single centre. One hundred
fine-grained aircraft variants do not: they overlap heavily in a globally-pooled representation.
The probe wins at every K there — by 4 points at K=5 and 12.6 at K=full with ResNet-18, and by
12.3 and 33.4 with DINOv2. The gap between the two heads is therefore a rough diagnostic for how
linearly separable, versus how clustered, a feature space is.

**Accuracy rises monotonically with K in all six pipelines**, and seed spread is small
(≤ 1.22 points, and typically under 0.5). The few-shot results reflect the method rather than a
lucky subset draw.

**Prototype accuracy saturates; probe accuracy does not.** From K=10 to K=full, DINOv2
prototypes gain 6.8 points while the DINOv2 probe gains 17.1. Additional labelled data is worth
far more to a method that can use it to reshape decision boundaries than to one that only
sharpens a class average.

**Validation loss and validation accuracy disagree about when to stop.** In every loss curve,
validation loss reaches its minimum around epoch 30–40 and then climbs steadily, while
validation *accuracy* keeps creeping up to epoch ~195 on Aircraft. This is not a bug: as
training continues the probe grows overconfident, so the cross-entropy it incurs on its
mistakes rises faster than its remaining correct predictions reduce it — yet the *ranking* of
classes still improves slightly. It matters because checkpoint selection uses accuracy, as the
spec requires, and would have stopped 160 epochs earlier had it used loss. The rising
validation loss is the overfitting the curves are meant to show.

**The residual errors are dominated by pairs that are near-indistinguishable by construction.**
The largest off-diagonal entries on Aircraft are C-47 → DC-3 (0.52) and DC-3 → C-47 (0.44) —
the C-47 is the military designation of the DC-3, so the two labels describe what is
essentially the same airframe. The rest are variants differing mainly in fuselage length:
737-900 → 737-800, 737-300 → 737-400, 747-100 → 747-200, E-190 → E-195, MD-90 → MD-80, all at
0.30–0.33. On DTD the pattern repeats semantically: dotted → polka-dotted (0.45) and
polka-dotted → dotted (0.25), plus lined → banded and grid → meshed. A meaningful share of the
remaining error is therefore a property of the label set rather than of the encoder or the
head, which bounds how much any Stage 2 or Stage 3 improvement can be expected to recover.

## 7. Open issue: the 200-epoch cap

The spec's suggested configuration (AdamW, lr 1e-3, weight decay 1e-4, batch 64, 200 epochs,
best-validation checkpoint) was used unchanged. The epoch at which validation accuracy peaked
reveals a caveat:

| Pipeline | K=5 | K=10 | K=full |
|---|---:|---:|---:|
| DTD · ResNet-18 | 52 | 39 | 21 |
| Aircraft · ResNet-18 | 185 | 169 | 64 |
| Aircraft · DINOv2 | 195 | 195 | 128 |

DTD at K=full peaks at epoch 21 of 200 and overfits thereafter — visible in the loss curves.
But DINOv2's few-shot runs peak at ~195, one of them at exactly 200, meaning validation accuracy
was **still improving when the cap stopped training**. Those two cells (36.66% and 50.56%) are
therefore mildly pessimistic.

The loss curves reduce the severity of this considerably. Validation loss bottoms out around
epoch 30–40 in every pipeline and rises thereafter, and validation accuracy is very nearly flat
over the final hundred epochs — the late "improvement" that pushes `best_epoch` to 195 is a
fraction of a percentage point, not a trend that a longer run would extend far. Raising the cap
would therefore change the reported numbers marginally at best.

**Resolution: reported as-is, with the 200-epoch cap and this caveat stated.** The spec permits
adjusting its suggested configuration given validation evidence, but here the evidence argues the
adjustment is unnecessary rather than overdue.

## 8. What the visualisations show

**The two-dimensional projections make the encoder gap visible.** On Aircraft, the same nine
classes and the same test images under ResNet-18 form a single undifferentiated cloud with all
nine prototypes crowded near its centre; under DINOv2 the same points separate into distinct
regions with prototypes sitting inside their own clusters. This is the qualitative counterpart of
the 29.6-point accuracy gap.

It also explains why prototypes fail on Aircraft: the method's assumption is that a class occupies
a compact region around one centre, and the ResNet-18 panel shows directly that it does not.

On DTD the ResNet-18 projection already shows clean clustering, which is consistent with
prototypes remaining competitive there.

Features and prototypes were both L2-normalised before projecting, and the projection was fitted
jointly to both, as the spec requires — without normalisation the plot would have separated
prototypes from images by vector magnitude alone rather than by class. The same classes, test
examples, and colours are used across both encoders on a dataset. These plots are qualitative:
t-SNE distances between clusters are not faithful and should not be read as accuracy.

## 9. Remaining work

1. README with reproduction commands (`make verify features runs table figures`).
2. Rehearse the protocol and findings for the oral discussion.
