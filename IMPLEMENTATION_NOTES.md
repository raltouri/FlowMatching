# Implementation notes

A running log of implementation decisions and the checks that verified them. Rationale
lives here so the code stays uncluttered and so any choice can be defended later without
reconstructing the reasoning from memory.

Results and findings: [REPORT_STAGE1.md](REPORT_STAGE1.md).
Concepts explained from scratch: [EXPLAINER.md](EXPLAINER.md).

---

## Stage 1

**Features are cached once, and that boundary is architectural.** Only `extract.py` reads
images. Everything else — training, evaluation, figures — reads cached tensors. This makes
the 48 runs cheap, removes data-loading nondeterminism from the comparison, and satisfies
the spec's caching requirement structurally rather than by convention.

**Class names are cached beside the features** (`features/{dataset}_classes.json`). Found
by testing a clone with no `data/` directory: the figures were reopening the dataset purely
to label axes, which would have failed for anyone who had not downloaded the 6.4 GB. Now
the caches are genuinely self-sufficient.

**One ledger, `results/runs.csv`.** Every reported number and figure derives from it, so
nothing is transcribed by hand and no figure can silently disagree with the table. Runs
already present are skipped, making re-runs idempotent and safe to share within the group.

**`config.run_tag()` owns checkpoint and curve filenames**, with the method in the name.
Without it, Stage 2's flow-matching runs would have overwritten Stage 1's probe
checkpoints — which the confusion matrices are regenerated from.

**No `tabulate` dependency.** The markdown accuracy table is six lines of string
formatting; not worth an install step for teammates.

---

## Stage 2

### M1 — `src/flow.py` (velocity network and the two objectives)

Four pieces: `VelocityMLP`, `fm_loss`, `rollout`, `rollout_loss`. 657,280 parameters at
d=384 — small, as the spec asks, since the point is the comparison between objectives, not
an architecture search.

#### Decision: the flow operates on L2-normalised features

`config.FM_NORMALIZE = True`.

Prototypes are unit vectors, but raw features have norms of roughly 9–54. On raw features
the target velocity `p - z` would be dominated by *shrinking the magnitude* — and the cosine
decision rule ignores magnitude entirely. The network would spend most of its capacity on a
quantity that cannot affect the answer. Normalising both sides puts the whole of the
learning problem into direction, which is the only thing classified.

#### Decision: no re-normalisation between Euler steps

Points may drift slightly off the unit sphere during the journey. That is harmless, because
cosine similarity ignores magnitude. Constraining every step back to the sphere would be a
different and more restrictive flow than the plain Euler integration the spec describes.

#### Decision: `rollout` returns intermediate states on request

`rollout(..., return_path=True)` returns `(batch, T + 1, dim)`, starting at the input and
ending at the final position. Inference needs only the endpoint, but the M5 trajectory
figure needs every state — designing it in now avoids later modifying a function every
experiment depends on.

#### Decision: `rollout` is not wrapped in `no_grad`

Gradients must flow through the entire sequence for rolled-out training to mean anything.
Callers performing inference apply `torch.no_grad()` themselves.

#### M1 gate — passed

```
parameters: 657,280
T= 4  final (32, 384)  path (32, 5, 384)
T=12  final (32, 384)  path (32, 13, 384)
fm_loss(oracle)      = 0.00e+00
rollout_loss(oracle) = 1.23e-17
parameters with no gradient: none
standard    loss 0.0060 -> 0.0000   cos to prototype -0.001 -> +1.000
rolled-out  loss 0.0060 -> 0.0000   cos to prototype -0.001 -> +1.000
```

Two of these carry real weight:

**The oracle test.** A network returning exactly `u = p - z` scores 0.00 on `fm_loss`. This
is what verifies the loss is wired to the correct target — a flipped sign or a reversed
interpolation would show up here as a large number instead of zero.

**No parameter without a gradient after a T=12 rollout.** If anything detached between Euler
steps, rolled-out training would silently collapse into a sequence of independent one-step
problems and the central comparison of Stage 2 would be meaningless. Verified rather than
assumed.

Both objectives also drive mean cosine-to-prototype from −0.001 to +1.000 on random data, so
the machinery genuinely transports points.

---

## Open decisions

*(none currently)*
