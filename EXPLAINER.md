# What this project is doing — plain-language explainer

Written for someone with no background in machine learning. No formulas are required to
follow it; the few that appear are explained in words first.

---

## 1. The task

Show the computer a photograph and have it answer *which category is this?* — which texture
(DTD, 47 categories) or which aircraft model (FGVC-Aircraft, 100 categories).

## 2. The one idea everything rests on: turning a picture into numbers

A computer cannot compare photographs directly. So the first step is always to convert each
image into a list of numbers — say 512 of them.

Think of those numbers as **coordinates**. Each image becomes a single point in a space with
512 directions instead of the usual 3. The conversion is built so that **similar-looking
images land near each other**: photos of striped things cluster in one region, dotted things
in another.

Once every image is a point, "which category is this?" becomes a geometry question:
*which region of the space did this point land in?*

- **Feature** (or **embedding**): the list of numbers for one image. Its coordinates.
- **Feature space**: the space all those points live in.

## 3. The encoder, and what "frozen" means

The **encoder** is the thing that does the converting: image in, numbers out.

Building a good one from scratch needs millions of images and enormous computing power. So
we don't. We download encoders other people already trained, and we use them **unchanged** —
that is what **frozen** means. We never adjust them; we only ever ask them for coordinates.

We use two, to compare them:

- **ResNet-18** — older, trained on ImageNet with human labels.
- **DINOv2** — newer, trained without labels, generally produces better coordinates.

Because a frozen encoder always returns the same numbers for the same image, we run every
image through it **once**, save the results to disk, and never touch the images again. That
saved file is the **feature cache**, and it is why the rest of the project is fast.

## 4. Stage 1 — two ways to classify, once you have coordinates

**Prototypes (the averaging method).** For each category, take the coordinates of its
labelled examples and average them. That average point is the category's **prototype** — its
typical location. To classify a new image, find which prototype it's closest to.

There is no training here at all. It is arithmetic.

**Linear probe (the trained method).** Instead of averaging, *learn* boundaries that separate
the categories. You show it examples, it adjusts itself, it improves, you keep the best
version. Only this small final piece is trained — the encoder stays frozen.

Stage 1 measured both, at 5, 10, and all available labelled examples per category. The
results are in [REPORT_STAGE1.md](REPORT_STAGE1.md).

---

## 5. Stage 2 — flow matching

Here is the situation Stage 1 leaves us with. Each category has a prototype. Images of that
category land *near* it, but scattered — some quite far away, close enough to another
category's prototype to be misclassified.

**The idea of Stage 2: instead of leaving the points where they fall, learn to move them.**
Nudge each image's point toward where its category actually lives, *then* classify. If the
nudging works, points that were ambiguous end up unambiguous.

The problem, of course, is that at test time we don't know which category an image belongs
to — that's the question. So the mover cannot be told "go to prototype 7." It has to look at
where a point *is* and infer which way it should go.

### What a velocity network is

**Velocity** just means *which direction to move, and how fast*. A velocity of "3 metres per
second, north-east" has both.

The **velocity network** is a small program that answers one question:

> Given a point at position `z`, and given how far along the journey we are (call it `t`),
> which direction should it move, and how fast?

Position in, direction-and-speed out. Nothing more.

A useful picture: imagine ocean currents. At every location the water pushes in some
direction at some speed. Drop a boat anywhere and the current carries it somewhere. The
velocity network is a **learned current** — and we are trying to shape that current so it
carries every image toward its own category's prototype.

"Network" here just means it's a small stack of arithmetic with adjustable knobs (about a
million of them). Training = tuning the knobs. Ours is deliberately small: two layers of 512.

### Why time (`t`) is one of the inputs

The journey happens in stages. Early on, points are still scattered and need large
corrections; near the end they are almost home and need small ones. Feeding in `t` — a
number from 0 (start) to 1 (finish) — lets the network behave differently at different
points in the trip, rather than applying one fixed rule throughout.

### How the journey is actually taken: Euler steps

In principle the point glides smoothly. In practice a computer takes a **finite number of
discrete steps** — this is called **Euler integration**, and it works exactly how you'd
guess:

> Ask the network which way to go. Move a little in that direction. Ask again from the new
> position. Repeat.

**T** is how many steps. We test **T = 4** and **T = 12**. More steps means a smoother, more
accurate path but more computation. It's the difference between walking to a destination
checking your map 4 times versus 12 times.

After the last step, the point has arrived somewhere — and we classify it exactly as in
Stage 1: whichever prototype it's now nearest to.

### Training method 1: standard flow matching

To teach the current, we need examples of correct movement. During training we *do* know
each image's category, so we know where it should end up.

Draw a straight line from the image's point to its prototype. Pick a random spot on that
line. Ask the network what velocity it would suggest there, and tell it the right answer:
*point straight at the prototype*. Repeat millions of times, for random images and random
spots.

Cheap and simple — one question per example.

**But there's a subtle flaw.** Training only ever asks about points **on the ideal straight
line**. At test time, the network steers using its own previous answers, so after one
imperfect step it is standing somewhere slightly *off* that line — a situation it was never
trained on. Errors then compound: each step lands somewhere a bit stranger than the last.

This is like learning to drive by only ever practising while perfectly centred in your lane.
The moment you drift, you're in a situation you've never rehearsed.

### Training method 2: rolled-out flow matching

**Rolled-out** training fixes exactly that. Instead of asking about a random point on the
ideal line, it runs **the entire journey during training** — the same T steps, starting from
the image's point, each step using the network's own output, drift and all.

Then it checks only one thing: **did the point end up at the prototype?** If not, it adjusts
all T decisions at once to make the ending better.

"Rolled out" means the multi-step sequence is unrolled into one long chain that is graded end
to end.

- **Standard FM** grades each step against an ideal it may never encounter.
- **Rolled-out FM** grades the destination, having actually made the trip.

The trade-off is cost: rolled-out training does T times the work, because every training
example requires the full T-step journey rather than a single question. T=12 rollout is the
slowest thing in this project.

### What Stage 2 is actually testing

Three questions:

1. Does moving the points at all beat leaving them where they fell? (flow matching vs. the
   Stage 1 prototype baseline)
2. Does rehearsing the real journey beat rehearsing the ideal one? (rolled-out vs. standard)
3. What does the flow *do* to the space? We plot the points before and after, and we plot
   individual journeys, to see whether categories genuinely become more separated.

---

## 6. "Are we training the velocity network, or actually moving the image vectors?"

Both — but they are different things happening at different times, and the distinction is
worth being precise about.

### What gets trained: only the velocity network

Three things are involved, and only one of them learns:

| | Trained? |
|---|---|
| **Encoder** (image → 512 numbers) | ❌ Frozen. Never touched, same as Stage 1 |
| **Prototypes** (the target points) | ❌ Fixed. Computed once by averaging, then held still as targets |
| **Velocity network** | ✅ **The only thing that learns** |

The cached features on disk are **never modified**. `features/*.pt` is exactly the same file
before and after Stage 2.

### What "moving the vectors" actually means

The movement is **temporary and computed on demand**. At test time:

1. Load the image's feature `z` from the cache
2. Run the T-step journey → get a new position `ẑ_T`
3. Classify `ẑ_T` against the prototypes
4. Throw `ẑ_T` away

Nothing is saved. The next time you classify that image, the journey is recomputed.

**This is the key point:** we are not repositioning 3,333 test points one by one. We are
shaping a *current*, and the current then moves any point you drop into it — including test
images the network has never seen. That is what makes it generalise. If we merely memorised
"move image #472 to here," it would be useless on new images.

### How the training actually works

Training means repeatedly: make a prediction, measure how wrong it was, nudge the network's
~1 million internal numbers to be slightly less wrong. Repeat thousands of times. That is
gradient descent — the same procedure that trained the Stage 1 linear probe.

The only difference between the two Stage 2 methods is **what gets measured**.

**Standard FM — one training step:**

1. Take a training image's feature `z` and its prototype `p` (we know its label during training)
2. Pick a random `t`, say 0.3 — jump to the point 30% along the straight line from `z` to `p`
3. Ask the network: "what velocity here?"
4. Compare its answer to the correct one — straight at `p`
5. Nudge the network's weights to close that gap

**Rolled-out FM — one training step:**

1. Take `z` and `p`
2. Run the **whole** T-step journey, each step using the network's own output
3. Measure only: how far is the final position from `p`?
4. Nudge **all T decisions at once** to make that ending better

Step 4 is why it is expensive — the correction has to flow backwards through all T
evaluations.

### One more thing worth knowing

You do not train *one* velocity network — you train **108 of them**, one per experimental
setting (dataset × encoder × K × seed × training mode × T). Each is a separate small network
trained from scratch on that setting's data.

That is why the compute matters: 108 trainings, and each rolled-out T=12 run does roughly
12× the work of a standard one.

---

## 7. What a "sweep" is, and why there are 108 of them

A **run** is one complete experiment: train one velocity network from scratch, then measure
how accurate it is on the test images. A **sweep** is running *every* combination of the
experimental settings, one after another, rather than picking a few interesting ones.

The number comes from multiplying out five choices:

| Choice | Options | |
|---|---|---|
| Which dataset + encoder | DTD·ResNet-18, Aircraft·ResNet-18, Aircraft·DINOv2 | 3 |
| How many labelled examples (K) | 5, 10, all | 3 |
| Random seed | 0, 1, 2 | 3 |
| Training method | standard, rolled-out | 2 |
| Euler steps (T) | 4, 12 | 2 |

3 × 3 × 3 × 2 × 2 = **108 runs**.

### Why every combination, rather than a sample

Every question Stage 2 asks is a *comparison*, and a comparison is only fair if everything
except the thing being compared is held identical. To answer "does rolled-out beat
standard?", both must be trained on the same data, with the same seed, the same T, and the
same architecture — so the training objective is the only difference left. The same applies
to "does T=12 beat T=4?" and "does flow matching beat the Stage 1 baseline?"

The three seeds are what let the results carry a "±". Repeating each setting three times
with different random draws shows whether a difference is real or just luck in which
examples happened to be sampled.

### What happens inside one run

1. Load the cached features (no images involved)
2. Build the class prototypes from that setting's labelled subset
3. Train the velocity network for 200 epochs
4. Transport the test features through the T-step journey
5. Measure top-1 accuracy
6. Append one row to `results/runs.csv`

Step 6 is what makes progress durable: results are written as they finish, so an interrupted
sweep resumes where it stopped instead of repeating completed work.

The `fm_roll_T12` runs are the slowest, because each training step backpropagates through
all twelve Euler steps rather than answering a single question.

---

## 8. Glossary

| Term | Plain meaning |
|---|---|
| **Feature / embedding** | The list of numbers representing one image; its coordinates |
| **Feature space** | The space those points live in |
| **Encoder** | The converter: image in, numbers out |
| **Frozen** | Never modified. We only ever read from it |
| **Feature cache** | The saved numbers, so images are processed only once |
| **Prototype** | A category's average point — its typical location |
| **Cosine similarity** | Measures whether two points lie in the same *direction*, ignoring how far out they are |
| **Linear probe** | A small trained classifier that separates categories with straight boundaries |
| **K** | How many labelled examples per category we allow (5, 10, or all) |
| **Seed** | A number fixing the random choices, so a run can be repeated exactly |
| **Top-1 accuracy** | Percentage of test images whose top guess was right |
| **Velocity network** | A small program: given a position and a time, which way to move and how fast |
| **Flow matching** | Training that velocity so points flow to the right destination |
| **t** | How far through the journey, from 0 (start) to 1 (end) |
| **Euler step** | One "check direction, move a little" increment |
| **T** | How many such steps the journey takes (we test 4 and 12) |
| **Standard FM** | Trains on random points along the ideal straight path |
| **Rolled-out FM** | Trains on the full self-steered journey, graded by where it ends |
| **Run** | One complete experiment: train one network, measure its accuracy |
| **Sweep** | Running every combination of the experimental settings in turn |
| **Epoch** | One pass through all the training examples. We do 200 per run |
| **Loss** | A score for how wrong the model currently is. Training tries to make it small |
| **Overfitting** | Memorising the training examples instead of learning the pattern; looks great on training data, worse on new data |
| **PCA / t-SNE** | Ways to squash 512 numbers down to 2 so points can be drawn on paper |
