# Stage 3 in plain language

Written for someone with no machine-learning background. Continues from
[EXPLAINER.md](EXPLAINER.md), which covers Stages 1 and 2 — read that first if the words
*feature*, *encoder*, *prototype* or *flow matching* are new.

Technical terms appear in **bold** the first time, each with what it actually means.

---

## 1. Where we are

**Stage 1** built two ways to sort images into categories, both working on frozen number-lists
("features") produced by a pretrained converter:

- averaging the examples of each class into a **prototype** and picking the nearest one;
- training a **linear classifier** — a small trained rule that draws straight dividing lines
  between the categories.

**Stage 2** added a **flow**: a learned "current" that moves each image's point toward its class
prototype before classifying. It helped a lot where the prototypes were doing badly.

**Stage 3 changes what sits at the end of the pipeline.** Instead of moving points toward
prototypes, we move them toward whatever makes the *Stage 1 linear classifier* answer correctly.

## 2. What Stage 3 actually builds

A sandwich of three parts:

```
image feature  ──►  the flow  ──►  moved feature  ──►  linear classifier  ──►  answer
      z              (4 steps)          ẑ                 (frozen)            "class 37"
```

- **z** is the frozen encoder's number-list for one image.
- The **flow** nudges it, in 4 small steps, to somewhere new.
- **ẑ** ("z-hat") is where it ends up.
- The **linear classifier** is the exact one you trained in Stage 1, now **frozen** — meaning
  we never change it again, we only use it.

Only the flow is trained. The encoder was already frozen in Stage 1, and now the classifier is
frozen too.

## 3. The question being asked

The Stage 1 classifier is fixed and imperfect. It gets, say, 52% of DTD images right.

**Could we get more out of it without touching it — just by rearranging what we feed it?**

That's the whole of Stage 3. Not a better classifier: a better *presentation* of the data to the
classifier we already have.

An analogy: imagine a marker who grades essays by a rigid rubric, and you cannot change the
rubric. Stage 3 asks whether rewriting the essays — same content, different arrangement — gets
better marks out of the same rubric.

## 4. Three ideas you need for this stage

### Starting at "do nothing"

The spec requires the flow to begin as an **identity** transformation — a fancy word for "changes
nothing". Feature goes in, identical feature comes out.

Why start with something useless? Because it makes the comparison honest. At the very first
moment, the Stage 3 system *is* the Stage 1 classifier, giving exactly the same accuracy. So any
change afterwards is unambiguously caused by the flow. There is no chance of accidentally
comparing against a slightly different starting point.

**How it's done:** the flow's little network has several layers of arithmetic. We set the numbers
in its **final layer** to all zeros. That makes its output zero, so each step adds nothing, so the
feature comes out unchanged.

We zero only the *last* layer, on purpose. Training works by tracing how each internal number
affected the final answer — the **gradient**, meaning "how much would the result change if I
nudged this number". If we zeroed an *early* layer, everything after it would be zero too and
those traces would vanish, so the network could never learn anything. Zeroing the last layer
leaves the traces intact, and it starts learning on the very first update.

### Raw numbers, not rescaled ones

In Stage 2 we rescaled every feature to have length 1, because prototypes are length-1 and the
comparison there only cared about *direction*.

**Stage 3 must not do that** — and this is the easiest mistake to make by copying Stage 2's code.
The Stage 1 classifier was trained on the raw, un-rescaled features (their lengths run from about
9 to 54). Hand it length-1 inputs and it is being shown a kind of data it has never seen, and
would fall apart. So Stage 3 works on raw features throughout.

### There is no target any more

This is the deep difference from Stage 2.

In Stage 2 the flow had a **destination**: your class's prototype, a specific point we could name
in advance. Training was "get closer to that point."

In Stage 3 there is no such point. Nobody can say where a feature *should* end up — only whether
the classifier got the answer right once it arrived. The flow has to work out for itself what
"better" means to a classifier it cannot see inside.

That's why there are two strategies: they're two different ways of coping with having no target.

## 5. The two strategies

Both are required by the spec. They share a goal and differ in how the flow is told what to do.

### Strategy 1 — end-to-end

Push the feature through all 4 steps, hand the result to the classifier, and see how wrong the
answer was. That wrongness is the **loss** — a single number scoring the mistake, here
**cross-entropy**, which is small when the classifier confidently picks the right class and large
when it confidently picks the wrong one.

Then **backpropagate**: trace that one number backwards through the classifier and through all
four steps, working out how every internal number in the flow contributed, and nudge each of them
to make the next answer better.

*Like learning to drive by taking the entire trip, being told only your final grade, and adjusting
everything you did.*

**The risk.** The classifier is frozen and the flow is unconstrained, so the *easiest* way to score
well is to shove features far into whatever region the classifier happens to feel confident about —
which may have nothing to do with what the image actually shows. It would look brilliant on the
training images and fail on new ones. That failure has a name: **overfitting**, memorising the
examples instead of learning the pattern.

The countermeasure is a **regularisation** term — an extra penalty added to the loss to discourage
unwanted behaviour. Here it penalises the **displacement**, the distance between where the feature
started and where it ended: "improve the answer, but do not move things further than you need to."
Its strength is a dial called **lambda (λ)**, and we try λ = 0, 0.01 and 0.1 to see which is best.

### Strategy 2 — classifier-guided targets

Rather than tracing back through the whole journey, invent a destination.

Ask the classifier: *"which way should this feature move to be graded better?"* That question has
a precise answer — the **gradient** of the loss with respect to the feature, meaning the direction
in which the score improves fastest. Step a little way in that direction and you have a new,
slightly-better feature ẑ′.

Now you have a destination, so you can train the flow exactly as in Stage 2: travel from z to ẑ′.
As the flow improves, recompute the destinations and repeat.

*Like asking an expert to mark a better spot on the map, then practising the drive there normally.*

**The risk.** The destination keeps moving — every time the flow changes, the recomputed target
changes too. That can chase its own tail and oscillate instead of settling. The dials that control
it are the step size **eta (η)**, how many steps to take, and how often to redraw the map.

## 6. What we measure

**Accuracy**, against the Stage 1 classifier alone. Since the system starts as exactly that
classifier, the flow can only help or hurt — and a result of "neither strategy helps" would be a
real finding, not a failure. We report what happens either way.

**Training and validation curves.** The **validation set** is a pool of labelled images kept
separate from training, used to check progress honestly. If the training score keeps improving
while the validation score stalls or drops, that's overfitting made visible.

**Pictures of the feature space.** The same images plotted before the flow and after each strategy,
squashed from hundreds of numbers down to two so they can be drawn. The point is to *see* how each
strategy rearranged the classes.

## 7. The scale of it

12 runs — 2 datasets × 2 strategies × 3 repeats — taking a few minutes in total. Far smaller than
Stage 2's 108, because the spec deliberately narrows this stage to one encoder per dataset, one
training-set size (**K = 10**, meaning 10 labelled examples per class) and one number of steps
(**T = 4**).

## 8. New terms in this stage

| Term | Plain meaning |
|---|---|
| **Linear classifier** | The Stage 1 trained rule: straight dividing lines between classes |
| **Logits** | The raw scores it gives each class before picking the biggest |
| **Frozen** | Never modified. Used, not trained |
| **Identity** | A transformation that changes nothing |
| **ẑ (z-hat)** | The feature after the flow has moved it |
| **Loss** | One number scoring how wrong the model currently is |
| **Cross-entropy** | The particular loss used for classification: small when confidently right, large when confidently wrong |
| **Gradient** | "Which way, and how much, would nudging this change the result" |
| **Backpropagation** | Tracing the loss backwards through every step to work out those nudges |
| **Regularisation** | An extra penalty that discourages unwanted behaviour |
| **Displacement** | How far the flow moved a feature from where it started |
| **λ (lambda)** | The dial controlling how strongly displacement is penalised |
| **η (eta)** | The step size used to build Strategy 2's target |
| **Validation set** | Held-out labelled images used to check progress honestly |
| **Overfitting** | Memorising the training examples instead of learning the pattern |
