# CVLAB Summer Project

## Stage 3: FM Before a Linear Classifier

## Goal

Start from the Stage 1 linear-probe setting and insert an FM transformation before the pretrained
linear classifier:

z ──FM──→ ẑ ──frozen linear classifier──→ s,

where z is the frozen image-encoder feature, ẑ is the feature after the FM transformation, and s is the
vector of classifier logits.

Train the linear classifier first, exactly as in Stage 1, and then keep it frozen. Initialize the FM close to
identity so that, before Stage 3 training, the complete system behaves approximately like the original
linear probe.

The goal is to test whether FM can transform the frozen encoder features into a representation that is
better handled by the existing linear classifier. Use the same two datasets chosen in Stage 1.

**Scope of this stage.** The training strategies below are intended as structured starting points rather
than fixed recipes. You are encouraged to modify their details, compare variants, and, if motivated,
propose and evaluate alternative training schemes. Any substantial changes should be clearly described
and justified experimentally.

## Experimental Setup

Use the same data splits and sampled training subsets as in the Stage 1 linear-probe experiments.

To keep this stage focused, choose one representative image encoder for each dataset and one training-set
size for the main experiments. A reasonable default is K = 10.

Choose a single number of Euler steps T and use it throughout Stage 3. The corresponding Stage 1
linear probe is the direct baseline.

Use the same general velocity-network design and Euler integration procedure as in Stage 2. Starting
from z, apply the FM for T Euler steps to obtain ẑ, and then pass ẑ through the frozen linear classifier.

## Suggested Strategy 1: End-to-End Rolled-Out Classification Training

For each training feature z, run the complete FM rollout to obtain ẑ, pass ẑ through the frozen
classifier, and compute

Lcls = CE(Wẑ + b, y).

Backpropagate through the complete rollout and update only the FM parameters.

You may also experiment with regularization that discourages unnecessarily large changes to the
representation, for example by penalizing the displacement between z and ẑ, or the magnitude of the
predicted velocities.

## Suggested Strategy 2: Classifier-Guided Targets and Standard FM Training

Here, use the frozen classifier to construct an explicit target representation for FM training.

For each training feature z:

1. Run z through the current FM to obtain ẑ.
2. Pass ẑ through the frozen classifier and compute the classification loss.
3. Use the gradient of this loss with respect to ẑ to construct a nearby improved representation ẑ′,
   for example by taking one or more gradient steps in feature space.
4. Treat z as the source and ẑ′ as the target.
5. Perform a standard FM training update between z and ẑ′.
6. Recompute the targets as the FM changes during training.

Experiment with choices such as the feature-space step size, the number of target-improvement steps,
how often targets are recomputed, and whether the target updates should be normalized or otherwise
constrained.

## Main Comparison

For each of the two datasets, compare:

- Stage 1 linear probe;
- end-to-end rolled-out classification training;
- classifier-guided FM training.

Use the same encoder, training subset, and pretrained classifier for all methods within each dataset.

## Results to Present

**Classification results.** Report top-1 test accuracy for the Stage 1 linear probe and both Stage
3 methods on the two datasets. Also report the change relative to the corresponding linear-probe
baseline.

**Training behavior.** Show representative training and validation curves for both Stage 3 methods.

**Feature-space visualization.** For a readable subset of classes, visualize the original features z and
the transported features ẑ for the two Stage 3 methods. Use the same test examples and class colors
across all comparisons.

PCA or t-SNE may be used. Compute the embedding jointly over the before/after feature sets being
compared. The purpose is to examine how each training strategy changes the class structure of the
frozen representation.

## Optional Extension: Jointly Fine-Tuning the Classifier

After completing the frozen-classifier experiments, you may also unfreeze the pretrained linear classifier
and jointly optimize the FM transformation and classifier.

Compare this with the frozen-classifier setting and with the original Stage 1 linear probe. You may
experiment with choices such as different learning rates for the FM and classifier, delayed unfreezing,
or additional regularization.
