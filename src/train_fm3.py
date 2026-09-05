"""Stage 3: a flow-matching transform in front of a frozen linear classifier.

    z ──FM (T Euler steps)──► z_hat ──frozen W, b──► logits

    python -m src.train_fm3 --check    # M1 gate: identity start reproduces Stage 1

The classifier is the Stage 1 linear probe for the same dataset, encoder, K and
seed, loaded from results/ckpt/ and frozen. Only the velocity network trains.

Two differences from Stage 2, both deliberate and both easy to get wrong:

- **Features are raw, not L2-normalised.** The probe was fitted to raw cached
  features; handing it unit-norm inputs would present a distribution it has
  never seen.
- **The flow starts at exact identity.** The velocity network's final layer is
  zero-initialised, so v(z, t) = 0, every Euler step adds nothing, and the whole
  system reproduces the Stage 1 probe bit for bit before training begins.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from src import config, evaluate, flow, utils
from src.train import subset_indices


def identity_init(net: flow.VelocityMLP) -> flow.VelocityMLP:
    """Zero the final layer so the flow starts as the identity map.

    Only the last layer. Its inputs are still non-zero, so its weight gradients
    are non-zero on the first backward pass and it leaves zero immediately.
    Zeroing an earlier layer would cut the gradient path and the network would
    never train at all.
    """
    last = [m for m in net.net if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return net


def load_probe(dataset: str, encoder: str, k, seed: int) -> nn.Linear:
    """The Stage 1 linear probe for this exact subset, frozen.

    Seed s used training subset s, so it must be paired with the probe trained
    on that subset — mixing them would silently break the comparison.
    """
    probe = nn.Linear(config.FEATURE_DIM[encoder], config.NUM_CLASSES[dataset])
    tag = config.run_tag(dataset, encoder, "linear", k, seed)
    probe.load_state_dict(torch.load(config.CKPT / f"{tag}.pt", map_location="cpu"))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def logits_through(net, probe, z: torch.Tensor, steps: int) -> torch.Tensor:
    """The full Stage 3 pipeline: transport, then classify."""
    return probe(flow.rollout(net, z, steps))


def features(dataset: str, encoder: str, split: str):
    z, y = utils.load_cache(dataset, encoder, split)
    if config.STAGE3_NORMALIZE:  # False; see the module docstring
        z = torch.nn.functional.normalize(z, dim=1)
    return z, y


def check_identity() -> None:
    """M1 gate: an untrained Stage 3 system must equal the Stage 1 probe.

    This is also what catches the normalisation trap. If the flow ran on
    normalised features the frozen classifier would see the wrong distribution
    and these numbers would diverge immediately.
    """
    import pandas as pd

    runs = pd.read_csv(config.RUNS_CSV)
    ledger = runs[(runs["head"] == "linear") & (runs["K"].astype(str) == str(config.STAGE3_K))]
    worst_direct = worst_ledger = 0.0

    print(f"{'pipeline':30s} {'seed':>4s} {'probe alone':>12s} {'via FM @ init':>14s} "
          f"{'diff':>9s} {'vs ledger':>10s}")
    for dataset, encoder in config.STAGE3_PIPELINES:
        for seed in config.SEEDS:
            probe = load_probe(dataset, encoder, config.STAGE3_K, seed)
            net = identity_init(flow.VelocityMLP(config.FEATURE_DIM[encoder])).eval()
            z, y = features(dataset, encoder, "test")

            with torch.no_grad():
                # The classifier on its own, and the same classifier reached
                # through an identity flow. These must agree exactly: a zero
                # velocity adds exactly zero at every Euler step.
                direct = evaluate.top1(probe(z), y)
                through = evaluate.top1(
                    logits_through(net, probe, z, config.STAGE3_STEPS), y
                )

            row = ledger[(ledger["dataset"] == dataset) & (ledger["encoder"] == encoder)
                         & (ledger["seed"] == seed)]
            recorded = float(row["top1"].iloc[0])
            worst_direct = max(worst_direct, abs(through - direct))
            worst_ledger = max(worst_ledger, abs(through - recorded))
            print(f"{dataset + ' · ' + encoder:30s} {seed:4d} {direct:12.6f} "
                  f"{through:14.6f} {abs(through - direct):9.1e} "
                  f"{abs(through - recorded):10.1e}")

            assert not any(p.requires_grad for p in probe.parameters()), "classifier not frozen"

    print(f"\nlargest difference against the probe alone: {worst_direct:.1e}")
    print(f"largest difference against the ledger:      {worst_ledger:.1e}"
          "   (the ledger stores top-1 rounded to 5 decimals)")
    assert worst_direct == 0.0, "identity start does not reproduce the Stage 1 probe"
    assert worst_ledger < 1e-5, "Stage 3 pipeline disagrees with the recorded Stage 1 result"
    print("\nM1 gate passed: the flow starts as exact identity and the classifier is frozen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run the M1 gate")
    args = parser.parse_args()
    if args.check:
        check_identity()
    else:
        parser.print_help()
