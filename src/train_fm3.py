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
import copy
import json
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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


def evaluate_split(net, probe, z, y, steps) -> tuple[float, float]:
    net.eval()
    with torch.no_grad():
        logits = logits_through(net, probe, z, steps)
        return F.cross_entropy(logits, y).item(), evaluate.top1(logits, y)


def run_e2e(dataset: str, encoder: str, k, seed: int, lam: float = 0.0):
    """Strategy 1: end-to-end rolled-out classification training.

    Roll the feature through T Euler steps, classify with the frozen probe, and
    backpropagate the cross-entropy through the whole rollout. Only the velocity
    network is updated.

    `lam` weights a displacement penalty on ||z_hat - z||^2. The classifier is
    frozen and the flow is unconstrained, so the cheapest way to cut the loss is
    to drive features into whatever region the classifier scores confidently —
    which need not generalise. This is the countermeasure.
    """
    utils.set_seed(seed)
    steps = config.STAGE3_STEPS

    probe = load_probe(dataset, encoder, k, seed)
    net = identity_init(flow.VelocityMLP(config.FEATURE_DIM[encoder]))
    # Constructed over the flow's parameters only. Handing it a combined module
    # would quietly train the classifier as well.
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.STAGE3_LR)

    z_train, y_train = features(dataset, encoder, "train")
    idx = subset_indices(y_train.numpy(), k, seed)
    z_fit, y_fit = z_train[idx], y_train[idx]
    z_val, y_val = features(dataset, encoder, "val")
    z_test, y_test = features(dataset, encoder, "test")

    loader = DataLoader(
        TensorDataset(z_fit, y_fit), batch_size=config.FM_BATCH_SIZE, shuffle=True
    )

    started = time.time()
    curve = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch, best_state = -1.0, 0, copy.deepcopy(net.state_dict())

    for epoch in range(1, config.STAGE3_EPOCHS + 1):
        net.train()
        running = 0.0
        for z_batch, y_batch in loader:
            z_hat = flow.rollout(net, z_batch, steps)
            loss = F.cross_entropy(probe(z_hat), y_batch)
            if lam:
                loss = loss + lam * (z_hat - z_batch).pow(2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y_batch)

        val_loss, val_acc = evaluate_split(net, probe, z_val, y_val, steps)
        curve["train_loss"].append(running / len(y_fit))
        curve["val_loss"].append(val_loss)
        curve["val_acc"].append(val_acc)

        # Model selection on validation accuracy, as in Stage 1. The test split
        # is never consulted here.
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    _, top1 = evaluate_split(net, probe, z_test, y_test, steps)

    method = f"fm_e2e_lam{lam:g}"
    tag = config.run_tag(dataset, encoder, method, k, seed)
    config.CKPT.mkdir(parents=True, exist_ok=True)
    config.CURVES.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, config.CKPT / f"{tag}.pt")
    (config.CURVES / f"{tag}.json").write_text(json.dumps(curve))

    return {
        "method": method,
        "top1": top1,
        "best_val": best_acc,
        "n_train": len(idx),
        "epochs_run": config.STAGE3_EPOCHS,
        "best_epoch": best_epoch,
        "seconds": round(time.time() - started, 1),
    }


def sweep(runner, lams, seeds=None) -> None:
    """Run one strategy over the Stage 3 pipelines, appending to the ledger."""
    done = utils.completed_runs()
    k = config.STAGE3_K
    for dataset, encoder in config.STAGE3_PIPELINES:
        for lam in lams:
            for seed in (seeds or config.SEEDS):
                method = f"fm_e2e_lam{lam:g}"
                if (dataset, encoder, method, str(k), str(seed)) in done:
                    continue
                r = runner(dataset, encoder, k, seed, lam)
                seconds, best_val = r.pop("seconds"), r.pop("best_val")
                utils.append_run(
                    dataset=dataset, encoder=encoder, head=r.pop("method"),
                    K=k, seed=seed, **{**r, "top1": round(r["top1"], 5)},
                )
                print(f"{dataset:9s} {encoder:14s} {method:16s} seed={seed}  "
                      f"val={best_val:.4f}  test={r['top1']:.4f}  "
                      f"(epoch {r['best_epoch']}, {seconds:.0f}s)")
    print(f"\nLedger: {config.RUNS_CSV}")


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
    parser.add_argument("--e2e", action="store_true", help="Strategy 1 sweep over lambda")
    parser.add_argument("--lam", type=float, nargs="*", help="lambda values to run")
    args = parser.parse_args()
    if args.check:
        check_identity()
    elif args.e2e:
        sweep(run_e2e, args.lam if args.lam else config.STAGE3_LAMBDAS)
    else:
        parser.print_help()
