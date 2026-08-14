"""Stage 2: train the velocity network and classify the transported features.

    python -m src.train_fm                                  # the full sweep
    python -m src.train_fm --dataset aircraft --K full \
                           --mode std --steps 4 --seed 0    # one setting

Everything upstream comes from Stage 1 unchanged — the same feature caches, the
same K-shot subsets for a given (K, seed), and prototypes built by the same
function. That is what makes the reported delta against the Stage 1 prototype
baseline meaningful; any divergence here would silently invalidate it.

Results append to the shared results/runs.csv under head names `fm_std_T4`,
`fm_std_T12`, `fm_roll_T4`, `fm_roll_T12`, alongside the Stage 1 rows.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import config, evaluate, flow, heads, utils
from src.train import subset_indices

MODES = ("std", "roll")


def method_name(mode: str, steps: int) -> str:
    return f"fm_{mode}_T{steps}"


def run_fm(dataset: str, encoder: str, k, seed: int, mode: str, steps: int):
    """Train one velocity network and evaluate it on the test split.

    Both training modes get an identical architecture, optimiser, epoch budget
    and batch size; the objective is the only difference between them, which is
    what the spec requires for the comparison to be fair.
    """
    utils.set_seed(seed)
    device = utils.pick_device()

    z_train, y_train = utils.load_cache(dataset, encoder, "train")
    z_test, y_test = utils.load_cache(dataset, encoder, "test")

    # Same subset as the Stage 1 run with this (K, seed), and prototypes built
    # from that subset by the same function.
    idx = subset_indices(y_train.numpy(), k, seed)
    z_fit, y_fit = z_train[idx], y_train[idx]
    prototypes = heads.build_prototypes(z_fit, y_fit, config.NUM_CLASSES[dataset])

    # The flow lives in normalised space; see IMPLEMENTATION_NOTES.md.
    if config.FM_NORMALIZE:
        z_fit, z_test = F.normalize(z_fit, dim=1), F.normalize(z_test, dim=1)

    z_fit, z_test = z_fit.to(device), z_test.to(device)
    prototypes = prototypes.to(device)
    targets = prototypes[y_fit.to(device)]

    net = flow.VelocityMLP(config.FEATURE_DIM[encoder]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.FM_LR)
    loader = DataLoader(
        TensorDataset(z_fit, targets), batch_size=config.FM_BATCH_SIZE, shuffle=True
    )

    started = time.time()
    curve = {"train_loss": []}
    for _ in range(config.FM_EPOCHS):
        net.train()
        running = 0.0
        for z_batch, p_batch in loader:
            loss = (
                flow.fm_loss(net, z_batch, p_batch)
                if mode == "std"
                else flow.rollout_loss(net, z_batch, p_batch, steps)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(z_batch)
        curve["train_loss"].append(running / len(z_fit))

    # Transport the test features, then classify with the Stage 1 rule.
    net.eval()
    with torch.no_grad():
        moved = flow.rollout(net, z_test, steps)
        top1 = evaluate.top1(heads.prototype_logits(moved, prototypes).cpu(), y_test)

    method = method_name(mode, steps)
    tag = config.run_tag(dataset, encoder, method, k, seed)
    config.CKPT.mkdir(parents=True, exist_ok=True)
    config.CURVES.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), config.CKPT / f"{tag}.pt")
    (config.CURVES / f"{tag}.json").write_text(json.dumps(curve))

    return {
        "top1": top1,
        "n_train": len(idx),
        "epochs_run": config.FM_EPOCHS,
        "seconds": round(time.time() - started, 1),
    }


def main(dataset=None, encoder=None, K=None, seed=None, mode=None, steps=None) -> None:
    done = utils.completed_runs()
    pipelines = [
        (d, e)
        for d, e in config.PIPELINES
        if (dataset is None or d == dataset) and (encoder is None or e == encoder)
    ]
    shots = config.SHOTS if K is None else (K,)
    seeds = config.SEEDS if seed is None else (seed,)
    modes = MODES if mode is None else (mode,)
    step_counts = config.FM_STEPS if steps is None else (steps,)

    for d, e in pipelines:
        for m in modes:
            for T in step_counts:
                method = method_name(m, T)
                for k in shots:
                    for s in seeds:
                        if (d, e, method, str(k), str(s)) in done:
                            continue
                        result = run_fm(d, e, k, s, m, T)
                        seconds = result.pop("seconds")
                        utils.append_run(
                            dataset=d,
                            encoder=e,
                            head=method,
                            K=k,
                            seed=s,
                            **{**result, "top1": round(result["top1"], 5)},
                        )
                        print(
                            f"{d:9s} {e:14s} {method:11s} K={str(k):5s} seed={s}  "
                            f"top1={result['top1']:.4f}  ({seconds:.0f}s)"
                        )

    print(f"\nLedger: {config.RUNS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=config.DATASETS)
    parser.add_argument("--encoder", choices=config.ENCODERS)
    parser.add_argument("--K")
    parser.add_argument("--seed", type=int, choices=config.SEEDS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--steps", type=int, choices=config.FM_STEPS)
    main(**vars(parser.parse_args()))
