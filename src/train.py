"""Run the experiments and append every result to results/runs.csv.

    python -m src.train                    # both heads
    python -m src.train --head prototypes  # one head only

Runs already in the ledger are skipped, so this is safe to re-run and safe for
group members to share results/runs.csv.
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import config, data, evaluate, heads, utils


def subset_indices(y_train: np.ndarray, k, seed: int) -> np.ndarray:
    """Training indices for one setting. K="full" uses the whole split."""
    if k == "full":
        return np.arange(len(y_train))
    return data.sample_k_shot(y_train, k, seed)


def run_prototypes(dataset: str, encoder: str, k, seed: int):
    """Image-derived prototypes. No training: a class mean and a dot product."""
    z_train, y_train = utils.load_cache(dataset, encoder, "train")
    z_test, y_test = utils.load_cache(dataset, encoder, "test")

    idx = subset_indices(y_train.numpy(), k, seed)
    prototypes = heads.build_prototypes(
        z_train[idx], y_train[idx], config.NUM_CLASSES[dataset]
    )
    logits = heads.prototype_logits(z_test, prototypes)
    return {"top1": evaluate.top1(logits, y_test), "n_train": len(idx)}


def run_probe(dataset: str, encoder: str, k, seed: int):
    """Train s = Wz + b with softmax cross-entropy on cached features.

    The seed drives both the K-shot subset draw and the classifier's
    initialisation. At K=full there is no subset to draw, so it is purely an
    initialisation seed — which is exactly what the spec asks for.
    """
    utils.set_seed(seed)
    z_train, y_train = utils.load_cache(dataset, encoder, "train")
    z_val, y_val = utils.load_cache(dataset, encoder, "val")
    z_test, y_test = utils.load_cache(dataset, encoder, "test")

    idx = subset_indices(y_train.numpy(), k, seed)
    z_fit, y_fit = z_train[idx], y_train[idx]

    model = heads.linear_probe(
        config.FEATURE_DIM[encoder], config.NUM_CLASSES[dataset]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    loader = DataLoader(
        TensorDataset(z_fit, y_fit), batch_size=config.BATCH_SIZE, shuffle=True
    )

    curve = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for z_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(z_batch), y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y_batch)

        model.eval()
        with torch.no_grad():
            val_logits = model(z_val)
            val_loss = F.cross_entropy(val_logits, y_val).item()
            val_acc = evaluate.top1(val_logits, y_val)

        curve["train_loss"].append(running / len(y_fit))
        curve["val_loss"].append(val_loss)
        curve["val_acc"].append(val_acc)

        # Checkpoint selection: highest validation accuracy. The test split is
        # never consulted here.
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        top1 = evaluate.top1(model(z_test), y_test)

    tag = f"{dataset}_{encoder}_K{k}_s{seed}"
    config.CKPT.mkdir(parents=True, exist_ok=True)
    config.CURVES.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, config.CKPT / f"{tag}.pt")
    (config.CURVES / f"{tag}.json").write_text(json.dumps(curve))

    return {
        "top1": top1,
        "n_train": len(idx),
        "epochs_run": config.MAX_EPOCHS,
        "best_epoch": best_epoch,
    }


HEADS = {"prototypes": run_prototypes, "linear": run_probe}


def main(head: str = "all") -> None:
    done = utils.completed_runs()
    selected = HEADS if head == "all" else {head: HEADS[head]}

    for name, run in selected.items():
        for dataset, encoder in config.PIPELINES:
            for k in config.SHOTS:
                # Prototypes at K=full are deterministic, so the spec asks for
                # a single run. The probe still needs 3 initialisation seeds.
                seeds = (
                    (0,) if (k == "full" and name == "prototypes") else config.SEEDS
                )
                for seed in seeds:
                    key = (dataset, encoder, name, str(k), str(seed))
                    if key in done:
                        continue
                    result = run(dataset, encoder, k, seed)
                    utils.append_run(
                        dataset=dataset,
                        encoder=encoder,
                        head=name,
                        K=k,
                        seed=seed,
                        **{**result, "top1": round(result["top1"], 5)},
                    )
                    print(
                        f"{dataset:9s} {encoder:14s} {name:10s} "
                        f"K={str(k):5s} seed={seed}  top1={result['top1']:.4f}"
                    )

    print(f"\nLedger: {config.RUNS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head", default="all", choices=["all", *HEADS], help="which head to run"
    )
    main(**vars(parser.parse_args()))
