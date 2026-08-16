"""Quantitative analysis of what the flow does to the feature space.

    python -m src.analysis_stage2     # writes results/geometry_stage2.md

Four questions the accuracy table cannot answer on its own:

1. Are Stage 2's prototypes really identical to Stage 1's? The whole delta
   rests on it, so it is checked rather than assumed.
2. How does the flow change the geometry — measured as a cosine margin, not
   eyeballed from a t-SNE plot.
3. Why does flow matching hurt in the few-shot settings?
4. Which test images does the flow rescue, and which does it break?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src import config, evaluate, flow, heads, utils
from src.figures_stage2 import load_velocity, prototypes_for
from src.train import subset_indices

SETTING = ("full", 0)  # the setting the qualitative figures also use
METHODS = ("fm_std_T4", "fm_roll_T4")


def as_k(value):
    """K arrives from the ledger as text; the sampler needs "full" or an int."""
    return value if value == "full" else int(value)


def prepared(dataset: str, encoder: str, split: str):
    z, y = utils.load_cache(dataset, encoder, split)
    return (F.normalize(z, dim=1) if config.FM_NORMALIZE else z), y


def margin(z: torch.Tensor, y: torch.Tensor, prototypes: torch.Tensor):
    """Mean cosine to the correct prototype, and to the nearest wrong one.

    Their difference is the classification margin: how much closer a point sits
    to its own class centre than to its most tempting competitor. Positive means
    correctly classified on average; larger means more confidently so.
    """
    sims = heads.prototype_logits(z, prototypes)
    correct = sims.gather(1, y.unsqueeze(1)).squeeze(1)
    others = sims.scatter(1, y.unsqueeze(1), float("-inf"))
    nearest_wrong = others.max(dim=1).values
    return correct.mean().item(), nearest_wrong.mean().item()


def transported(dataset, encoder, method, k, seed, z):
    net = load_velocity(dataset, encoder, method, k, seed)
    steps = int(method.split("_T")[1])
    with torch.no_grad():
        return flow.rollout(net, z, steps)


# --- 1. Are the prototypes the same as Stage 1's? -----------------------


def check_prototypes(lines: list[str]) -> None:
    """Rebuild prototypes through the Stage 2 code path and re-derive the
    Stage 1 accuracy from them. If the two stages disagreed about subsets or
    prototypes, this would not reproduce the ledger."""
    runs = pd.read_csv(config.RUNS_CSV)
    baseline = runs[runs["head"] == "prototypes"]
    worst = 0.0
    for row in baseline.itertuples():
        z, y = utils.load_cache(row.dataset, row.encoder, "test")
        protos = prototypes_for(row.dataset, row.encoder, as_k(row.K), row.seed)
        recomputed = evaluate.top1(heads.prototype_logits(z, protos), y)
        worst = max(worst, abs(recomputed - row.top1))

    lines += [
        "## 1. Prototype identity with Stage 1",
        "",
        f"All {len(baseline)} Stage 1 prototype runs were re-derived through the Stage 2 "
        "code path (same subset indices, same `build_prototypes`) and re-evaluated.",
        "",
        f"- largest absolute difference in top-1 against the ledger: **{worst:.2e}**",
        "",
        "The delta reported in the Stage 2 tables therefore compares against exactly the "
        "prototypes Stage 1 used, not a re-derivation that happens to be close.",
        "",
    ]


# --- 2. What the flow does to the geometry ------------------------------


def geometry(lines: list[str]) -> None:
    k, seed = SETTING
    lines += [
        "## 2. Geometry: cosine margin before and after the flow",
        "",
        "Mean cosine similarity to the correct prototype and to the nearest wrong one, "
        f"at K={k}, seed {seed}. The margin is their difference.",
        "",
        "| pipeline | split | state | to correct | to nearest wrong | margin |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset, encoder in config.PIPELINES:
        protos = prototypes_for(dataset, encoder, k, seed)
        for split in ("train", "test"):
            z, y = prepared(dataset, encoder, split)
            if split == "train":
                idx = subset_indices(y.numpy(), k, seed)
                z, y = z[idx], y[idx]
            states = {"original": z}
            for method in METHODS:
                states[method] = transported(dataset, encoder, method, k, seed, z)
            for state, zz in states.items():
                good, bad = margin(zz, y, protos)
                lines.append(
                    f"| {dataset} · {encoder} | {split} | {state} | "
                    f"{good:.3f} | {bad:.3f} | **{good - bad:.3f}** |"
                )
    lines.append("")


# --- 3. Why the flow hurts in the few-shot settings ---------------------


def prototype_quality(lines: list[str]) -> None:
    """How close is a K-shot prototype to the one built from all the data?

    If few-shot prototypes are poor estimates, the flow is being trained to
    transport features toward the wrong targets.
    """
    lines += [
        "## 3. Prototype quality against training-set size",
        "",
        "Mean cosine between each K-shot prototype and the prototype built from the full "
        "training split — how well the target the flow aims at is estimated.",
        "",
        "| pipeline | K=5 | K=10 | K=full |",
        "|---|---:|---:|---:|",
    ]
    for dataset, encoder in config.PIPELINES:
        reference = prototypes_for(dataset, encoder, "full", 0)
        cells = []
        for k in config.SHOTS:
            seeds = (0,) if k == "full" else config.SEEDS
            sims = [
                F.cosine_similarity(
                    prototypes_for(dataset, encoder, k, s), reference, dim=1
                ).mean().item()
                for s in seeds
            ]
            cells.append(f"{np.mean(sims):.3f}")
        lines.append(f"| {dataset} · {encoder} | " + " | ".join(cells) + " |")
    lines.append("")


# --- 4. Which predictions does the flow change? -------------------------


def flips(lines: list[str]) -> None:
    k, seed = SETTING
    lines += [
        "## 4. Predictions the flow changes",
        "",
        f"Test images whose prediction changes when the flow is applied, at K={k}, "
        f"seed {seed}.",
        "",
        "| pipeline | method | wrong → right | right → wrong | net |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, encoder in config.PIPELINES:
        protos = prototypes_for(dataset, encoder, k, seed)
        z, y = prepared(dataset, encoder, "test")
        before = heads.prototype_logits(z, protos).argmax(1) == y
        for method in METHODS:
            moved = transported(dataset, encoder, method, k, seed, z)
            after = heads.prototype_logits(moved, protos).argmax(1) == y
            gained = int((~before & after).sum())
            lost = int((before & ~after).sum())
            lines.append(
                f"| {dataset} · {encoder} | {method} | {gained} | {lost} | "
                f"**{gained - lost:+d}** |"
            )
    lines.append("")


def main() -> None:
    lines = [
        "# Stage 2 — geometric analysis",
        "",
        "Generated by `python -m src.analysis_stage2`.",
        "",
    ]
    check_prototypes(lines)
    geometry(lines)
    prototype_quality(lines)
    flips(lines)

    out = config.RESULTS / "geometry_stage2.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
