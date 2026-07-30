"""Top-1 accuracy, confusion matrices, and the aggregate results table.

    python -m src.evaluate    # rebuild results/accuracy_table.md from runs.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src import config


def top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of examples whose highest-scoring class is the correct one."""
    return (logits.argmax(dim=1) == y).float().mean().item()


def confusion_matrix(
    logits: torch.Tensor, y: torch.Tensor, num_classes: int, normalize: bool = True
) -> np.ndarray:
    """Rows are true classes, columns predicted.

    Row-normalised by default, so each row reads as "of the images that really
    were class c, what fraction went where" — which is what makes the matrix
    readable when classes differ in size.
    """
    pred = logits.argmax(dim=1)
    cm = np.zeros((num_classes, num_classes))
    np.add.at(cm, (y.numpy(), pred.numpy()), 1)
    if normalize:
        cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return cm


# --- Aggregation --------------------------------------------------------
#
# Every reported number comes from results/runs.csv via these two functions,
# so nothing in the report is transcribed by hand.


def aggregate() -> pd.DataFrame:
    """Mean and standard deviation of top-1 over the seeds of each setting."""
    runs = pd.read_csv(config.RUNS_CSV)
    stats = (
        runs.groupby(["dataset", "encoder", "head", "K"])["top1"]
        .agg(mean="mean", std="std", runs="count")
        .reset_index()
    )
    stats[["mean", "std"]] *= 100
    # Order K as 5, 10, full rather than alphabetically.
    stats["K"] = pd.Categorical(
        stats["K"], [str(k) for k in config.SHOTS], ordered=True
    )
    return stats.sort_values(["dataset", "encoder", "head", "K"])


def table_markdown() -> str:
    """The accuracy table: one row per pipeline and head, one column per K."""
    stats = aggregate()
    cell = stats.apply(
        # A single run has no standard deviation to report.
        lambda r: f"{r['mean']:.2f}" if r["runs"] == 1 else f"{r['mean']:.2f} ± {r['std']:.2f}",
        axis=1,
    )
    wide = (
        stats.assign(cell=cell)
        .pivot(index=["dataset", "encoder", "head"], columns="K", values="cell")
        .fillna("—")
    )
    missing = int((wide == "—").to_numpy().sum())

    columns = ["dataset", "encoder", "head", *(str(c) for c in wide.columns)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for (dataset, encoder, head), row in wide.iterrows():
        lines.append("| " + " | ".join([dataset, encoder, head, *row]) + " |")

    return (
        "# Accuracy (top-1 %, complete official test split)\n\n"
        f"Generated from `results/runs.csv`. {len(wide)} rows x {len(wide.columns)} "
        f"settings = {wide.size} cells, {missing} missing.\n\n"
        + "\n".join(lines)
        + "\n"
    )


if __name__ == "__main__":
    out = config.RESULTS / "accuracy_table.md"
    text = table_markdown()
    out.write_text(text)
    print(text)
    print(f"written to {out}")
