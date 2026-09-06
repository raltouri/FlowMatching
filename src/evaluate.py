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


# Which ledger rows belong to which stage. Named explicitly rather than matched
# by prefix: Stage 3's heads also begin with "fm_", and a prefix match would
# silently pull them into the Stage 2 table.
STAGE1_HEADS = ("prototypes", "linear")
STAGE2_HEADS = ("fm_std_T4", "fm_std_T12", "fm_roll_T4", "fm_roll_T12")
# Stage 3 heads carry their hyperparameter in the name, so they are matched by
# prefix; the variants of one strategy are compared and one is chosen on
# validation.
STAGE3_PREFIXES = ("fm_e2e_", "fm_guided_")


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
    stats = stats[stats["head"].isin(STAGE1_HEADS)]
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
        "# Stage 1 — accuracy (top-1 %, complete official test split)\n\n"
        f"Generated from `results/runs.csv`. {len(wide)} rows x {len(wide.columns)} "
        f"settings = {wide.size} cells, {missing} missing.\n\n"
        + "\n".join(lines)
        + "\n"
    )


def aggregate_stage2() -> pd.DataFrame:
    """Flow-matching results, each paired with its Stage 1 prototype baseline.

    The delta is the quantity Stage 2 is actually about, so it is computed here
    rather than by eye: every FM row is matched to the baseline row with the
    same dataset, encoder and K.
    """
    stats = aggregate()
    baseline = stats[stats["head"] == "prototypes"].set_index(
        ["dataset", "encoder", "K"]
    )["mean"]
    fm = stats[stats["head"].isin(STAGE2_HEADS)].copy()
    fm["baseline"] = [
        baseline.loc[(row.dataset, row.encoder, row.K)] for row in fm.itertuples()
    ]
    fm["delta"] = fm["mean"] - fm["baseline"]
    return fm


def table_stage2_markdown() -> str:
    """One table per pipeline: accuracy, spread, and delta against baseline."""
    fm = aggregate_stage2()
    shots = [str(k) for k in config.SHOTS]
    lines = [
        "# Stage 2 — accuracy (top-1 %, complete official test split)",
        "",
        "Generated from `results/runs.csv`. Bracketed values are the change against",
        "the Stage 1 prototype baseline for the same dataset, encoder and K.",
    ]

    for (dataset, encoder), group in fm.groupby(["dataset", "encoder"], observed=True):
        lines += [
            "",
            f"## {dataset} · {encoder}",
            "",
            "| method | " + " | ".join(f"K={k}" for k in shots) + " |",
            "|" + "|".join(["---"] * (len(shots) + 1)) + "|",
        ]
        base = group.set_index("K")["baseline"]
        lines.append(
            "| prototype baseline (Stage 1) | "
            + " | ".join(f"{base.loc[k].iloc[0]:.2f}" for k in shots)
            + " |"
        )
        for method, rows in group.groupby("head", observed=True):
            rows = rows.set_index("K")
            cells = []
            for k in shots:
                r = rows.loc[k]
                cells.append(f"{r['mean']:.2f} ± {r['std']:.2f} ({r['delta']:+.2f})")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def aggregate_stage3() -> pd.DataFrame:
    """Stage 3 results, each paired with its Stage 1 *linear probe* baseline.

    Note the baseline differs from Stage 2's: there the flow was compared with
    the prototype rule, here with the trained probe that sits downstream of it.
    """
    stats = aggregate()
    baseline = stats[stats["head"] == "linear"].set_index(
        ["dataset", "encoder", "K"]
    )["mean"]
    is_stage3 = stats["head"].str.startswith(STAGE3_PREFIXES)
    fm = stats[is_stage3].copy()
    fm["baseline"] = [
        baseline.loc[(row.dataset, row.encoder, row.K)] for row in fm.itertuples()
    ]
    fm["delta"] = fm["mean"] - fm["baseline"]
    fm["strategy"] = np.where(
        fm["head"].str.startswith("fm_e2e_"), "end-to-end", "classifier-guided"
    )
    return fm


def table_stage3_markdown() -> str:
    """Every Stage 3 variant, with its change against the Stage 1 linear probe."""
    fm = aggregate_stage3()
    lines = [
        "# Stage 3 — accuracy (top-1 %, complete official test split)",
        "",
        "Generated from `results/runs.csv`. Bracketed values are the change against the",
        "Stage 1 **linear probe** for the same dataset, encoder and K.",
    ]
    for (dataset, encoder), group in fm.groupby(["dataset", "encoder"], observed=True):
        lines += ["", f"## {dataset} · {encoder}", ""]
        for k, rows in group.groupby("K", observed=True):
            lines += [
                f"**K = {k}**",
                "",
                "| strategy | variant | top-1 | delta |",
                "|---|---|---|---:|",
                f"| — | Stage 1 linear probe | {rows['baseline'].iloc[0]:.2f} | |",
            ]
            for r in rows.sort_values("head").itertuples():
                spread = "" if pd.isna(r.std) else f" ± {r.std:.2f}"
                lines.append(
                    f"| {r.strategy} | {r.head} | {r.mean:.2f}{spread} | {r.delta:+.2f} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for name, build in (
        ("accuracy_table.md", table_markdown),
        ("accuracy_table_stage2.md", table_stage2_markdown),
        ("accuracy_table_stage3.md", table_stage3_markdown),
    ):
        out = config.RESULTS / name
        out.write_text(build())
        print(f"written to {out}")
