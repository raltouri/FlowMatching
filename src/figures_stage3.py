"""The Stage 3 figures, all derived from results/ and features/.

    python -m src.figures_stage3

Style and palette are shared with Stages 1 and 2 so the three reports read as
one document. The variant of each strategy that gets plotted is chosen by
validation accuracy, in code, so the choice cannot drift from the report.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from src import config, data, evaluate, flow, utils
from src.figures_stage1 import (
    BLUE,
    CLASS_COLORS,
    ENCODER_LABEL,
    INK,
    INK_2,
    MUTED,
    ORANGE,
    SURFACE,
    apply_style,
    viz_classes,
)
from src.train_fm3 import features, identity_init, load_probe, logits_through

STRATEGY_COLOR = {"end-to-end": BLUE, "classifier-guided": ORANGE}
PREFIX = {"end-to-end": "fm_e2e_", "classifier-guided": "fm_guided_"}


def save(fig, name: str) -> None:
    config.FIGURES_STAGE3.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_STAGE3 / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(config.ROOT)}")


def mean_best_val(dataset: str, encoder: str, method: str, k) -> float:
    accs = []
    for seed in config.SEEDS:
        tag = config.run_tag(dataset, encoder, method, k, seed)
        accs.append(max(json.loads((config.CURVES / f"{tag}.json").read_text())["val_acc"]))
    return float(np.mean(accs))


def select_on_validation(dataset: str, encoder: str, strategy: str, k) -> str:
    """The variant of one strategy with the best mean validation accuracy.

    Computed here rather than hardcoded, so the figures and the report can never
    disagree about which variant was chosen — and so the test split plays no
    part in the choice.
    """
    runs = pd.read_csv(config.RUNS_CSV)
    candidates = sorted(
        runs[
            (runs["dataset"] == dataset)
            & (runs["encoder"] == encoder)
            & (runs["K"].astype(str) == str(k))
            & (runs["head"].str.startswith(PREFIX[strategy]))
        ]["head"].unique()
    )
    return max(candidates, key=lambda m: mean_best_val(dataset, encoder, m, k))


# --- Figure 1: the main comparison --------------------------------------


def fig_accuracy() -> None:
    stats = evaluate.aggregate()
    k = config.STAGE3_K
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    for ax, (dataset, encoder) in zip(axes, config.STAGE3_PIPELINES):
        rows = stats[
            (stats["dataset"] == dataset)
            & (stats["encoder"] == encoder)
            & (stats["K"] == str(k))
        ].set_index("head")

        probe = rows.loc["linear"]
        bars = [("Stage 1\nlinear probe", probe["mean"], probe["std"], MUTED)]
        for strategy in PREFIX:
            method = select_on_validation(dataset, encoder, strategy, k)
            r = rows.loc[method]
            label = strategy.replace("-", "-\n", 1) + f"\n({method.split('_')[-1]})"
            bars.append((label, r["mean"], r["std"], STRATEGY_COLOR[strategy]))

        x = np.arange(len(bars))
        ax.bar(x, [b[1] for b in bars], yerr=[b[2] for b in bars], width=0.55,
               color=[b[3] for b in bars], capsize=4, error_kw={"elinewidth": 1})
        # The probe is the reference, so the deltas are what the reader wants.
        for i, b in enumerate(bars[1:], start=1):
            ax.annotate(f"{b[1] - probe['mean']:+.2f}", (i, b[1]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", color=INK_2, fontsize=8)
        ax.axhline(probe["mean"], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [b[0] for b in bars])
        ax.set_title(f"{dataset} · {ENCODER_LABEL[encoder]}")
        ax.set_ylim(0, max(b[1] for b in bars) * 1.25)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("top-1 accuracy (%)")
    fig.suptitle(
        f"Neither strategy meaningfully beats the frozen probe at K={k}\n"
        "Dashed line is the Stage 1 probe; labels give the change against it. "
        "Error bars are one standard deviation over three seeds.",
        fontsize=10, color=INK, y=1.08,
    )
    save(fig, "accuracy.png")


# --- Figure 2: training and validation behaviour ------------------------


def fig_curves() -> None:
    k = config.STAGE3_K
    seed = 0
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))

    for row, (dataset, encoder) in zip(axes, config.STAGE3_PIPELINES):
        probe_acc = None
        chosen = {}
        for col, strategy in enumerate(PREFIX):
            method = select_on_validation(dataset, encoder, strategy, k)
            chosen[strategy] = method
            tag = config.run_tag(dataset, encoder, method, k, seed)
            curve = json.loads((config.CURVES / f"{tag}.json").read_text())
            epochs = np.arange(1, len(curve["train_loss"]) + 1)
            ax = row[col]
            ax.plot(epochs, curve["train_loss"], color=STRATEGY_COLOR[strategy],
                    linewidth=2)
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_title(f"{strategy}: training loss\n({method})", fontsize=9)

        ax = row[2]
        for strategy, method in chosen.items():
            tag = config.run_tag(dataset, encoder, method, k, seed)
            curve = json.loads((config.CURVES / f"{tag}.json").read_text())
            acc = np.array(curve["val_acc"]) * 100
            ax.plot(np.arange(1, len(acc) + 1), acc,
                    color=STRATEGY_COLOR[strategy], linewidth=2, label=strategy)
        # The probe's own *validation* accuracy — the curves on this axis are
        # validation, so a test-set reference line would compare two different
        # things.
        probe = load_probe(dataset, encoder, k, seed)
        z_val, y_val = features(dataset, encoder, "val")
        with torch.no_grad():
            probe_acc = evaluate.top1(probe(z_val), y_val) * 100
        ax.axhline(probe_acc, color=MUTED, linewidth=0.8, linestyle="--")
        ax.annotate("Stage 1 probe (validation)", (len(acc) * 0.45, probe_acc),
                    textcoords="offset points", xytext=(0, 4), color=MUTED, fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_title("validation accuracy", fontsize=9)
        ax.legend(loc="lower right")

        row[0].set_ylabel(f"{dataset}\n\ntraining loss (log)")
        row[2].set_ylabel("validation accuracy (%)")

    fig.suptitle(
        f"Training and validation behaviour, K={k}, seed {seed}\n"
        "The two training losses measure different quantities and are not comparable "
        "in value; validation accuracy is.",
        fontsize=10, color=INK, y=1.04,
    )
    fig.tight_layout()
    save(fig, "loss_curves.png")


# --- Figure 3: what each strategy does to the feature space -------------


def transported(dataset, encoder, method, k, seed, z):
    net = flow.VelocityMLP(config.FEATURE_DIM[encoder])
    tag = config.run_tag(dataset, encoder, method, k, seed)
    net.load_state_dict(torch.load(config.CKPT / f"{tag}.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        return flow.rollout(net, z, config.STAGE3_STEPS)


def fig_feature_space(dataset: str, encoder: str) -> None:
    k, seed = config.STAGE3_K, 0
    classes = viz_classes(dataset)
    names = data.cached_class_names(dataset)

    z_test, y_test = features(dataset, encoder, "test")
    keep = np.isin(y_test.numpy(), classes)
    points, labels = z_test[keep], y_test[keep].numpy()

    views = {"original features": points}
    for strategy in PREFIX:
        method = select_on_validation(dataset, encoder, strategy, k)
        views[f"{strategy}\n({method})"] = transported(
            dataset, encoder, method, k, seed, points
        )

    # One embedding fitted over all three views together, so the panels share a
    # single frame and can be compared.
    joint = torch.cat(list(views.values())).numpy()
    embedded = TSNE(n_components=2, perplexity=30, init="pca",
                    random_state=0).fit_transform(joint)

    n = len(points)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for i, (ax, title) in enumerate(zip(axes, views)):
        xy = embedded[i * n : (i + 1) * n]
        for slot, cls in enumerate(classes):
            ax.scatter(*xy[labels == cls].T, s=13, alpha=0.6, linewidths=0,
                       color=CLASS_COLORS[slot])
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)

    handles = [
        Line2D([], [], color=CLASS_COLORS[s], marker="o", linestyle="",
               markersize=6, label=names[c])
        for s, c in enumerate(classes)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"{dataset} · {ENCODER_LABEL[encoder]}, K={k}: the frozen representation before "
        "and after each strategy\n"
        "Same test images and one jointly fitted t-SNE across all three panels. "
        "Qualitative only.",
        fontsize=10, color=INK, y=1.06,
    )
    save(fig, f"feature_space_{dataset}.png")


# --- Figure 4: the effect of the displacement penalty -------------------


def fig_regularisation() -> None:
    fm = evaluate.aggregate_stage3()
    e2e = fm[fm["head"].str.startswith("fm_e2e_")].copy()
    e2e["lam"] = e2e["head"].str.replace("fm_e2e_lam", "").astype(float)

    # One shared x position per lambda across both K values: the two series do
    # not cover the same lambdas, so indexing each series separately would plot
    # them at mismatched positions.
    lam_order = sorted(e2e["lam"].unique())
    position = {lam: i for i, lam in enumerate(lam_order)}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, (dataset, encoder) in zip(axes, config.STAGE3_PIPELINES):
        rows = e2e[(e2e["dataset"] == dataset) & (e2e["encoder"] == encoder)]
        for kk, marker in ((str(config.STAGE3_K), "o"), ("full", "s")):
            sub = rows[rows["K"] == kk].sort_values("lam")
            if sub.empty:
                continue
            ax.errorbar([position[v] for v in sub["lam"]], sub["delta"],
                        yerr=sub["std"],
                        color=BLUE if kk == "full" else MUTED, marker=marker,
                        linewidth=2, markersize=6, capsize=3, elinewidth=1,
                        label=f"K={kk}")
        ax.set_xticks(list(position.values()), [f"λ={v:g}" for v in lam_order])
        ax.set_xlim(-0.3, len(lam_order) - 0.7)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{dataset} · {ENCODER_LABEL[encoder]}")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower right")

    axes[0].set_ylabel("change vs. the Stage 1 probe (pp)")
    fig.suptitle(
        "End-to-end training needs the displacement penalty: without it the flow "
        "overfits and ends up worse than the classifier it started from",
        fontsize=10, color=INK, y=1.04,
    )
    save(fig, "regularisation.png")


def main() -> None:
    apply_style()
    print("figures:")
    fig_accuracy()
    fig_curves()
    for dataset, encoder in config.STAGE3_PIPELINES:
        fig_feature_space(dataset, encoder)
    fig_regularisation()
    print("\nThe Stage 3 accuracy table is generated by `make table`.")


if __name__ == "__main__":
    main()
