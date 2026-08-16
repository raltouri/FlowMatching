"""The four Stage 2 figures, all derived from results/ and features/.

    python -m src.figures_stage2

Palette and chart style are shared with Stage 1 so the two reports read as one
document. Training mode carries hue and the number of Euler steps carries line
style, which keeps four methods plus a baseline legible with two hues.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import functional as F

from src import config, data, evaluate, flow, heads, utils
from src.figures_stage1 import (
    AQUA,
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
from src.train import subset_indices

# Mode is the hue, T is the line style — two channels rather than four hues.
METHOD_STYLE = {
    "fm_std_T4": (BLUE, "-"),
    "fm_std_T12": (BLUE, "--"),
    "fm_roll_T4": (ORANGE, "-"),
    "fm_roll_T12": (ORANGE, "--"),
}

# The pipeline the qualitative figures use: the one where flow matching has the
# largest effect, so the geometry is actually visible.
REPRESENTATIVE = ("aircraft", "dinov2_vits14", "full", 0)
TRAJECTORY_METHOD = "fm_std_T12"  # 12 steps draws a more readable path than 4
TRAJECTORY_CLASSES = 4
TRAJECTORY_PER_CLASS = 3


def save(fig, name: str) -> None:
    config.FIGURES_STAGE2.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_STAGE2 / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(config.ROOT)}")


def load_velocity(dataset: str, encoder: str, method: str, k, seed: int):
    net = flow.VelocityMLP(config.FEATURE_DIM[encoder])
    tag = config.run_tag(dataset, encoder, method, k, seed)
    net.load_state_dict(torch.load(config.CKPT / f"{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def prototypes_for(dataset: str, encoder: str, k, seed: int) -> torch.Tensor:
    """Rebuilt from the same K-shot subset the run trained on."""
    z, y = utils.load_cache(dataset, encoder, "train")
    idx = subset_indices(y.numpy(), k, seed)
    return heads.build_prototypes(z[idx], y[idx], config.NUM_CLASSES[dataset])


# --- Figure 1: accuracy versus K, baseline and all four variants --------


def fig_accuracy_vs_k() -> None:
    stats = evaluate.aggregate()
    x = np.arange(len(config.SHOTS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    for ax, (dataset, encoder) in zip(axes, config.PIPELINES):
        subset = stats[(stats["dataset"] == dataset) & (stats["encoder"] == encoder)]

        base = subset[subset["head"] == "prototypes"].sort_values("K")
        ax.errorbar(
            x, base["mean"], yerr=base["std"].fillna(0.0), color=MUTED,
            linewidth=2, marker="o", markersize=5, capsize=3, elinewidth=1,
        )
        for method, (color, style) in METHOD_STYLE.items():
            group = subset[subset["head"] == method].sort_values("K")
            ax.errorbar(
                x, group["mean"], yerr=group["std"], color=color, linestyle=style,
                linewidth=2, marker="o", markersize=4, capsize=3, elinewidth=1,
            )
        ax.set_title(f"{dataset} · {ENCODER_LABEL[encoder]}")
        ax.set_xticks(x, [str(k) for k in config.SHOTS])
        ax.set_xlabel("labelled images per class (K)")
        ax.set_xlim(-0.25, len(x) - 0.75)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("top-1 accuracy (%)")

    handles = [
        Line2D([], [], color=MUTED, linewidth=2, label="prototype baseline (Stage 1)"),
        *(
            Line2D([], [], color=c, linestyle=s, linewidth=2, label=m)
            for m, (c, s) in METHOD_STYLE.items()
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        "Flow matching helps most where the prototype baseline was weakest; "
        "standard training beats rolled-out everywhere",
        fontsize=11, color=INK, y=1.03,
    )
    save(fig, "accuracy_vs_k.png")


# --- Figure 2: training-loss curves -------------------------------------


def fig_loss_curves() -> None:
    dataset, encoder, k, seed = REPRESENTATIVE
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for ax, mode in zip(axes, ("std", "roll")):
        for steps, style in ((4, "-"), (12, "--")):
            method = f"fm_{mode}_T{steps}"
            tag = config.run_tag(dataset, encoder, method, k, seed)
            curve = json.loads((config.CURVES / f"{tag}.json").read_text())["train_loss"]
            ax.plot(
                np.arange(1, len(curve) + 1), curve,
                color=BLUE if mode == "std" else ORANGE,
                linestyle=style, linewidth=2, label=f"T={steps}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title("standard FM" if mode == "std" else "rolled-out FM")
        ax.legend(loc="upper right")

    axes[0].set_ylabel("training loss (log scale)")
    fig.suptitle(
        f"{dataset} · {ENCODER_LABEL[encoder]}, K={k}, seed {seed}: both objectives train "
        "stably.\nThe two losses measure different quantities and are not comparable in "
        "value — only in shape.",
        fontsize=10, color=INK, y=1.10,
    )
    save(fig, "loss_curves.png")


# --- Figure 3: the feature space before and after the flow --------------


def fig_feature_space() -> None:
    dataset, encoder, k, seed = REPRESENTATIVE
    classes = viz_classes(dataset)
    names = data.cached_class_names(dataset)

    z_test, y_test = utils.load_cache(dataset, encoder, "test")
    prototypes = prototypes_for(dataset, encoder, k, seed)
    keep = np.isin(y_test.numpy(), classes)
    points = F.normalize(z_test[keep], dim=1) if config.FM_NORMALIZE else z_test[keep]
    labels = y_test[keep].numpy()

    views = {"original features": points}
    for mode in ("std", "roll"):
        method = f"fm_{mode}_T4"
        net = load_velocity(dataset, encoder, method, k, seed)
        with torch.no_grad():
            views[f"after {method}"] = flow.rollout(net, points, 4)

    # One projection fitted over every view and the prototypes together, so the
    # three panels share a single low-dimensional frame and can be compared.
    joint = torch.cat([*views.values(), prototypes[classes]]).numpy()
    embedded = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=0
    ).fit_transform(joint)

    n = len(points)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (title, offset) in zip(axes, [(t, i * n) for i, t in enumerate(views)]):
        xy = embedded[offset : offset + n]
        proto_xy = embedded[3 * n :]
        for slot, cls in enumerate(classes):
            color = CLASS_COLORS[slot]
            ax.scatter(*xy[labels == cls].T, s=13, alpha=0.55, linewidths=0, color=color)
            ax.scatter(
                *proto_xy[slot], s=190, color=color, edgecolors=SURFACE,
                linewidths=2, zorder=3,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)

    handles = [
        Line2D(
            [], [], color=CLASS_COLORS[s], marker="o", linestyle="",
            markersize=6, label=names[c],
        )
        for s, c in enumerate(classes)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        f"{dataset} · {ENCODER_LABEL[encoder]}, K={k}: the flow pulls each class toward its "
        "prototype\nSame test images and one jointly fitted t-SNE across all three panels. "
        "Qualitative only.",
        fontsize=10, color=INK, y=1.06,
    )
    save(fig, "feature_space.png")


# --- Figure 4: flow trajectories ----------------------------------------


def fig_trajectories() -> None:
    dataset, encoder, k, seed = REPRESENTATIVE
    steps = int(TRAJECTORY_METHOD.split("_T")[1])
    classes = viz_classes(dataset)[:TRAJECTORY_CLASSES]
    names = data.cached_class_names(dataset)

    z_test, y_test = utils.load_cache(dataset, encoder, "test")
    prototypes = prototypes_for(dataset, encoder, k, seed)
    net = load_velocity(dataset, encoder, TRAJECTORY_METHOD, k, seed)

    chosen, chosen_labels = [], []
    for cls in classes:
        idx = np.flatnonzero(y_test.numpy() == cls)[:TRAJECTORY_PER_CLASS]
        chosen.append(z_test[idx])
        chosen_labels += [cls] * len(idx)
    z_sel = torch.cat(chosen)
    if config.FM_NORMALIZE:
        z_sel = F.normalize(z_sel, dim=1)

    with torch.no_grad():
        _, path = flow.rollout(net, z_sel, steps, return_path=True)

    # PCA, as the spec recommends: it is a linear map, so a straight path in
    # feature space stays straight on the page and the geometry is readable.
    flat = path.reshape(-1, path.shape[-1]).numpy()
    pca = PCA(n_components=2, random_state=0).fit(
        np.concatenate([flat, prototypes[classes].numpy()])
    )
    path_2d = pca.transform(flat).reshape(path.shape[0], path.shape[1], 2)
    proto_2d = pca.transform(prototypes[classes].numpy())

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for i, cls in enumerate(chosen_labels):
        slot = list(classes).index(cls)
        color = CLASS_COLORS[slot]
        xy = path_2d[i]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1, alpha=0.7, zorder=1)
        ax.scatter(*xy.T, s=9, color=color, alpha=0.6, linewidths=0, zorder=2)
        ax.scatter(*xy[0], s=45, color=color, marker="o", edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
        ax.scatter(*xy[-1], s=70, color=color, marker="X", edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
    for slot, cls in enumerate(classes):
        ax.scatter(*proto_2d[slot], s=260, color=CLASS_COLORS[slot],
                   marker="*", edgecolors=SURFACE, linewidths=2, zorder=4)

    handles = [
        Line2D([], [], color=CLASS_COLORS[s], marker="o",
               linestyle="-", markersize=6, label=names[c])
        for s, c in enumerate(classes)
    ] + [
        Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=6,
               label="start (original feature)"),
        Line2D([], [], color=MUTED, marker="X", linestyle="", markersize=8,
               label="end (after the flow)"),
        Line2D([], [], color=MUTED, marker="*", linestyle="", markersize=12,
               label="class prototype"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    ax.set_title(
        f"{dataset} · {ENCODER_LABEL[encoder]}, {TRAJECTORY_METHOD}, K={k}\n"
        f"{steps} Euler steps per example, projected with PCA fitted jointly over all "
        "states and prototypes",
        fontsize=10,
    )
    save(fig, "trajectories.png")


def main() -> None:
    apply_style()
    print("figures:")
    fig_accuracy_vs_k()
    fig_loss_curves()
    fig_feature_space()
    fig_trajectories()
    print("\nThe Stage 2 accuracy table is generated by `make table`.")


if __name__ == "__main__":
    main()
