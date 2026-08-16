"""The five required figures, all derived from results/ and features/.

    python -m src.figures

Colours come from a validated categorical palette. These are static figures for
a printed report, so they deliberately commit to the light surface only.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from src import config, data, evaluate, heads, utils

# --- Palette ------------------------------------------------------------

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# Magnitude gets one hue, light to dark — never a rainbow.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues",
    [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

# Encoder carries hue, head carries line style. Two channels rather than four
# hues, so identity never rests on colour alone.
ENCODER_COLOR = {"resnet18": BLUE, "dinov2_vits14": ORANGE}
ENCODER_LABEL = {"resnet18": "ResNet-18", "dinov2_vits14": "DINOv2 ViT-S/14"}
HEAD_STYLE = {"linear": "-", "prototypes": "--"}

# Class identity is carried by hue alone: the eight validated palette slots, one
# per class. Marker shapes would separate them further under colour-vision
# deficiency, but were dropped at the supervisor's request; the legend carries
# the colour-to-class mapping instead.
CLASS_COLORS = (
    BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)

# The setting whose errors are worth showing: the best probe on each dataset.
REPRESENTATIVE = {"dtd": ("resnet18", "full", 0), "aircraft": ("dinov2_vits14", "full", 0)}
CURVE_SETTING = ("10", 0)  # representative 10-shot run for the loss curves


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # solid hairline; dashing reads as a threshold
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def save(fig, name: str) -> None:
    config.FIGURES_STAGE1.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_STAGE1 / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(config.ROOT)}")


# --- Figure 2: accuracy versus training-set size ------------------------


def fig_accuracy_vs_k() -> None:
    stats = evaluate.aggregate()
    x = np.arange(len(config.SHOTS))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)

    for ax, dataset in zip(axes, config.DATASETS):
        # The ledger is shared with Stage 2, so restrict to the Stage 1 heads.
        subset = stats[
            (stats["dataset"] == dataset) & (stats["head"].isin(HEAD_STYLE))
        ]
        for (encoder, head), group in subset.groupby(["encoder", "head"], observed=True):
            group = group.sort_values("K")
            ax.errorbar(
                x,
                group["mean"],
                yerr=group["std"].fillna(0.0),
                color=ENCODER_COLOR[encoder],
                linestyle=HEAD_STYLE[head],
                linewidth=2,
                marker="o",
                markersize=5,
                capsize=3,
                elinewidth=1,
            )
            # Label the endpoint only — never a number on every point.
            ax.annotate(
                f"{group['mean'].iloc[-1]:.1f}",
                (x[-1], group["mean"].iloc[-1]),
                textcoords="offset points",
                xytext=(7, -3),
                color=INK_2,
                fontsize=8,
            )
        chance = 100 / config.NUM_CLASSES[dataset]
        ax.axhline(chance, color=MUTED, linewidth=0.8)
        ax.annotate(
            f"chance {chance:.1f}%",
            (0, chance),
            textcoords="offset points",
            xytext=(0, 5),
            color=MUTED,
            fontsize=7,
        )
        ax.set_title(f"{dataset.upper() if dataset == 'dtd' else 'FGVC-Aircraft'}")
        ax.set_xticks(x, [str(k) for k in config.SHOTS])
        ax.set_xlabel("labelled images per class (K)")
        ax.set_xlim(-0.25, len(x) - 0.55)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("top-1 accuracy (%)")
    axes[0].set_ylim(0, 75)

    handles = [
        Line2D([], [], color=c, linewidth=2, label=ENCODER_LABEL[e])
        for e, c in ENCODER_COLOR.items()
    ] + [
        Line2D([], [], color=MUTED, linewidth=2, linestyle=s, label=h)
        for h, s in HEAD_STYLE.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Accuracy rises with K; the encoder matters more than the head",
        fontsize=11,
        color=INK,
        y=1.02,
    )
    save(fig, "accuracy_vs_k.png")


# --- Figure 3: training and validation loss curves ----------------------


def fig_loss_curves() -> None:
    k, seed = CURVE_SETTING
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    for ax, (dataset, encoder) in zip(axes, config.PIPELINES):
        curve = json.loads(
            (config.CURVES / f"{config.run_tag(dataset, encoder, 'linear', k, seed)}.json").read_text()
        )
        epochs = np.arange(1, len(curve["train_loss"]) + 1)
        ax.plot(epochs, curve["train_loss"], color=BLUE, linewidth=2, label="train")
        ax.plot(epochs, curve["val_loss"], color=ORANGE, linewidth=2, label="validation")

        best = int(np.argmax(curve["val_acc"])) + 1
        # Dashed here is correct: this is a threshold annotation, not a gridline.
        ax.axvline(best, color=MUTED, linewidth=0.8, linestyle="--")
        ax.annotate(
            f"best val acc\nepoch {best}",
            (best, ax.get_ylim()[1]),
            textcoords="offset points",
            xytext=(5, -18),
            color=MUTED,
            fontsize=7,
        )
        ax.set_title(f"{dataset} · {ENCODER_LABEL[encoder]}")
        ax.set_xlabel("epoch")

    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend(loc="upper right")
    fig.suptitle(
        f"Linear probe, K={k}, seed {seed}: training is stable; the train–validation "
        "gap shows where overfitting begins",
        fontsize=11,
        color=INK,
        y=1.06,
    )
    save(fig, "loss_curves.png")


# --- Figure 4: confusion matrices ---------------------------------------


def probe_logits(dataset: str, encoder: str, k, seed: int):
    """Restore a saved probe checkpoint and score the test split with it."""
    z, y = utils.load_cache(dataset, encoder, "test")
    model = heads.linear_probe(config.FEATURE_DIM[encoder], config.NUM_CLASSES[dataset])
    model.load_state_dict(torch.load(config.CKPT / f"{config.run_tag(dataset, encoder, 'linear', k, seed)}.pt"))
    model.eval()
    with torch.no_grad():
        return model(z), y


def fig_confusion() -> None:
    lines = ["# Largest off-diagonal confusions\n"]

    for dataset, (encoder, k, seed) in REPRESENTATIVE.items():
        logits, y = probe_logits(dataset, encoder, k, seed)
        names = data.cached_class_names(dataset)
        cm = evaluate.confusion_matrix(logits, y, config.NUM_CLASSES[dataset])

        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        image = ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.grid(visible=False)
        ax.set_title(
            f"{dataset} · {ENCODER_LABEL[encoder]} · linear probe, K={k}\n"
            f"row-normalised; diagonal = per-class recall",
            fontsize=9,
        )
        bar = fig.colorbar(image, ax=ax, fraction=0.046)
        bar.set_label("fraction of the true class", color=INK_2, fontsize=8)
        bar.outline.set_visible(False)
        save(fig, f"confusion_{dataset}.png")

        # The matrix shows that errors are structured; this names them.
        off = cm.copy()
        np.fill_diagonal(off, 0)
        lines.append(f"\n## {dataset} · {ENCODER_LABEL[encoder]} · K={k}\n")
        lines.append("| true class | predicted as | fraction |")
        lines.append("|---|---|---:|")
        for flat in np.argsort(off, axis=None)[::-1][:8]:
            i, j = np.unravel_index(flat, off.shape)
            lines.append(f"| {names[i]} | {names[j]} | {off[i, j]:.2f} |")

    (config.RESULTS / "top_confusions.md").write_text("\n".join(lines) + "\n")
    print(f"  {(config.RESULTS / 'top_confusions.md').relative_to(config.ROOT)}")


# --- Figure 5: two-dimensional feature projections ----------------------


def viz_classes(dataset: str) -> np.ndarray:
    """Nine classes spread evenly through the label index.

    Evenly spaced rather than the first nine, so the subset is not biased toward
    one region of an alphabetical class list (on Aircraft, the first nine
    variants are all Boeing narrowbodies).
    """
    return np.linspace(
        0, config.NUM_CLASSES[dataset] - 1, config.VIZ_NUM_CLASSES
    ).astype(int)


def fig_feature_projection(dataset: str) -> None:
    classes = viz_classes(dataset)
    names = data.cached_class_names(dataset)
    encoders = [e for d, e in config.PIPELINES if d == dataset]

    fig, axes = plt.subplots(1, len(encoders), figsize=(5.2 * len(encoders), 4.6))
    axes = np.atleast_1d(axes)

    for ax, encoder in zip(axes, encoders):
        z_test, y_test = utils.load_cache(dataset, encoder, "test")
        z_train, y_train = utils.load_cache(dataset, encoder, "train")
        prototypes = heads.build_prototypes(
            z_train, y_train, config.NUM_CLASSES[dataset]
        )[classes]

        keep = np.isin(y_test.numpy(), classes)
        # Normalise the features too: prototypes live on the unit sphere, so
        # without this the projection would separate them by magnitude alone.
        points = torch.nn.functional.normalize(z_test[keep], dim=1)
        joint = torch.cat([points, prototypes]).numpy()

        # Fitted jointly to features and prototypes, as the projection must be.
        embedded = TSNE(
            n_components=2, perplexity=30, init="pca", random_state=0
        ).fit_transform(joint)
        xy, proto_xy = embedded[: len(points)], embedded[len(points) :]
        labels = y_test[keep].numpy()

        for slot, cls in enumerate(classes):
            color = CLASS_COLORS[slot]
            ax.scatter(
                *xy[labels == cls].T, s=14, alpha=0.55, linewidths=0, color=color
            )
            # The prototype gets a surface ring so it stays readable where it
            # overlaps its own cloud, and a direct label so the class can be
            # identified without relying on hue.
            ax.scatter(
                *proto_xy[slot], s=200, color=color, edgecolors=SURFACE,
                linewidths=2, zorder=3,
            )

        ax.set_title(ENCODER_LABEL[encoder])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)

    handles = [
        Line2D(
            [], [], color=CLASS_COLORS[s], marker="o", linestyle="",
            markersize=6, label=names[c],
        )
        for s, c in enumerate(classes)
    ] + [
        Line2D(
            [], [], color=MUTED, marker="o", linestyle="", markersize=12,
            markeredgecolor=SURFACE, markeredgewidth=2,
            label="prototype (enlarged, ringed, labelled)",
        )
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.02)
    )
    fig.suptitle(
        f"{dataset}: t-SNE of test features with image-derived prototypes "
        f"({config.VIZ_NUM_CLASSES} classes)\n"
        "Qualitative only — t-SNE distances between clusters are not faithful.",
        fontsize=10,
        color=INK,
        y=1.04,
    )
    save(fig, f"features_{dataset}.png")


def main() -> None:
    apply_style()
    print("figures:")
    fig_accuracy_vs_k()
    fig_loss_curves()
    fig_confusion()
    for dataset in config.DATASETS:
        fig_feature_projection(dataset)
    print("\nThe accuracy table is generated separately by `make table`.")


if __name__ == "__main__":
    main()
