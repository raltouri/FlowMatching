"""Every tunable number for Stage 1, in one place.

Deliberately plain constants rather than a config file: there is exactly one
configuration, and indirection would only put these values further from the
code that reads them.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
CKPT = RESULTS / "ckpt"
CURVES = RESULTS / "curves"
FIGURES = ROOT / "figures"
FIGURES_STAGE1 = FIGURES / "stage1"
FIGURES_STAGE2 = FIGURES / "stage2"
RUNS_CSV = RESULTS / "runs.csv"

# --- Protocol (fixed by the spec) ---------------------------------------

DATASETS = ("dtd", "aircraft")          # DTD partition 1; Aircraft variant level
ENCODERS = ("resnet18", "dinov2_vits14")
SHOTS = (5, 10, "full")
SEEDS = (0, 1, 2)                       # subset seeds at K=5/10, init seeds at K=full

# The six pipelines: (dataset, encoder). Each is evaluated with both heads.
# ResNet-18 on both datasets; DINOv2 on Aircraft only (see PLAN.md §1).
PIPELINES = (
    ("dtd", "resnet18"),
    ("aircraft", "resnet18"),
    ("aircraft", "dinov2_vits14"),
)

NUM_CLASSES = {"dtd": 47, "aircraft": 100}
FEATURE_DIM = {"resnet18": 512, "dinov2_vits14": 384}

# --- Linear probe (spec's suggested configuration) ----------------------

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
# Checkpoint selection: highest validation accuracy.

# --- Stage 2: flow matching ---------------------------------------------

FM_HIDDEN = 512                         # two hidden layers of this width, SiLU
FM_STEPS = (4, 12)                      # Euler steps T, evaluated at both
FM_LR = 1e-3
FM_EPOCHS = 200
FM_BATCH_SIZE = 64
# The flow operates on L2-normalised features. Prototypes are unit vectors
# while raw features have norms of roughly 9-54, so on raw features the target
# velocity p - z would be dominated by shrinking the magnitude — which the
# cosine decision rule ignores entirely. Normalising both sides puts the whole
# of the learning problem into direction, which is the only thing classified.
FM_NORMALIZE = True

# --- Feature extraction -------------------------------------------------

IMAGE_SIZE = 224                        # divisible by 14, required by ViT-S/14
EXTRACT_BATCH_SIZE = 64
# Every FGVC-Aircraft image carries a copyright banner along the bottom;
# it is cropped before the encoder's own transform (see PLAN.md §6).
AIRCRAFT_BANNER_PX = 20

# --- Figures ------------------------------------------------------------

VIZ_NUM_CLASSES = 9                     # readable subset for the 2-D projections


def cache_path(dataset: str, encoder: str, split: str):
    """Location of one feature cache."""
    return FEATURES / f"{dataset}_{encoder}_{split}.pt"


def run_tag(dataset: str, encoder: str, method: str, k, seed: int) -> str:
    """Filename stem for one run's checkpoint and loss curve.

    The method is part of the name so Stage 2's flow-matching runs cannot
    overwrite Stage 1's linear probes.
    """
    return f"{dataset}_{encoder}_{method}_K{k}_s{seed}"


def classes_path(dataset: str):
    """Class names, written at extraction time.

    Kept beside the caches so that everything below the L3 boundary — including
    the figures, which label axes and legends — works without the images.
    """
    return FEATURES / f"{dataset}_classes.json"
