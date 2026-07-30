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
FIGURES = ROOT / "figures"
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

# --- Feature extraction -------------------------------------------------

IMAGE_SIZE = 224                        # divisible by 14, required by ViT-S/14
EXTRACT_BATCH_SIZE = 64
# Every FGVC-Aircraft image carries a copyright banner along the bottom;
# it is cropped before the encoder's own transform (see PLAN.md §6).
AIRCRAFT_BANNER_PX = 20

# --- Figures ------------------------------------------------------------

VIZ_NUM_CLASSES = 9                     # readable subset for the 2-D projections
