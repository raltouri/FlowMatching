"""Seeding, cache loading, and the results ledger.

results/runs.csv is append-only: one row per run, written the moment it
finishes. Every table and figure is derived from it, so no number in the
report is ever transcribed by hand.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timezone

import numpy as np
import torch

from src import config

RUN_FIELDS = (
    "dataset",
    "encoder",
    "head",
    "K",
    "seed",
    "top1",
    "n_train",
    "epochs_run",
    "best_epoch",
    "timestamp",
)

# Identifies a run, so re-running is idempotent and group members can share
# results/runs.csv without repeating each other's work.
RUN_KEY = ("dataset", "encoder", "head", "K", "seed")


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cache(dataset: str, encoder: str, split: str):
    """Return (features, labels) for one split."""
    blob = torch.load(config.cache_path(dataset, encoder, split))
    return blob["features"], blob["labels"]


def completed_runs() -> set[tuple[str, ...]]:
    """Keys of runs already in the ledger."""
    if not config.RUNS_CSV.exists():
        return set()
    with open(config.RUNS_CSV, newline="") as f:
        return {tuple(row[k] for k in RUN_KEY) for row in csv.DictReader(f)}


def append_run(**row) -> None:
    config.RESULTS.mkdir(exist_ok=True)
    row.setdefault("epochs_run", "")
    row.setdefault("best_epoch", "")
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    is_new = not config.RUNS_CSV.exists()
    with open(config.RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
