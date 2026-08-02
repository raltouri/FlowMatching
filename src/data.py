"""Official dataset splits and the balanced K-shot sampler.

The protocol is expressed here once — DTD partition 1, Aircraft variant level,
official splits never merged — so no other module can get it wrong.

Run `python -m src.data --verify` to check the splits and the sampler.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from torchvision import datasets

from src import config

# Published split sizes, asserted by verify(). These are the values that catch
# a wrong DTD partition or Aircraft annotation level, which would otherwise
# only surface as quietly mediocre accuracies much later.
EXPECTED_SIZES = {
    ("dtd", "train"): 1880,
    ("dtd", "val"): 1880,
    ("dtd", "test"): 1880,
    ("aircraft", "train"): 3334,
    ("aircraft", "val"): 3333,
    ("aircraft", "test"): 3333,
}

SPLITS = ("train", "val", "test")


def load_split(dataset: str, split: str, transform=None, download: bool = True):
    """Return one official split. Splits are returned as-is and never merged."""
    root = str(config.DATA)
    if dataset == "dtd":
        # partition=1 is torchvision's default, but the spec requires it, so it
        # is passed explicitly to keep the protocol visible in the code.
        return datasets.DTD(
            root, split=split, partition=1, transform=transform, download=download
        )
    if dataset == "aircraft":
        return datasets.FGVCAircraft(
            root,
            split=split,
            annotation_level="variant",
            transform=transform,
            download=download,
        )
    raise ValueError(f"unknown dataset: {dataset!r}")


def labels(ds) -> np.ndarray:
    """Label vector for a split, without decoding any image."""
    return np.asarray(ds._labels)


def class_names(ds) -> list[str]:
    """Class names in label-index order."""
    return list(ds.classes)


def cached_class_names(dataset: str) -> list[str]:
    """Class names from the sidecar written during extraction.

    The figures label with these rather than reopening the dataset, so anyone
    holding only the feature caches can regenerate them without the images.
    """
    return json.loads(config.classes_path(dataset).read_text())


def sample_k_shot(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Indices of k examples per class, drawn with the given seed.

    Classes holding fewer than k examples contribute all of theirs.
    """
    rng = np.random.default_rng(seed)
    picked = [
        rng.permutation(np.flatnonzero(y == c))[:k] for c in np.unique(y)
    ]
    return np.sort(np.concatenate(picked))


def verify() -> None:
    """M0/M1 gate: split sizes, class counts, and sampler invariants."""
    for dataset in config.DATASETS:
        print(f"\n{dataset}")
        train_y = None
        for split in SPLITS:
            ds = load_split(dataset, split)
            y = labels(ds)
            n_classes = len(class_names(ds))

            expected = EXPECTED_SIZES[(dataset, split)]
            assert len(ds) == expected, f"{dataset}/{split}: {len(ds)} != {expected}"
            assert n_classes == config.NUM_CLASSES[dataset], (
                f"{dataset}/{split}: {n_classes} classes, "
                f"expected {config.NUM_CLASSES[dataset]}"
            )
            assert len(y) == len(ds)

            per_class = np.bincount(y, minlength=n_classes)
            print(
                f"  {split:5s} {len(ds):5d} images  {n_classes:3d} classes  "
                f"{per_class.min():3d}-{per_class.max():3d} per class"
            )
            if split == "train":
                train_y = y

        # Sampler invariants (M1).
        for k in (5, 10):
            a = sample_k_shot(train_y, k, seed=0)
            assert np.array_equal(a, sample_k_shot(train_y, k, seed=0)), "not deterministic"
            assert not np.array_equal(a, sample_k_shot(train_y, k, seed=1)), "seeds identical"
            counts = np.bincount(train_y[a], minlength=config.NUM_CLASSES[dataset])
            assert counts.min() == counts.max() == k, f"unbalanced {k}-shot subset"
            assert len(np.unique(a)) == len(a), "duplicate indices"
        print(f"  sampler  deterministic, balanced, seed-sensitive  (k=5,10)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="run the M0/M1 gate")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        parser.print_help()
