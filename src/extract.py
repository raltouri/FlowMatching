"""Build the feature caches — the only module that reads images.

This is the L3 boundary from PLAN.md: everything downstream reads these files
and never touches a pixel again.

    python -m src.extract            # build any missing caches
    python -m src.extract --force    # rebuild everything
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config, data
from src.encoders import build_transform, load_encoder


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def cache_path(dataset: str, encoder: str, split: str):
    return config.FEATURES / f"{dataset}_{encoder}_{split}.pt"


@torch.no_grad()
def extract_split(model, transform, dataset: str, split: str, device):
    ds = data.load_split(dataset, split, transform=transform, download=False)
    # shuffle=False is load-bearing: it keeps rows aligned with data.labels().
    loader = DataLoader(
        ds, batch_size=config.EXTRACT_BATCH_SIZE, shuffle=False, num_workers=4
    )
    feats, ys = [], []
    for x, y in tqdm(loader, desc=f"  {split}", leave=False):
        feats.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(feats), torch.cat(ys)


def main(force: bool = False) -> None:
    config.FEATURES.mkdir(exist_ok=True)
    device = pick_device()
    print(f"device: {device.type}")

    for dataset, encoder in config.PIPELINES:
        paths = {s: cache_path(dataset, encoder, s) for s in data.SPLITS}
        if not force and all(p.exists() for p in paths.values()):
            print(f"{dataset} / {encoder}: cached, skipping")
            continue

        print(f"{dataset} / {encoder}")
        model, preprocess = load_encoder(encoder)
        model.to(device)
        assert not model.training, "encoder not in eval mode"
        assert not any(p.requires_grad for p in model.parameters()), "encoder not frozen"
        transform = build_transform(preprocess, dataset)

        for split in data.SPLITS:
            z, y = extract_split(model, transform, dataset, split, device)

            assert z.shape == (len(y), config.FEATURE_DIM[encoder]), z.shape
            # Features must line up with the split's own label vector, in order.
            expected = torch.as_tensor(
                data.labels(data.load_split(dataset, split, download=False))
            )
            assert torch.equal(y, expected), f"{dataset}/{split}: labels misaligned"

            torch.save({"features": z, "labels": y}, paths[split])
            mb = paths[split].stat().st_size / 1e6
            print(f"  {split:5s} {tuple(z.shape)}  {mb:.1f} MB")

        del model

    print("\nCaches written to features/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild existing caches")
    main(**vars(parser.parse_args()))
