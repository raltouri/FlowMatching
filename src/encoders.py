"""The two frozen encoders and their preprocessing.

Both are written out concretely rather than behind a registry: with exactly two
encoders, the whole truth fits on one screen.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms

from src import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ToRGB:
    """Force three channels. A handful of images in both datasets are greyscale."""

    def __call__(self, img):
        return img.convert("RGB")


class CropBottom:
    """Remove a fixed strip from the bottom of a PIL image.

    Every FGVC-Aircraft image carries a copyright banner there. Left in, the
    encoder spends part of its representation on text rather than the aircraft.
    """

    def __init__(self, px: int):
        self.px = px

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, max(1, h - self.px)))


def load_encoder(name: str):
    """Return (frozen encoder, its own preprocessing transform)."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        # Identity in place of fc exposes the 512-d representation that sits
        # before the classification layer.
        model.fc = nn.Identity()
        preprocess = weights.transforms()

    elif name == "dinov2_vits14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # DINOv2's published evaluation transform. forward() returns the final
        # class token, which is the 384-d representation we cache.
        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    else:
        raise ValueError(f"unknown encoder: {name!r}")

    # Frozen means both: eval() stops BatchNorm updating its running statistics,
    # requires_grad_(False) stops gradients being tracked at all.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess


def build_transform(preprocess, dataset: str):
    """Prepend the dataset-specific steps to an encoder's own transform."""
    steps = [ToRGB()]
    if dataset == "aircraft":
        steps.append(CropBottom(config.AIRCRAFT_BANNER_PX))
    steps.append(preprocess)
    return transforms.Compose(steps)
