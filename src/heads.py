"""The two classifier heads — the only parts of the project that are trained.

Both are deliberately tiny: the linear probe is a single nn.Linear, and the
prototype head has no parameters at all.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_prototypes(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One L2-normalised prototype per class.

    Normalising twice is deliberate. The inner normalise makes every image
    contribute equally to its class mean regardless of feature magnitude; the
    outer one puts each prototype back on the unit sphere, so comparing with a
    dot product is exactly cosine similarity.
    """
    zn = F.normalize(z, dim=1)
    means = torch.stack([zn[y == c].mean(0) for c in range(num_classes)])
    return F.normalize(means, dim=1)


def prototype_logits(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between each feature and every prototype."""
    return F.normalize(z, dim=1) @ prototypes.T


def linear_probe(in_dim: int, num_classes: int) -> nn.Linear:
    """s = Wz + b. W and b are the only trained parameters in the project."""
    return nn.Linear(in_dim, num_classes)
