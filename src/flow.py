"""Flow matching from image features to class prototypes.

The velocity network answers one question: given a point `z` in feature space
and how far through the journey we are (`t`, from 0 to 1), which direction
should the point move, and how fast? Training shapes that field of directions
so it carries every feature toward its own class prototype.

Two training objectives, differing only in what they measure:

- `fm_loss` supervises the velocity at a random point on the straight line from
  the feature to its prototype. Cheap — one network evaluation per example —
  but it only ever sees points on the ideal path, while inference visits states
  the network generated itself.
- `rollout_loss` runs the full inference procedure during training and measures
  only where the point ended up. It sees exactly the self-generated states that
  inference produces, at roughly `steps` times the cost.

Nothing here modifies the cached features. The transported position is computed
on demand and discarded; what persists is the trained velocity field.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src import config


class VelocityMLP(nn.Module):
    """v_theta(z, t): position and time in, velocity out.

    Deliberately small, per the spec — the point of Stage 2 is the comparison
    between training objectives, not an architecture search.
    """

    def __init__(self, dim: int, hidden: int = config.FM_HIDDEN):
        super().__init__()
        # Time enters as one extra input dimension alongside the feature, so the
        # network can behave differently early and late in the journey.
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        return self.net(torch.cat([z, t.unsqueeze(1)], dim=1))


def fm_loss(net: VelocityMLP, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Standard flow matching.

    For each example draw t ~ U(0,1), step to z_t = (1-t)z + t*p on the straight
    line between the feature and its prototype, and regress the predicted
    velocity onto the direction of that line, u = p - z.

    Note the target does not depend on t: along a straight path the correct
    velocity is the same everywhere, so the network is being taught a constant
    answer per example but must infer it from position alone.
    """
    t = torch.rand(len(z), device=z.device)
    z_t = (1 - t).unsqueeze(1) * z + t.unsqueeze(1) * p
    return F.mse_loss(net(z_t, t), p - z)


def rollout(
    net: VelocityMLP, z: torch.Tensor, steps: int, return_path: bool = False
):
    """Transport `z` with `steps` Euler steps: ask, move a little, repeat.

    Gradients flow through the whole sequence, which is what makes rolled-out
    training possible — so this is deliberately not wrapped in `no_grad`.
    Callers doing inference should use `torch.no_grad()` themselves.

    With `return_path`, also returns every intermediate state as
    (batch, steps + 1, dim), starting at `z` and ending at the final position.
    The trajectory figure needs these; inference needs only the endpoint.
    """
    state = z
    path = [z]
    for k in range(steps):
        t = torch.full((len(z),), k / steps, device=z.device)
        state = state + net(state, t) / steps
        path.append(state)
    return (state, torch.stack(path, dim=1)) if return_path else state


def rollout_loss(
    net: VelocityMLP, z: torch.Tensor, p: torch.Tensor, steps: int
) -> torch.Tensor:
    """Rolled-out flow matching: run the whole journey, grade only the arrival.

    Backpropagates through all `steps` velocity evaluations at once. Use the
    same `steps` for training and inference — a network trained at T=4 and
    evaluated at T=12 is a different experiment.
    """
    return F.mse_loss(rollout(net, z, steps), p)
