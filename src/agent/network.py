import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(39, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, observation):
        device = self.layers[0].weight.device
        values = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if values.ndim not in (1, 2) or values.shape[-1] != 39:
            raise ValueError("Observation shape must be (39,) or (batch, 39)")
        if not torch.isfinite(values).all():
            raise ValueError("Observations must contain finite values")
        return self.layers(values)
