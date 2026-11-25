"""Readouts only. No parameters belong to the fixed connectome."""
import torch
from torch import nn


class NonlinearReadout(nn.Module):
    def __init__(self, width, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 128), nn.Tanh())
        self.actor = nn.Linear(128, 36)
        self.critic = nn.Linear(128, 1)
        for module in self.body:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 2**.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1.)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        z = self.body(x)
        return torch.cat((self.actor(z), self.critic(z)), dim=-1)


def make_head(width, cfg, device):
    if cfg.get("hidden", 0):
        return NonlinearReadout(width, cfg["hidden"]).to(device)
    head = nn.Linear(width, 37).to(device)
    with torch.no_grad():
        nn.init.orthogonal_(head.weight[:36], .01)
        nn.init.orthogonal_(head.weight[36:], 1.)
        head.bias.zero_()
    return head
