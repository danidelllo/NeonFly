"""Bounded RND on local geometry, with predictor replay across episodes."""
import numpy as np
import torch
from torch import nn

from python.vision.range_sensor import geometry_vector


class GeometryNovelty:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        devices = [torch.cuda.current_device()] if device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg["seed"]+101)
            self.target = nn.Sequential(nn.Linear(130, 128), nn.Tanh(), nn.Linear(128, 32)).to(device)
            self.predictor = nn.Sequential(nn.Linear(130, 128), nn.Tanh(), nn.Linear(128, 32)).to(device)
        self.target.requires_grad_(False).eval()
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)
        self.buffer = np.zeros((cfg["rnd_replay_capacity"], 130), np.float32)
        self.size = self.cursor = self.calibration_n = 0
        self.calibration_sum = 0.
        self.rng = np.random.default_rng(cfg["seed"]+102)
        self.begin_episode()

    def begin_episode(self):
        self.spent = 0.

    def reward(self, game):
        if not game.get("rays", {}).get("insideView", True):
            return 0.
        features = geometry_vector(game)
        x = torch.as_tensor(features, device=self.device)
        with torch.no_grad():
            error = float((self.predictor(x)-self.target(x)).square().mean())
        if self.calibration_n < 256:
            self.calibration_sum += error
            self.calibration_n += 1
        baseline = max(self.calibration_sum/max(self.calibration_n, 1), 1e-6)
        reward = min(self.cfg["rnd_step_cap"], self.cfg["rnd_coefficient"]*error/baseline,
                     max(0., self.cfg["rnd_episode_cap"]-self.spent))
        self.spent += reward
        self.buffer[self.cursor] = features
        self.cursor = (self.cursor+1) % len(self.buffer)
        self.size = min(self.size+1, len(self.buffer))
        indices = self.rng.integers(self.size, size=min(32, self.size))
        batch = torch.as_tensor(self.buffer[indices], device=self.device)
        with torch.no_grad():
            target = self.target(batch)
        loss = (self.predictor(batch)-target).square().mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1., error_if_nonfinite=True)
        self.opt.step()
        if not np.isfinite(reward):
            raise FloatingPointError("invalid novelty reward")
        return float(reward)

    def state_dict(self):
        return dict(target=self.target.state_dict(), predictor=self.predictor.state_dict(), optimizer=self.opt.state_dict(),
                    buffer=torch.from_numpy(self.buffer.copy()), size=self.size, cursor=self.cursor,
                    calibration_n=self.calibration_n, calibration_sum=self.calibration_sum,
                    rng=self.rng.bit_generator.state, spent=self.spent)

    def load_state_dict(self, state):
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.opt.load_state_dict(state["optimizer"])
        self.buffer[:] = state["buffer"].numpy()
        for key in ("size", "cursor", "calibration_n", "calibration_sum", "spent"):
            setattr(self, key, state[key])
        self.rng.bit_generator.state = state["rng"]
