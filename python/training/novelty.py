"""Geometry RND and episodic position coverage, with optional reward caps."""
import numpy as np
import torch
from torch import nn

from python.vision.range_sensor import geometry_vector, LEGACY_RAY_ENCODER


class GeometryNovelty:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        for key in ('rnd_step_cap', 'rnd_episode_cap'):
            if not np.isfinite(cfg[key]) or cfg[key] < 0:
                raise ValueError('novelty reward caps must be nonnegative; zero disables clipping')
        self.coverage_enabled = bool(cfg.get('episodic_coverage', False))
        self.ray_count = cfg.get('sensor_rays', 32 if cfg.get('encoder') == LEGACY_RAY_ENCODER else 126)
        if self.ray_count not in (32, 126):
            raise ValueError('unsupported novelty sensor layout')
        self.geometry_width = self.ray_count*4+2
        bins = cfg.get('coverage_bins', [12, 8])
        if len(bins) != 2 or any(type(n) is not int or not 2 <= n <= 128 for n in bins):
            raise ValueError('invalid coverage grid')
        self.coverage_bins = tuple(bins)
        self.visited = np.zeros(self.coverage_bins, dtype=bool)
        devices = [torch.cuda.current_device()] if device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg["seed"]+101)
            self.target = nn.Sequential(nn.Linear(self.geometry_width, 128), nn.Tanh(), nn.Linear(128, 32)).to(device)
            self.predictor = nn.Sequential(nn.Linear(self.geometry_width, 128), nn.Tanh(), nn.Linear(128, 32)).to(device)
        self.target.requires_grad_(False).eval()
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)
        self.buffer = np.zeros((cfg["rnd_replay_capacity"], self.geometry_width), np.float32)
        self.size = self.cursor = self.calibration_n = 0
        self.calibration_sum = 0.
        self.rng = np.random.default_rng(cfg["seed"]+102)
        self.begin_episode()

    def coverage_cell(self, scan):
        return tuple(min(n-1, max(0, int(np.floor(float(scan[key])*n))))
                     for key, n in zip(('screenX', 'screenY'), self.coverage_bins))

    def begin_episode(self, game=None):
        self.spent = 0.
        self.visited.fill(False)
        self.stats = dict(new_cells=0, coverage_reward=0., rnd_reward=0., blocked_bonus_steps=0)
        if self.coverage_enabled and game is not None:
            self.visited[self.coverage_cell(game['rays'])] = True

    def reward(self, game, *, allow_bonus=True):
        scan = game.get('rays', {})
        # With bridge 0.8, a ship just outside the viewport can still observe
        # visible geometry. Clamp its coverage position into the boundary cell;
        # do not create extra reward cells outside the screen.
        visible = any(end > start for end, start in zip(scan.get('visible', []),
                      scan.get('visibleStart', [0.]*self.ray_count)))
        if not scan.get('insideView', True) and not (self.coverage_enabled and visible):
            return 0.
        features = geometry_vector(game)
        x = torch.as_tensor(features, device=self.device)
        with torch.no_grad():
            error = float((self.predictor(x)-self.target(x)).square().mean())
        if self.calibration_n < 256:
            self.calibration_sum += error
            self.calibration_n += 1
        baseline = max(self.calibration_sum/max(self.calibration_n, 1), 1e-6)
        rnd_bonus = self.cfg['rnd_coefficient']*error/baseline
        coverage_bonus = 0.
        if self.coverage_enabled:
            cell = self.coverage_cell(scan)
            if not self.visited[cell]:
                self.visited[cell] = True
                self.stats['new_cells'] += 1
                coverage_bonus = self.cfg['coverage_cell_bonus']
        total = rnd_bonus+coverage_bonus
        if not np.isfinite(total):
            raise FloatingPointError('invalid novelty reward before clipping')
        reward = total if allow_bonus else 0.
        if self.cfg['rnd_step_cap'] > 0:
            reward = min(reward, self.cfg['rnd_step_cap'])
        if self.cfg['rnd_episode_cap'] > 0:
            reward = min(reward, max(0., self.cfg['rnd_episode_cap']-self.spent))
        if not allow_bonus:
            self.stats['blocked_bonus_steps'] += 1
        fraction = coverage_bonus/total if total > 0 else 0.
        self.stats['coverage_reward'] += reward*fraction
        self.stats['rnd_reward'] += reward*(1-fraction)
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
                    rng=self.rng.bit_generator.state, spent=self.spent,
                    coverage=torch.from_numpy(self.visited.copy()), coverage_stats=dict(self.stats))

    def load_state_dict(self, state):
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.opt.load_state_dict(state["optimizer"])
        self.buffer[:] = state["buffer"].numpy()
        for key in ("size", "cursor", "calibration_n", "calibration_sum", "spent"):
            setattr(self, key, state[key])
        self.rng.bit_generator.state = state["rng"]
        if 'coverage' in state and self.coverage_enabled:
            self.visited[:] = state['coverage'].numpy()
            self.stats = dict(state['coverage_stats'])
