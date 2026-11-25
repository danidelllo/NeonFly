import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from python.communication.observe import request
from python.vision.frames import VisualStack


@dataclass(frozen=True)
class ShipConfig:
    action_ticks: int = 4
    max_steps: int = 256
    resolution: int = 84
    stack: int = 4
    rgb: bool = False
    normalize: bool = False
    crop: tuple[int, int, int, int] | None = None
    living_reward_per_applied_tick: float = 0.01
    survival_per_second: float = 0.0
    survival_cap_seconds: float = 50.0
    death_reward: float = -1.0
    completion_reward: float = 1.0
    controlled: bool = False
    damage_penalty: float = 0.0
    max_seconds: float = 0.0
    sensors: bool = False

    def __post_init__(self):
        for value, low, high in ((self.action_ticks, 1, 32), (self.max_steps, 1, 1_000_000),
                                 (self.stack, 1, 16), (self.resolution, 8, 512)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"expected integer in [{low}, {high}], got {value!r}")
        if any(type(v) is not bool for v in (self.rgb, self.normalize, self.controlled, self.sensors)):
            raise ValueError("rgb and normalize must be booleans")
        if self.crop is not None and (len(self.crop) != 4 or any(type(i) is not int for i in self.crop)):
            raise ValueError("crop must contain four integer pixel coordinates")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (
                self.living_reward_per_applied_tick, self.death_reward, self.completion_reward,
                self.survival_per_second, self.survival_cap_seconds, self.damage_penalty, self.max_seconds)):
            raise ValueError("reward coefficients must be finite")
        if self.survival_per_second < 0 or self.survival_cap_seconds <= 0 or (
                self.survival_per_second and self.living_reward_per_applied_tick):
            raise ValueError("choose only one survival reward clock and a positive cap")
        if self.damage_penalty < 0 or self.max_seconds < 0:
            raise ValueError("damage penalty and time limit must be nonnegative")

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream:
            return cls(**tomllib.load(stream))


class ShipEnv(gym.Env):
    """Live, single-instance Level_1 control. No game launch or implicit action retries."""

    def __init__(self, config=ShipConfig()):
        self.config = config
        self.action_space = gym.spaces.Discrete(36)  # X, Y, primary, rocket; see docs/CONTROLS.md
        self.observation_space = gym.spaces.Box(
            0, 1 if config.normalize else 255,
            (config.stack * (3 if config.rgb else 1), config.resolution, config.resolution),
            np.float32 if config.normalize else np.uint8)
        self.visual = VisualStack(config.resolution, config.stack, config.rgb, config.crop, config.normalize)
        self.active = False
        self._key_path = Path(os.environ["LOCALAPPDATA"]) / "NeonFly" / "session.key"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.active = False
        self._key = self._key_path.read_bytes()
        before, _ = request(1)
        if self.config.controlled and before.get("bridgeVersion") not in ("0.6.0", "0.7.0", "0.8.0"):
            raise RuntimeError("controlled training requires NeonFly 0.6, 0.7 or 0.8")
        if self.config.survival_per_second and "physicsSeconds" not in before:
            raise RuntimeError("simulation-time reward requires NeonFly 0.5")
        if before.get("shipActionCount") != self.action_space.n:
            raise RuntimeError("full ship controls require NeonFly plugin 0.4 or compatible")
        if before["scene"] != "Level_1" and not (self.config.controlled and getattr(self, "completed", False)):
            raise RuntimeError(f"expected Level_1, found {before['scene']}; refusing to reset another scene")
        state, png = request(10 if self.config.controlled else 6)
        if (state["scene"] != "Level_1" or before["sceneHandle"] == state["sceneHandle"]
                or state["controller"] != "Player" or state["gameOver"]
                or (state["timeScale"] != 0 if self.config.controlled else state["timeScale"] <= 0)
                or self._key != self._key_path.read_bytes()):
            raise RuntimeError("reset did not produce a fresh, active Level_1 ship")
        self.origin = state
        if self.config.sensors:
            from python.vision.range_sensor import sensor_vector
            sensor_vector(state)
        self.last_fixed = state["fixedUpdates"]
        self.steps = 0
        self.survived = 0.0
        self.damage = state.get("damageFraction", 0.)
        if self.config.damage_penalty and ("damageFraction" not in state or state.get("maxHealth", 0) <= 0):
            raise RuntimeError("damage reward requires verified health telemetry")
        self.completed = False
        obs = self.visual.push(png, reset=True)
        self.active = True
        return obs, {"game": state}

    def _events(self, state):
        deaths = state["deaths"] - self.origin["deaths"]
        completions = state["completions"] - self.origin["completions"]
        if min(deaths, completions) < 0 or (deaths and completions):
            raise RuntimeError("ambiguous or reset event counters")
        if state["sceneHandle"] != self.origin["sceneHandle"] and not completions:
            raise RuntimeError("unexpected scene change")
        return bool(deaths or state["gameOver"]), bool(completions)

    def step(self, action):
        if not self.active:
            raise gym.error.ResetNeeded("reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid ship action: {action!r}")
        # An uncertain write must not be replayed or followed by another step.
        self.active = False
        if self._key != self._key_path.read_bytes():
            raise RuntimeError("game restarted; reset required")
        guard = tuple(self.origin[k] for k in ("sceneHandle", "deaths", "completions"))
        state, png = request(9 if self.config.controlled else 8, int(action), self.config.action_ticks, guard)
        if self.config.controlled and (not state.get("controlled") or state["timeScale"] != 0):
            raise RuntimeError("game was not held at the observation boundary")
        ticks = state["appliedTicks"]
        if self._key != self._key_path.read_bytes():
            raise RuntimeError("game restarted during step")
        died, completed = self._events(state)
        if self.config.sensors and not (died or completed):
            from python.vision.range_sensor import sensor_vector
            try:
                sensor_vector(state)
            except ValueError as error:
                raise RuntimeError(f"range sensor unavailable: controller={state['controller']}, position=({state['x']},{state['y']}), scan={state.get('rays')}") from error
        terminated = died or completed
        elapsed = state["fixedUpdates"] - self.last_fixed
        if elapsed < ticks:
            raise RuntimeError("physics clock inconsistent with action acknowledgement")
        self.last_fixed = state["fixedUpdates"]
        if not 0 <= ticks <= self.config.action_ticks or (not terminated and ticks != self.config.action_ticks):
            raise RuntimeError("incomplete action without terminal event")
        self.steps += 1
        truncated = not terminated and self.steps >= self.config.max_steps
        reward = (ticks * self.config.living_reward_per_applied_tick + died * self.config.death_reward
                  + completed * self.config.completion_reward)
        if "physicsSeconds" in self.origin:
            clock = "deathSeconds" if died else "completionSeconds" if completed else "physicsSeconds"
            seconds = state[clock] - self.origin["physicsSeconds"]
            if not math.isfinite(seconds) or seconds < self.survived - 1e-5:
                raise RuntimeError("invalid survival clock")
            cap = self.config.survival_cap_seconds
            reward += self.config.survival_per_second * (min(seconds, cap) - min(self.survived, cap))
            self.survived = seconds
        damage = state.get("damageFraction", self.damage)
        if not math.isfinite(damage) or damage < self.damage:
            raise RuntimeError("invalid damage counter")
        damage_delta = damage - self.damage
        reward -= self.config.damage_penalty * damage_delta
        self.damage = damage
        truncated = not terminated and (truncated or bool(self.config.max_seconds and self.survived >= self.config.max_seconds))
        self.completed = completed
        reason = "death" if died else "completion" if completed else "time_limit" if truncated else None
        obs = self.visual.push(png)
        self.active = not (terminated or truncated)
        return obs, float(reward), terminated, truncated, {
            "game": state, "reason": reason, "applied_ticks": ticks, "elapsed_ticks": elapsed,
            "survived_seconds": self.survived, "damage_fraction": damage - self.origin.get("damageFraction", 0.),
            "reward_terms": dict(survival=reward - (died*self.config.death_reward + completed*self.config.completion_reward)
                                 + self.config.damage_penalty*damage_delta,
                                 damage=-self.config.damage_penalty*damage_delta,
                                 terminal=died*self.config.death_reward + completed*self.config.completion_reward)}
