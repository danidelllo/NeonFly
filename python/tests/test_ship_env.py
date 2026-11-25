import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from PIL import Image

from python.environment.ship import ShipConfig, ShipEnv
from python.vision.frames import VisualStack


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, "PNG")
    return stream.getvalue()


class VisualTests(unittest.TestCase):
    def test_stack_order_and_reset_isolation(self):
        stack = VisualStack(8, 2, rgb=True, normalize=True)
        stack.push(png("red"), reset=True)
        frame = stack.push(png("blue"))
        self.assertEqual(frame.dtype, np.float32)
        np.testing.assert_array_equal(frame[:, 0, 0], [1, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(stack.push(png("white"), reset=True), np.ones_like(frame))

    def test_invalid_crop_and_toml(self):
        with self.assertRaises(ValueError):
            VisualStack(8, 2, crop=(0, 0, 18, 9)).push(png("white"), reset=True)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "invalid.toml"
            config.write_text("action_ticks = 0")
            with self.assertRaises(ValueError):
                ShipConfig.load(config)


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        base = {"scene": "Level_1", "sceneHandle": 2, "deaths": 3, "completions": 0,
                "timeScale": 1, "controller": "Player", "gameOver": False, "fixedUpdates": 100,
                "shipActionCount": 36}
        self.base = base
        self.env = ShipEnv(ShipConfig(max_steps=1))
        self.key = tempfile.TemporaryDirectory()
        self.addCleanup(self.key.cleanup)
        self.env._key_path = Path(self.key.name) / "key"
        self.env._key_path.write_bytes(b"x" * 32)
        self.wire = self.enterContext(patch("python.environment.ship.request"))
        self.wire.side_effect = [({**base, "sceneHandle": 1}, b""), (base, png("black"))]
        self.env.reset()

    def test_zero_tick_death_takes_priority_over_time_limit(self):
        self.wire.side_effect = [({**self.base, "deaths": 4, "appliedTicks": 0}, png("white"))]
        _, reward, term, cutoff, info = self.env.step(1)
        self.wire.assert_called_with(8, 1, 4, (2, 3, 0))
        self.assertEqual((reward, term, cutoff, info["reason"]), (-1, True, False, "death"))
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.step(1)

    def test_invalid_scene_does_not_become_training_data(self):
        self.wire.side_effect = [({**self.base, "sceneHandle": 9, "appliedTicks": 4}, png("black"))]
        with self.assertRaisesRegex(RuntimeError, "scene change"):
            self.env.step(0)
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.step(0)

    def test_survival_clock_caps_and_stops_at_death(self):
        self.env.config = ShipConfig(living_reward_per_applied_tick=0, survival_per_second=.02,
                                     survival_cap_seconds=50, max_steps=3)
        self.env.origin = {**self.base, "physicsSeconds": 100}
        self.wire.side_effect = [
            ({**self.base, "fixedUpdates": 2600, "physicsSeconds": 149, "appliedTicks": 4}, png("black")),
            ({**self.base, "fixedUpdates": 2700, "physicsSeconds": 154, "deathSeconds": 151,
              "deaths": 4, "appliedTicks": 0}, png("white"))]
        self.assertAlmostEqual(self.env.step(0)[1], .98)
        _, reward, terminal, _, info = self.env.step(0)
        self.assertTrue(terminal)
        self.assertAlmostEqual(reward, -.98)
        self.assertEqual(info["survived_seconds"], 51)

    def test_disconnect_never_replays_action(self):
        self.wire.side_effect = TimeoutError("timed out")
        with self.assertRaises(TimeoutError):
            self.env.step(2)
        self.wire.assert_called_with(8, 2, 4, (2, 3, 0))
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.step(2)
