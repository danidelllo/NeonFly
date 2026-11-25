import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from python.environment.ship import ShipConfig, ShipEnv
from python.training.heads import make_head
from python.training.reservoir_ppo import Trainer, ReplayPolicy, compatible_signature, CHECKPOINT_ONLY_PREDECESSORS
from python.vision.connectome import visual_channels, sensory_mapping, resources
from python.tests.test_ship_env import png


class EncodingTests(unittest.TestCase):
    def test_mapping_covers_all_channels_without_fabricating_neurons(self):
        sensory = np.arange(11426, dtype=np.int64)*2
        visual, slots, extra = sensory_mapping(sensory)
        np.testing.assert_array_equal(np.sort(np.concatenate((visual, extra))), sensory)
        np.testing.assert_array_equal(np.unique(slots), np.arange(6*42*42))
        np.testing.assert_array_equal(sensory_mapping(sensory)[0], visual)

    def test_motion_is_signed_and_reset_stack_has_no_motion(self):
        observation = torch.zeros((12, 84, 84), dtype=torch.uint8)
        self.assertEqual(float(visual_channels(observation)[3:5].abs().max()), 0)
        observation[-3:] = 255
        self.assertGreater(float(visual_channels(observation)[3].mean()), .9)
        observation[:] = 255
        observation[-3:] = 0
        self.assertLess(float(visual_channels(observation)[3].mean()), -.9)

    def test_health_and_ammo_have_bounded_mapping(self):
        state = dict(health=50, maxHealth=100, bullets=10, maxBullets=10, rockets=0, maxRockets=10)
        self.assertEqual(resources(state), [0, 1, -1])
        state["health"] = -1
        with self.assertRaises(ValueError):
            resources(state)


class ControlledTests(unittest.TestCase):
    def test_paused_wall_time_and_damage_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = ShipEnv(ShipConfig(controlled=True, damage_penalty=1., living_reward_per_applied_tick=0,
                                     survival_per_second=.01, max_seconds=10))
            env._key_path = Path(tmp)/"key"
            env._key_path.write_bytes(b"x"*32)
            state = dict(scene="Level_1", sceneHandle=2, deaths=0, completions=0, timeScale=0,
                         controller="Player", gameOver=False, fixedUpdates=100, physicsSeconds=10.,
                         shipActionCount=36, bridgeVersion="0.6.0", damageFraction=0., maxHealth=100,
                         controlled=True)
            with patch("python.environment.ship.request") as wire:
                wire.side_effect = [({**state, "sceneHandle": 1}, b""), (state, png("black"))]
                env.reset()
                self.assertEqual(wire.call_args.args, (10,))
                wire.side_effect = [({**state, "appliedTicks": 4, "fixedUpdates": 104,
                                     "physicsSeconds": 10.08, "damageFraction": .25}, png("white"))]
                _, reward, _, _, info = env.step(9)
                wire.assert_called_with(9, 9, 4, (2, 0, 0))
                self.assertAlmostEqual(reward, -.2492)
                self.assertAlmostEqual(sum(info["reward_terms"].values()), reward)


class LearningTests(unittest.TestCase):
    def test_nonlinear_ppo_updates_and_checkpoint_replays(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp, patch("python.training.reservoir_ppo.Path", side_effect=lambda p: Path(tmp)/p), \
             patch("python.training.reservoir_ppo.graph_digest", return_value="test"):
            config = Path(tmp)/"config.toml"
            config.write_text('seed=783\nupdates=2\nepisodes_per_update=1\nepochs=2\nminibatch=8\nlearning_rate=0.001\ngamma=0.99\ngae_lambda=0.95\nclip=0.2\nentropy_coef=0.01\nmax_grad_norm=0.5\neval_every_updates=1\neval_episodes=1\nhidden=32\n')
            trainer = Trainer("test", 12, "cpu", ShipConfig(), config_path="config.toml")
            before = torch.cat([p.detach().flatten().clone() for p in trainer.head.parameters()])
            trajectory = [(trainer.decision(torch.randn(12))[1], float(i % 3 - 1)) for i in range(24)]
            trainer.end_episode(trajectory, 0., "death", 96, 1.92)
            trainer.update()
            after = torch.cat([p.detach().flatten() for p in trainer.head.parameters()])
            self.assertTrue(torch.isfinite(after).all())
            self.assertGreater(float((after-before).norm()), 0)
            trainer.save_policy("best", {"test": True})
            replay = ReplayPolicy(trainer.ckpt/"best.pt", 12, "cpu")
            x = torch.randn(12)
            torch.testing.assert_close(trainer.head(x), replay.head(x))
            trainer.episodes = 19
            trajectory = [(trainer.decision(torch.randn(12))[1], float(i % 3 - 1)) for i in range(24)]
            trainer.end_episode(trajectory, 0., "death", 96, 1.92)
            trainer.update()
            archive = trainer.ckpt/"episode-000020.pt"
            self.assertTrue(archive.exists())
            saved = torch.load(archive, weights_only=True)
            self.assertEqual(saved["progress"][1], 20)
            self.assertTrue(saved["optimizer"]["state"])
            self.assertEqual(saved["pending_batch"], [])
            trainer.finish_eval([dict(reason="death", seconds=1.92, reward=-1.)])
            best = torch.load(trainer.ckpt/"best.pt", weights_only=True)
            self.assertIn("optimizer", best)
            self.assertEqual(best["best_rank"], trainer.best)
            resumed = Trainer("test", 12, "cpu", ShipConfig(), resume=archive, config_path="config.toml")
            torch.testing.assert_close(trainer.head(x), resumed.head(x))
            self.assertEqual(resumed.opt.state_dict()["param_groups"], trainer.opt.state_dict()["param_groups"])
            self.assertEqual(resumed.episodes, 20)
            # A first-win/episode-boundary snapshot must preserve a partial PPO batch.
            resumed.end_episode(trajectory[:3], 0., "death", 12, .24)
            resumed.save_policy("partial", {"test": True})
            partial = Trainer("test", 12, "cpu", ShipConfig(), resume=resumed.ckpt/"partial.pt", config_path="config.toml")
            self.assertEqual(len(partial.batch), 3)
            self.assertEqual(len(partial.returns), 1)
            partial.writer.close()
            resumed.writer.close()
            trainer.writer.close()

    def test_checkpoint_only_migration_rejects_encoder_changes(self):
        old = dict(config={"gamma": .99}, sources={**{k:v[-1] for k,v in CHECKPOINT_ONLY_PREDECESSORS.items()}, "encoder": "same"})
        new = dict(config={"gamma": .99}, sources={k:"new" for k in CHECKPOINT_ONLY_PREDECESSORS})
        new["sources"]["encoder"] = "same"
        self.assertTrue(compatible_signature(old, new))
        new["sources"]["encoder"] = "different"
        self.assertFalse(compatible_signature(old, new))


if __name__ == "__main__":
    unittest.main()
