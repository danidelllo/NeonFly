import copy
import io
import unittest

import numpy as np
import torch

from python.training.novelty import GeometryNovelty
from python.vision.range_sensor import sensor_vector, geometry_vector, SENSOR_WIDTH, sensor_mapping


def game():
    return dict(sceneHandle=1, physicsSeconds=10., rays=dict(valid=True, range=12.,
        distance=[6.]*32, visible=[12.]*32, kind=[1]*32, geometryDistance=[6.]*32,
        geometryKind=[1]*32, vx=0., vy=0., screenX=.5, screenY=.5))


class RangeTests(unittest.TestCase):
    def test_encoding_and_closing_speed(self):
        before = game(); after = copy.deepcopy(before)
        after["physicsSeconds"] += .25
        after["rays"]["distance"][0] = 4.
        vector = sensor_vector(after, before)
        self.assertEqual(vector.shape, (SENSOR_WIDTH,))
        self.assertAlmostEqual(float(vector[0]), 2/3)
        self.assertAlmostEqual(float(vector[1]), 1.)
        self.assertAlmostEqual(float(vector[2]), 2/3)
        self.assertTrue(np.all(np.abs(vector) <= 1))
        after["sceneHandle"] = 2
        self.assertEqual(float(sensor_vector(after, before)[1]), 0.)

    def test_novelty_ignores_projectile_occlusion(self):
        a = game(); b = copy.deepcopy(a)
        b["rays"]["distance"][0] = 1.
        b["rays"]["kind"][0] = 3
        np.testing.assert_array_equal(geometry_vector(a), geometry_vector(b))
        self.assertFalse(np.array_equal(sensor_vector(a), sensor_vector(b)))

    def test_invalid_sensor_is_rejected(self):
        state = game(); state["rays"]["distance"][0] = float('nan')
        with self.assertRaises(ValueError): sensor_vector(state)

    def test_offscreen_ship_is_a_valid_unobserved_state(self):
        state = game(); scan = state['rays']
        scan.update(insideView=False, screenY=1.2, distance=[0.]*32, visible=[0.]*32,
                    kind=[0]*32, geometryDistance=[0.]*32, geometryKind=[0]*32)
        vector = sensor_vector(state)
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(float(vector[-1]), 1.)
        cfg = dict(seed=783, rnd_replay_capacity=32, rnd_coefficient=.002,
                   rnd_step_cap=.004, rnd_episode_cap=.25)
        novelty = GeometryNovelty(cfg, 'cpu')
        self.assertEqual(novelty.reward(state), 0.)
        self.assertEqual(novelty.size, 0)
        state = game(); state["rays"]["distance"][0] = 13
        with self.assertRaises(ValueError): sensor_vector(state)

    def test_mechanosensory_mapping_retains_real_ids(self):
        neurons, slots = sensor_mapping()
        self.assertEqual(len(np.unique(neurons)), 2674)
        np.testing.assert_array_equal(np.unique(slots), np.arange(SENSOR_WIDTH))

    def test_offscreen_visible_segments_reach_policy(self):
        state = game(); scan = state['rays']
        scan.update(insideView=False, screenY=1.01, visibleStart=[.1]*32)
        vector = sensor_vector(state)
        self.assertEqual(vector.shape, (SENSOR_WIDTH,))
        self.assertAlmostEqual(float(vector[0]), .5)
        self.assertTrue(np.any(geometry_vector(state)[:128] > 0))
        scan['distance'][0] = .05
        with self.assertRaises(ValueError): sensor_vector(state)
        scan['distance'][0] = 6.
        scan['geometryDistance'][0] = .05
        with self.assertRaises(ValueError): geometry_vector(state)


class NoveltyTests(unittest.TestCase):
    def test_budget_learning_and_checkpoint_resume(self):
        torch.set_num_threads(2)
        cfg = dict(seed=783, rnd_replay_capacity=128, rnd_coefficient=.002,
                   rnd_step_cap=.004, rnd_episode_cap=.01)
        learner = GeometryNovelty(cfg, 'cpu'); state = game()
        features = torch.from_numpy(geometry_vector(state))
        def error():
            with torch.no_grad(): return float((learner.predictor(features)-learner.target(features)).square().mean())
        initial = error()
        rewards = [learner.reward(state) for _ in range(100)]
        self.assertAlmostEqual(sum(rewards), .01)
        self.assertLess(error(), initial*.5)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in learner.target.parameters()))
        stream = io.BytesIO(); torch.save(learner.state_dict(), stream); stream.seek(0)
        restored = GeometryNovelty(cfg, 'cpu')
        restored.load_state_dict(torch.load(stream, weights_only=True))
        learner.begin_episode(); restored.begin_episode()
        self.assertAlmostEqual(learner.reward(state), restored.reward(state), places=9)
        for left,right in zip(learner.predictor.parameters(),restored.predictor.parameters()):
            torch.testing.assert_close(left,right)


if __name__ == '__main__': unittest.main()
