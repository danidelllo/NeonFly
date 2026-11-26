import copy
import io
import unittest

import numpy as np
import torch

from python.training.novelty import GeometryNovelty
from python.vision.range_sensor import sensor_vector, geometry_vector, SENSOR_WIDTH, sensor_mapping, RAY_COUNT, RAY_LAYOUT, ray_angles, LEGACY_RAY_ENCODER


def game():
    return dict(sceneHandle=1, physicsSeconds=10., rays=dict(valid=True, range=12., layout=RAY_LAYOUT, angles=ray_angles().tolist(),
        distance=[6.]*RAY_COUNT, visible=[12.]*RAY_COUNT, kind=[1]*RAY_COUNT, geometryDistance=[6.]*RAY_COUNT,
        geometryKind=[1]*RAY_COUNT, vx=0., vy=0., screenX=.5, screenY=.5))


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
        scan.update(insideView=False, screenY=1.2, distance=[0.]*RAY_COUNT, visible=[0.]*RAY_COUNT,
                    kind=[0]*RAY_COUNT, geometryDistance=[0.]*RAY_COUNT, geometryKind=[0]*RAY_COUNT)
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
        self.assertEqual(len(np.unique(neurons)), 2658)
        np.testing.assert_array_equal(np.unique(slots), np.arange(SENSOR_WIDTH))
        np.testing.assert_array_equal(np.bincount(slots), np.full(SENSOR_WIDTH, 3))
        old_neurons, old_slots = sensor_mapping(LEGACY_RAY_ENCODER)
        self.assertEqual(len(old_neurons), 2674)
        self.assertEqual(len(np.unique(old_slots)), 228)

    def test_quadrant_counts_spacing_and_layout_validation(self):
        angles = np.rad2deg(ray_angles())
        for part, count, low, high in ((angles[:55],55,-45,45), (angles[55:63],8,45,135),
                                      (angles[63:71],8,135,225), (angles[71:],55,225,315)):
            self.assertEqual(len(part), count)
            self.assertTrue(np.all((part > low) & (part < high)))
            np.testing.assert_allclose(np.diff(part), 90/count, atol=1e-4)
        self.assertEqual(len(np.unique(angles)), 126)
        state = game(); state['rays']['angles'][0] += .1
        with self.assertRaisesRegex(ValueError, 'layout'): sensor_vector(state)

    def test_legacy_sensor_packets_remain_decodable(self):
        state = game(); scan = state['rays']
        for key in ('distance','visible','kind','geometryDistance','geometryKind'):
            scan[key] = scan[key][:32]
        scan.pop('angles'); scan.pop('layout')
        self.assertEqual(sensor_vector(state).shape, (228,))
        self.assertEqual(geometry_vector(state).shape, (130,))

    def test_offscreen_visible_segments_reach_policy(self):
        state = game(); scan = state['rays']
        scan.update(insideView=False, screenY=1.01, visibleStart=[.1]*RAY_COUNT)
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


class CoverageTests(unittest.TestCase):
    def learner(self):
        torch.set_num_threads(2)
        return GeometryNovelty(dict(seed=783, rnd_replay_capacity=128, rnd_coefficient=0.,
            rnd_step_cap=.008, rnd_episode_cap=.25, episodic_coverage=True,
            coverage_bins=[12, 8], coverage_cell_bonus=.006), 'cpu')

    def test_new_positions_rewarded_once_not_hovering_bouncing_or_damage(self):
        learner = self.learner(); state = game(); learner.begin_episode(state)
        self.assertEqual(learner.reward(state), 0.)
        state['rays']['screenY'] = .35
        self.assertAlmostEqual(learner.reward(state), .006)
        self.assertEqual(learner.reward(state), 0.)
        state['rays']['screenY'] = .5
        self.assertEqual(learner.reward(state), 0.)
        state['rays']['screenY'] = .2
        self.assertEqual(learner.reward(state, allow_bonus=False), 0.)
        self.assertEqual(learner.reward(state), 0.)
        self.assertEqual(learner.stats['blocked_bonus_steps'], 1)
        learner.begin_episode(game())
        self.assertAlmostEqual(learner.reward(state), .006)

    def test_total_reward_budget_is_smaller_than_full_bar_damage(self):
        learner = self.learner(); state = game(); total = 0.
        for x in range(12):
            for y in range(8):
                state['rays'].update(screenX=(x+.5)/12, screenY=(y+.5)/8)
                reward = learner.reward(state)
                self.assertLessEqual(reward, .008)
                total += reward
        self.assertAlmostEqual(total, .25)
        self.assertAlmostEqual(learner.stats['coverage_reward']+learner.stats['rnd_reward'], total)
        self.assertEqual(learner.reward(state), 0.)

    def test_visible_ceiling_state_counts_as_boundary_cell_not_extra_space(self):
        learner = self.learner(); state = game()
        state['rays'].update(insideView=False, screenY=1.01, visibleStart=[.1]*RAY_COUNT)
        self.assertAlmostEqual(learner.reward(state), .006)
        state['rays'].update(insideView=True, screenY=.99)
        self.assertEqual(learner.reward(state), 0.)
        size = learner.size
        state['rays'].update(insideView=False, screenX=.1, visible=[0.]*RAY_COUNT, visibleStart=[0.]*RAY_COUNT,
                             distance=[0.]*RAY_COUNT, kind=[0]*RAY_COUNT)
        self.assertEqual(learner.reward(state), 0.)
        self.assertEqual(learner.size, size)

    def test_checkpoint_preserves_visited_cells_budget_and_learning_state(self):
        learner = self.learner(); state = game(); learner.reward(state)
        saved = learner.state_dict(); restored = self.learner(); restored.load_state_dict(saved)
        self.assertEqual(restored.reward(state), 0.)
        state['rays']['screenY'] = .2
        self.assertAlmostEqual(learner.reward(state), restored.reward(state), places=9)
        np.testing.assert_array_equal(learner.visited, restored.visited)
        self.assertAlmostEqual(learner.spent, restored.spent)

    def test_zero_novelty_caps_do_not_clip_steps_or_episode_total(self):
        learner = self.learner()
        learner.cfg.update(rnd_step_cap=0., rnd_episode_cap=0., coverage_cell_bonus=.02)
        state = game(); total = 0.
        for x in range(12):
            for y in range(8):
                state['rays'].update(screenX=(x+.5)/12, screenY=(y+.5)/8)
                self.assertAlmostEqual(learner.reward(state), .02)
                total += .02
        self.assertAlmostEqual(learner.spent, 1.92)
        self.assertEqual(learner.reward(state), 0.)


if __name__ == '__main__': unittest.main()
