import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python.environment.ship import ShipConfig, ShipEnv
from python.tests.test_ship_env import png


class CombatRewardTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(scene='Level_1', sceneHandle=2, deaths=3, completions=0,
            timeScale=1, controller='Player', gameOver=False, fixedUpdates=100,
            shipActionCount=36, combatVersion=1, combatDamageFraction=5., combatKills=20)
        self.env = ShipEnv(ShipConfig(max_steps=20, living_reward_per_applied_tick=0.,
            enemy_damage_reward=.05, enemy_kill_reward=.1, combat_reward_cap=.1))
        self.key = tempfile.TemporaryDirectory(); self.addCleanup(self.key.cleanup)
        self.env._key_path = Path(self.key.name)/'key'; self.env._key_path.write_bytes(b'x'*32)
        self.wire = self.enterContext(patch('python.environment.ship.request'))
        self.wire.side_effect = [({**self.base, 'sceneHandle': 1}, b''), (self.base, png('black'))]
        self.env.reset(); self.tick = 100

    def step(self, **extra):
        self.tick += 4
        self.wire.side_effect = [({**self.base, 'fixedUpdates': self.tick, 'appliedTicks': 4, **extra}, png('black'))]
        return self.env.step(0)

    def test_real_damage_then_no_duplicate_credit(self):
        _, reward, _, _, info = self.step(combatDamageFraction=5.4)
        self.assertAlmostEqual(reward, .02)
        self.assertAlmostEqual(info['reward_terms']['combat_damage'], .02)
        self.assertAlmostEqual(info['base_task_reward'], 0.)
        self.assertEqual(self.step(combatDamageFraction=5.4)[1], 0.)
        _, reward, _, _, info = self.step(combatDamageFraction=5.4, combatKills=21)
        self.assertAlmostEqual(reward, .08)  # Remaining shared cap.
        self.assertEqual(info['combat_kills'], 1)

    def test_damage_and_kill_share_cap_and_terms_sum(self):
        _, reward, _, _, info = self.step(combatDamageFraction=6., combatKills=21)
        self.assertAlmostEqual(reward, .1)
        self.assertAlmostEqual(sum(info['reward_terms'].values()), reward)
        self.assertEqual(self.step(combatDamageFraction=7., combatKills=22)[1], 0.)

    def test_missing_or_rewound_counters_reject_transition(self):
        with self.assertRaisesRegex(RuntimeError, 'decreased'):
            self.step(combatDamageFraction=4.)

    def test_missing_schema_rejects_transition(self):
        with self.assertRaisesRegex(RuntimeError, 'disappeared'):
            self.step(combatVersion=0)

    def test_death_does_not_erase_existing_penalty(self):
        _, reward, terminal, _, info = self.step(deaths=4, combatDamageFraction=6., combatKills=21)
        self.assertTrue(terminal)
        self.assertAlmostEqual(reward, -.9)  # Default death -1 plus capped combat .1.
        self.assertAlmostEqual(info['base_task_reward'], -1.)

    def test_invalid_coefficients_are_rejected(self):
        with self.assertRaises(ValueError): ShipConfig(enemy_kill_reward=-.1)
        with self.assertRaises(ValueError): ShipConfig(combat_reward_cap=-1)

    def test_zero_cap_pays_all_new_damage_and_kills_across_steps(self):
        from dataclasses import replace
        self.env.config = replace(self.env.config, combat_reward_cap=0.)
        _, reward, _, _, info = self.step(combatDamageFraction=15., combatKills=30)
        self.assertAlmostEqual(reward, 1.5)
        self.assertAlmostEqual(sum(info['reward_terms'].values()), reward)
        self.assertAlmostEqual(self.step(combatDamageFraction=25., combatKills=40)[1], 1.5)
        self.assertAlmostEqual(self.env.combat_spent, 3.)
        self.assertEqual(self.step(combatDamageFraction=25., combatKills=40)[1], 0.)
