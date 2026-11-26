import unittest

import numpy as np

from python.training.reservoir_ppo import episode_targets, evaluation_rank


class TargetsTest(unittest.TestCase):
    def test_reward_ranks_without_survival_or_completion_override(self):
        combat = [dict(reason='death', seconds=5, reward=-1.)]
        avoidance = [dict(reason='death', seconds=80, reward=-2.)]
        self.assertGreater(evaluation_rank(combat), evaluation_rank(avoidance))
        self.assertEqual(evaluation_rank(combat), evaluation_rank([dict(reason='completion', seconds=2, reward=-1.)]))
    def test_time_limit_bootstraps_terminal_does_not(self):
        for boot, expected in [(0, [2, 2]), (4, [3, 4])]:
            advantages, returns = episode_targets([1, 2], [0.2, 0.4], boot, 0.5, 1.0)
            np.testing.assert_allclose(returns, expected)
            np.testing.assert_allclose(advantages, np.asarray(expected)-[0.2, 0.4])

    def test_zero_lambda_uses_next_value(self):
        _, returns = episode_targets([1, 2], [3, 4], 0, 0.5, 0)
        np.testing.assert_allclose(returns, [3, 2])
