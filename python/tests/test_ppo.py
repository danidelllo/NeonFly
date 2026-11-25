import unittest

import numpy as np

from python.training.reservoir_ppo import episode_targets, evaluation_rank


class TargetsTest(unittest.TestCase):
    def test_wins_outrank_survival_and_faster_wins_break_ties(self):
        slow = [{"reason": "completion", "seconds": 80}, {"reason": "death", "seconds": 5}]
        fast = [{"reason": "completion", "seconds": 70}, {"reason": "death", "seconds": 3}]
        stalls = [{"reason": "time_limit", "seconds": 480}] * 2
        self.assertGreater(evaluation_rank(fast), evaluation_rank(slow))
        self.assertGreater(evaluation_rank(slow), evaluation_rank(stalls))
    def test_time_limit_bootstraps_terminal_does_not(self):
        for boot, expected in [(0, [2, 2]), (4, [3, 4])]:
            advantages, returns = episode_targets([1, 2], [0.2, 0.4], boot, 0.5, 1.0)
            np.testing.assert_allclose(returns, expected)
            np.testing.assert_allclose(advantages, np.asarray(expected)-[0.2, 0.4])

    def test_zero_lambda_uses_next_value(self):
        _, returns = episode_targets([1, 2], [3, 4], 0, 0.5, 0)
        np.testing.assert_allclose(returns, [3, 2])
