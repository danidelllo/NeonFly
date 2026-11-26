import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from python.environment.ship import ShipConfig
from python.training.reservoir_awr import Trainer,monte_carlo_returns,reconstruct_returns,regression_weights
from python.training.reservoir_dqn import nstep_transitions


class AWRTests(unittest.TestCase):
    def test_observed_death_reaches_all_earlier_states(self):
        np.testing.assert_allclose(monte_carlo_returns([0,0,0,-100],.5),[-12.5,-25,-50,-100])

    def test_reconstruct_all_returns_from_saved_nstep_blocks(self):
        rng=np.random.default_rng(3)
        for count in (1,3,5,6,100):
            rewards=rng.normal(size=count)
            trajectory=[((np.zeros(2),0),r) for r in rewards]
            steps=list(nstep_transitions(trajectory,np.zeros(2),True,.999,5))
            result=reconstruct_returns(np.array([x[2] for x in steps]),np.array([x[4] for x in steps]),5)
            np.testing.assert_allclose(result,monte_carlo_returns(rewards,.999),rtol=1e-5,atol=1e-5)
        with self.assertRaises(ValueError):
            reconstruct_returns(np.ones(3),np.full(3,.9),5)

    def test_negative_advantages_do_not_erase_all_policy_gradients(self):
        weights=regression_weights(torch.tensor([-100.,-101.,-99.]),1.,20.)
        self.assertAlmostEqual(float(weights.mean()),1.,places=6)
        self.assertGreater(float(weights[2]),float(weights[1]))
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(regression_weights(torch.full((8,),-100.),1.,20.),torch.ones(8))

    def test_measured_targets_and_full_optimizer_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=Path('configs/radar_awr.toml').read_text().replace('replay_capacity = 1000000','replay_capacity = 32').replace('learning_starts = 1024','learning_starts = 2').replace('batch_size = 128','batch_size = 4')
            (root/'awr.toml').write_text(cfg)
            def paths(p):
                p=Path(p)
                return root/p if p.as_posix() in ('logs/ppo','checkpoints','awr.toml') else p
            with patch('python.training.reservoir_dqn.Path',side_effect=paths),patch('python.training.reservoir_dqn.graph_digest',return_value='test'):
                t=Trainer('test',12,'cpu',ShipConfig(),config_path='awr.toml')
                trajectory=[(t.decision(torch.randn(12))[1],r) for r in (5.,0.,0.,-100.)]
                t.episodes=19
                t.end_episode(trajectory,None,'death',32,.64)
                expected=monte_carlo_returns([5,0,0,-100],t.cfg['gamma'])
                np.testing.assert_allclose(t.memory.batch(np.arange(4))['rewards'],expected)
                original=t.memory.batch(np.arange(4))['rewards'].copy()
                # A changed critic cannot change any measured training targets.
                with torch.no_grad():t.head.critic.bias.fill_(10000.)
                t.learn()
                np.testing.assert_array_equal(t.memory.batch(np.arange(4))['rewards'],original)
                t.save_policy('latest',{})
                resumed=Trainer('test',12,'cpu',ShipConfig(),resume=True,config_path='awr.toml')
                self.assertTrue((t.ckpt/'episode-000020.pt').exists())
                t.learn();resumed.learn()
                for k,v in t.head.state_dict().items():torch.testing.assert_close(v,resumed.head.state_dict()[k])
                t.writer.close();resumed.writer.close();t.memory.close();resumed.memory.close()


if __name__=='__main__':unittest.main()
