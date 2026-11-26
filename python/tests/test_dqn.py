import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from python.environment.ship import ShipConfig
from python.training.reservoir_dqn import Replay, Trainer, double_targets, nstep_transitions


class DQNTests(unittest.TestCase):
    def test_multistep_death_and_time_limit(self):
        trajectory = [((np.array([i], np.float32), i), float(i+1)) for i in range(3)]
        death = list(nstep_transitions(trajectory, np.array([9]), True, .5, 2))
        self.assertEqual([x[2] for x in death], [2., 3.5, 3.])
        self.assertEqual([x[4] for x in death], [.25, 0., 0.])
        np.testing.assert_array_equal(death[0][3], [2])
        limited = list(nstep_transitions(trajectory, np.array([9]), False, .5, 2))
        self.assertEqual([x[4] for x in limited], [.25, .25, .5])
        np.testing.assert_array_equal(limited[-1][3], [9])

    def test_double_dqn_selects_online_evaluates_target(self):
        online = lambda x:torch.tensor([[1., 3.], [4., 1.]])
        target = lambda x:torch.tensor([[10., 2.], [3., 20.]])
        y = double_targets(online, target, torch.zeros(2, 1), torch.tensor([1., 2.]), torch.tensor([.5, 0.]))
        torch.testing.assert_close(y, torch.tensor([2., 2.]))

    def test_priorities_wrap_and_rng_restore(self):
        replay = Replay(4, 2, 3)
        for i in range(6):
            replay.add(np.full(2, i), i, i, np.full(2, i+1), .9)
        self.assertEqual((replay.size, replay.cursor), (4, 2))
        replay.reprioritize(np.array([0,0,1]), np.array([2., 5., 1.]))
        self.assertAlmostEqual(replay.priorities[0], 5.00001, places=4)
        restored = Replay(4,2,9)
        restored.load_state_dict(replay.state_dict())
        a, w = replay.sample(1000, .6, .4)
        b, v = restored.sample(1000, .6, .4)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(w,v)
        self.assertGreater(np.sum(a==0), np.sum(a==1))
        self.assertTrue(np.all((w>0)&(w<=1)))
        np.testing.assert_array_equal(restored.states,replay.states)

    def test_learning_full_resume_and_reward_selected_best(self):
        self.learning_resume(False)

    def test_disk_learning_full_resume_and_reward_selected_best(self):
        self.learning_resume(True)

    def learning_resume(self, disk):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Path('configs/dqn.toml').read_text().replace('replay_capacity = 20000','replay_capacity = 32').replace('learning_starts = 1024','learning_starts = 2').replace('batch_size = 128','batch_size = 4').replace('train_every = 4','train_every = 1').replace('target_sync_updates = 250','target_sync_updates = 2')
            if disk:
                cfg += '\nreplay_storage = "disk"\n'
            (root/'dqn.toml').write_text(cfg)
            def paths(p):
                p = Path(p)
                return root/p if p.as_posix() in ('logs/ppo','checkpoints','dqn.toml') else p
            with patch('python.training.reservoir_dqn.Path', side_effect=paths), patch('python.training.reservoir_dqn.graph_digest',return_value='test'):
                t = Trainer('test', 12, 'cpu', ShipConfig(), config_path='dqn.toml')
                before = {k:v.clone() for k,v in t.head.state_dict().items()}
                t.episodes = 19
                trajectory = [(t.decision(torch.randn(12))[1], float(i%2)) for i in range(4)]
                t.end_episode(trajectory, np.zeros(12), 'death',32,.64)
                self.assertEqual(t.updates,4)
                self.assertTrue(any(not torch.equal(before[k],v) for k,v in t.head.state_dict().items()))
                self.assertTrue(all(torch.equal(t.target.state_dict()[k],v) for k,v in t.head.state_dict().items()))
                archive = t.ckpt/'episode-000020.pt'
                self.assertTrue(archive.exists())
                t.finish_eval([dict(reason='death',seconds=2.,reward=-1.)])
                t.finish_eval([dict(reason='completion',seconds=80.,reward=-2.)])
                best = torch.load(t.ckpt/'best.pt',weights_only=True)
                self.assertEqual(best['selection']['mean_reward'],-1.)
                self.assertEqual(t.best,-1.)
                resumed = Trainer('test',12,'cpu',ShipConfig(),resume=True,config_path='dqn.toml')
                self.assertEqual((resumed.steps,resumed.episodes,resumed.updates),(4,20,4))
                self.assertEqual(resumed.best,-1.)
                np.testing.assert_array_equal(t.memory.sample(4,.6,.4)[0],resumed.memory.sample(4,.6,.4)[0])
                for k,v in t.target.state_dict().items():
                    torch.testing.assert_close(v,resumed.target.state_dict()[k])
                for _ in range(4):
                    x=torch.randn(12)
                    self.assertEqual(t.decision(x)[0],resumed.decision(x)[0])
                t.learn();resumed.learn()
                for k,v in t.head.state_dict().items():
                    torch.testing.assert_close(v,resumed.head.state_dict()[k])
                t.writer.close();resumed.writer.close()
                if disk:
                    t.memory.close();resumed.memory.close()


if __name__ == '__main__':
    unittest.main()
