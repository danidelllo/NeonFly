import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from python.environment.ship import ShipConfig
from python.training.plastic_brain import PlasticBrain
from python.training.brain_awr import Trainer
from python.training.sequence_replay import SequenceReplay


def tiny_brain():
    b=PlasticBrain.__new__(PlasticBrain);nn.Module.__init__(b)
    b.device='cpu';b.n=6;b.outputs=torch.tensor([2,3,4,5]);b.feature_width=4
    matrix=torch.diag(torch.full((6,),.1))
    for i in range(1,6):matrix[i,i-1]=.7
    matrix[0,5]=-.2
    b.register_buffer('connectome',matrix.to_sparse_csr(),persistent=False)
    b.raw_gain=nn.Parameter(torch.full((6,),math.log(.75)))
    b.bias=nn.Parameter(torch.zeros(6))
    b.raw_alpha=nn.Parameter(torch.full((6,),math.log(.55/.35)))
    b.input_neurons=torch.tensor([0,1]);b.input_slots=torch.tensor([0,1])
    b.last_drive=torch.zeros(889)
    return b


class PlasticBrainTests(unittest.TestCase):
    def test_initial_dynamics_match_and_temporal_gradients_reach_intrinsics(self):
        b=tiny_brain();gain,bias,alpha=b.intrinsic()
        torch.testing.assert_close(gain,torch.ones(6));torch.testing.assert_close(alpha,torch.full((6,),.6))
        torch.testing.assert_close(bias,torch.zeros(6))
        drives=torch.randn(8,889,requires_grad=True)
        before=b.connectome.values().clone()
        outputs,_=b.forward_sequence(drives)
        outputs[-1].square().sum().backward()
        self.assertGreater(float(drives.grad[0].norm()),0)
        for name,p in b.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertGreater(float(p.grad.norm()),0,name)
        self.assertFalse(b.connectome.requires_grad)
        torch.testing.assert_close(before,b.connectome.values(),rtol=0,atol=0)
        self.assertEqual(set(b.state_dict()),{'raw_gain','bias','raw_alpha'})

    def test_disk_raw_sequence_old_checkpoint_survives_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory=SequenceReplay(tmp,4,1)
            first=memory.add(np.ones((3,889)),[1,2,3],[0,0,-100])
            saved=memory.state_dict()
            memory.add(np.zeros((3,889)),[4,5,6],[1,1,-100])
            self.assertEqual(memory.size,3)
            restored=SequenceReplay(tmp,4,9);restored.load_state_dict(saved)
            np.testing.assert_array_equal(restored.load(first)['actions'],[1,2,3])
            path=Path(tmp)/first['name']
            with path.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):restored.load(first)

    def test_actual_return_loss_changes_brain_and_resume_restores_optimizer(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=Path('configs/plastic_brain.toml').read_text().replace('bptt_steps = 16','bptt_steps = 2')
            (root/'brain.toml').write_text(config)
            def paths(p):
                p=Path(p)
                return root/p if p.as_posix() in ('brain.toml','logs/ppo','checkpoints') else p
            with patch('python.training.brain_awr.Path',side_effect=paths),patch('python.training.brain_awr.graph_digest',return_value='test'):
                b=tiny_brain();t=Trainer('test',b,'cpu',ShipConfig(),config_path='brain.toml')
                self.assertIsInstance(t.head,nn.Linear)
                self.assertEqual(len(list(t.head.modules())),1)
                rng=np.random.default_rng(3)
                trajectory=[((rng.normal(size=889).astype(np.float32),i%36),5. if i==2 else -100. if i==7 else 0.) for i in range(8)]
                before={k:p.clone() for k,p in b.named_parameters()}
                fixed=b.connectome.values().clone()
                t.episodes=19;t.end_episode(trajectory,None,'death',64,1.28)
                for k,p in b.named_parameters():self.assertGreater(float((p-before[k]).norm()),0,k)
                torch.testing.assert_close(fixed,b.connectome.values(),rtol=0,atol=0)
                self.assertTrue((t.ckpt/'episode-000020.pt').exists())
                t.finish_eval([dict(reward=-95.,reason='death',seconds=1.28)])
                restored=Trainer('test',tiny_brain(),'cpu',ShipConfig(),resume=True,config_path='brain.toml')
                self.assertEqual(restored.best,-95.)
                self.assertTrue(restored.opt.state_dict()['state'])
                item=t.memory.episodes[0]
                t.learn_sequences([item]);restored.learn_sequences([item])
                for k,p in b.named_parameters():torch.testing.assert_close(p,dict(restored.model.named_parameters())[k])
                for k,v in t.head.state_dict().items():torch.testing.assert_close(v,restored.head.state_dict()[k])
                t.writer.close();restored.writer.close()


if __name__=='__main__':unittest.main()
