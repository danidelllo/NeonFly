import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from python.environment.ship import ShipConfig
from python.tests.test_plastic_brain import tiny_brain
from python.training.brain_ppo import Trainer,clipped_actor_loss


class BrainPPOTests(unittest.TestCase):
    def test_negative_advantage_reduces_action_probability_positive_increases(self):
        for sign in (-1.,1.):
            logits=torch.zeros(2,requires_grad=True)
            before=float(logits.detach().softmax(0)[0])
            optimizer=torch.optim.SGD([logits],lr=.1)
            logp=logits.log_softmax(0)[0]
            loss=clipped_actor_loss(logp,torch.tensor(-np.log(2),dtype=torch.float32),torch.tensor(sign),.2)
            loss.backward();optimizer.step()
            after=float(logits.detach().softmax(0)[0])
            self.assertGreater(sign*(after-before),0)

    def test_eight_fresh_episodes_update_brain_and_pending_resume_is_exact(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg=Path('configs/plastic_ppo.toml').read_text().replace('bptt_steps = 16','bptt_steps = 2')
            (root/'ppo.toml').write_text(cfg)
            def paths(p):
                p=Path(p)
                return root/p if p.as_posix() in ('logs/ppo','checkpoints','ppo.toml') else p
            with patch('python.training.brain_awr.Path',side_effect=paths),patch('python.training.brain_awr.graph_digest',return_value='test'):
                t=Trainer('test',tiny_brain(),'cpu',ShipConfig(),config_path='ppo.toml')
                rng=np.random.default_rng(3)
                def episode(trainer,drives):
                    state=None;trajectory=[]
                    for i,raw in enumerate(drives):
                        with torch.no_grad():features,state=trainer.model.forward_sequence(torch.as_tensor(raw)[None],state)
                        trainer.model.last_drive=torch.as_tensor(raw)
                        _,sample=trainer.decision(features[0])
                        trajectory.append((sample,5. if i==1 else -100. if i==3 else 0.))
                    trainer.end_episode(trajectory,None,'death',32,.64)
                before={k:p.detach().clone() for k,p in t.model.named_parameters()}
                fixed=t.model.connectome.values().clone()
                for i in range(8):
                    episode(t,rng.uniform(-1,1,(4,889)).astype(np.float32))
                    if i<7:self.assertEqual(t.updates,0)
                self.assertEqual(t.updates,1);self.assertEqual(len(t.pending),0)
                for k,p in t.model.named_parameters():self.assertGreater(float((p.detach()-before[k]).norm()),0,k)
                torch.testing.assert_close(fixed,t.model.connectome.values(),rtol=0,atol=0)
                import json
                row=json.loads((t.run/'updates.jsonl').read_text().splitlines()[-1])
                self.assertLess(row['initial_logp_error'],1e-5)
                self.assertLessEqual(row['approx_kl'],t.cfg['target_kl'])
                self.assertGreater(row['negative_advantages'],0);self.assertGreater(row['positive_advantages'],0)
                t.episodes=19
                episode(t,rng.uniform(-1,1,(4,889)).astype(np.float32))
                self.assertTrue((t.ckpt/'episode-000020.pt').exists())
                restored=Trainer('test',tiny_brain(),'cpu',ShipConfig(),resume=True,config_path='ppo.toml')
                self.assertEqual(len(restored.pending),1)
                np.testing.assert_array_equal(t.pending[0]['logp'],restored.pending[0]['logp'])
                for _ in range(7):
                    raw=rng.uniform(-1,1,(4,889)).astype(np.float32)
                    episode(t,raw);episode(restored,raw)
                self.assertEqual(t.updates,2);self.assertEqual(restored.updates,2)
                for k,p in t.model.named_parameters():torch.testing.assert_close(p,dict(restored.model.named_parameters())[k])
                for k,v in t.head.state_dict().items():torch.testing.assert_close(v,restored.head.state_dict()[k])
                # Force an excessive post-update change to test actual rollback,
                # including Adam moments, rather than just a KL diagnostic.
                episode(t,rng.uniform(-1,1,(4,889)).astype(np.float32))
                params=list(t.model.parameters())+list(t.head.parameters())
                previous=[p.detach().clone() for p in params]
                previous_optimizer=copy.deepcopy(t.opt.state_dict())
                t.cfg['max_backtracks']=1
                original_statistics=t.policy_statistics
                calls=[0]
                def excessive(rollouts):
                    logp,entropy=original_statistics(rollouts);calls[0]+=1
                    return (logp if calls[0]==1 else logp+10),entropy
                with patch.object(t,'policy_statistics',side_effect=excessive):row=t.update_policy()
                self.assertEqual(row['accepted_epochs'],0)
                for p,v in zip(params,previous):torch.testing.assert_close(p,v,rtol=0,atol=0)
                actual_optimizer=t.opt.state_dict()
                for key,state in previous_optimizer['state'].items():
                    for name,value in state.items():
                        torch.testing.assert_close(value,actual_optimizer['state'][key][name],rtol=0,atol=0)
                t.writer.close();restored.writer.close()


if __name__=='__main__':unittest.main()
