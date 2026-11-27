"""Sequence replay learns intrinsic fly-neuron dynamics and a linear decoder."""
import hashlib
import json
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from python.training.heads import make_head
from python.training.reservoir_ppo import graph_digest
from python.training.reservoir_awr import monte_carlo_returns,regression_weights
from python.training.reservoir_dqn import Trainer as Utilities
from python.training.sequence_replay import SequenceReplay


class Trainer:
    algorithm = 'brain_awr'
    log = Utilities.log
    finished = Utilities.finished
    should_eval = Utilities.should_eval
    finish_eval = Utilities.finish_eval
    save_policy = Utilities.save_policy

    def __init__(self,run,model,device,env_config,resume=False,warm_start=None,config_path='configs/plastic_brain.toml'):
        if warm_start:
            raise ValueError('plastic brain starts fresh or resumes its own full state')
        self.cfg = tomllib.loads(Path(config_path).read_text())
        if self.cfg['hidden'] != 0 or self.cfg['algorithm'] != self.algorithm:
            raise ValueError('plastic brain requires a single linear readout')
        self.model,self.device,self.width = model,device,model.feature_width
        self.run,self.ckpt = Path('logs/ppo')/run,Path('checkpoints')/run
        self.run.mkdir(parents=True,exist_ok=bool(resume));self.ckpt.mkdir(parents=True,exist_ok=bool(resume))
        names = ('python/training/plastic_brain.py','python/training/brain_awr.py',
                 'python/training/sequence_replay.py','python/training/reservoir_awr.py',
                 'python/training/reservoir_dqn.py','python/training/heads.py',
                 'python/viewer/live.py','python/viewer/control.py','python/environment/ship.py',
                 'python/vision/range_sensor.py','python/vision/connectome.py','python/communication/observe.py',
                 'unity-mod/src/Bridge.cs','unity-mod/src/GameAdapter.cs','unity-mod/src/RangeSensor.cs',
                 'unity-mod/src/Combat.cs','unity-mod/src/CombatLedger.cs')
        self.signature = dict(config=self.cfg,environment=asdict(env_config),graph=graph_digest(),width=self.width,
                              representation='intrinsic-gain-bias-alpha;raw-radar889;motor-current;linear37-v1',
                              sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names})
        if self.algorithm == 'brain_ppo':
            for p in ('python/training/brain_ppo.py','python/training/reservoir_ppo.py'):
                self.signature['sources'][p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        torch.manual_seed(self.cfg['seed'])
        self.head = make_head(self.width,self.cfg,device)
        self.opt = torch.optim.Adam([{'params':list(model.parameters()),'lr':self.cfg['brain_learning_rate']},
                                     {'params':list(self.head.parameters()),'lr':self.cfg['learning_rate']}])
        self.memory = SequenceReplay(self.ckpt/'replay-data',self.cfg['replay_capacity'],self.cfg['seed']+1)
        self.rng = np.random.default_rng(self.cfg['seed']+2)
        self.steps = self.episodes = self.updates = 0
        self.best,self.last_eval,self.novelty = None,-1,None
        self.keep_alive = lambda:None
        if resume:
            path = self.ckpt/'latest.pt' if resume is True else Path(resume)
            saved = torch.load(path,map_location='cpu',weights_only=True)
            if saved['signature'] != self.signature:
                raise ValueError('trainable-brain checkpoint source/config mismatch')
            model.load_state_dict(saved['brain']);self.head.load_state_dict(saved['head'])
            self.opt.load_state_dict(saved['optimizer']);self.memory.load_state_dict(saved['replay'])
            self.steps,self.episodes,self.updates = saved['progress']
            self.best,self.last_eval = saved['best_rank'],saved['last_eval']
            self.rng.bit_generator.state = saved['action_rng'];torch.set_rng_state(saved['rng'])
            if device=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'])
        else:
            (self.run/'config.json').write_text(json.dumps(self.signature,indent=2))
            self.save_policy('initial',dict(reason='fresh intrinsic-neuron parameters and linear readout'))
        self.writer = SummaryWriter(str(self.run/'tensorboard'))

    def begin_episode(self):
        pass

    @torch.no_grad()
    def decision(self,features):
        logits = self.head(F.layer_norm(features,features.shape))[:36]
        p = logits.softmax(-1).cpu().numpy().astype(np.float64);p/=p.sum()
        action = int(self.rng.choice(36,p=p))
        return action,(self.model.last_drive.cpu().numpy().copy(),action)

    def learn_sequences(self,items):
        episodes = [self.memory.load(item) for item in items]
        total = sum(len(e['actions']) for e in episodes)
        # First pass recomputes current brain state from episode start. Stored
        # activations from old parameters are NEVER used for learning.
        values,targets = [],[]
        with torch.no_grad():
            for episode in episodes:
                state = None
                for start in range(0,len(episode['actions']),self.cfg['bptt_steps']):
                    raw = torch.as_tensor(episode['drives'][start:start+self.cfg['bptt_steps']],device=self.device)
                    features,state = self.model.forward_sequence(raw,state)
                    values.append(self.head(F.layer_norm(features,(self.width,)))[:,36])
                    self.keep_alive()
                targets.append(torch.as_tensor(episode['returns'],device=self.device)*self.cfg['reward_scale'])
        all_targets = torch.cat(targets)
        all_weights = regression_weights(all_targets-torch.cat(values),self.cfg['temperature'],self.cfg['max_actor_weight'])
        before = {k:p.detach().clone() for k,p in self.model.named_parameters()}
        self.opt.zero_grad(set_to_none=True)
        position = 0
        losses = []
        for episode in episodes:
            state = None
            for start in range(0,len(episode['actions']),self.cfg['bptt_steps']):
                stop = min(start+self.cfg['bptt_steps'],len(episode['actions']));count=stop-start
                raw = torch.as_tensor(episode['drives'][start:stop],device=self.device)
                features,state = self.model.forward_sequence(raw,state)
                output = self.head(F.layer_norm(features,(self.width,)))
                distribution = Categorical(logits=output[:,:36])
                actions = torch.as_tensor(episode['actions'][start:stop],device=self.device)
                weights = all_weights[position:position+count]
                expected = all_targets[position:position+count]
                loss = (-(weights*distribution.log_prob(actions)).sum()
                        +.5*F.smooth_l1_loss(output[:,36],expected,reduction='sum')
                        -self.cfg['entropy_coef']*distribution.entropy().sum())/total
                if not torch.isfinite(loss):raise FloatingPointError('non-finite brain sequence loss')
                loss.backward()
                losses.append(float(loss.detach()))
                state = state.detach() # Truncation changes gradients, not forward memory.
                position += count
                self.keep_alive()
        gradient = {}
        for name,p in self.model.named_parameters():
            gradient[name] = float(p.grad.norm()) if p.grad is not None else 0.
        torch.nn.utils.clip_grad_norm_(list(self.model.parameters())+list(self.head.parameters()),
                                      self.cfg['max_grad_norm'],error_if_nonfinite=True)
        self.opt.step();self.updates+=1
        delta = {name:float((p.detach()-before[name]).norm()) for name,p in self.model.named_parameters()}
        metrics = dict(update=self.updates,episodes=self.episodes,steps=self.steps,loss=sum(losses),
                       measured_return_mean=float(all_targets.mean()),value_mean=float(torch.cat(values).mean()),
                       neuron_gradient_norms=gradient,neuron_parameter_deltas=delta,
                       sequence_actions=total,bptt_steps=self.cfg['bptt_steps'])
        self.log('updates',metrics)
        for k,v in gradient.items():self.writer.add_scalar('brain-gradient/'+k,v,self.steps)
        for k,v in delta.items():self.writer.add_scalar('brain-delta/'+k,v,self.steps)
        self.writer.flush()
        return metrics

    def end_episode(self,trajectory,tail,reason,elapsed_ticks,seconds,details=None):
        drives = np.stack([sample[0] for sample,_ in trajectory])
        actions = np.array([sample[1] for sample,_ in trajectory],np.int64)
        rewards = [reward for _,reward in trajectory]
        item = self.memory.add(drives,actions,monte_carlo_returns(rewards,self.cfg['gamma']))
        self.episodes+=1;self.steps+=len(trajectory)
        self.log('episodes',dict(episode=self.episodes,steps=self.steps,length=len(trajectory),reward=sum(rewards),
                                reason=reason,elapsed_ticks=elapsed_ticks,survived_seconds=seconds,**(details or {})))
        items = [item]+self.memory.sample(self.cfg['replay_episodes']-1)
        metrics = self.learn_sequences(items)
        print(f'BRAIN episode {self.episodes}: {reason}, {seconds:.2f}s, reward={sum(rewards):.4f}; update={self.updates}, neuron deltas={metrics["neuron_parameter_deltas"]}',flush=True)
        self.save_policy('latest',{}) # Matches every live linear-readout snapshot.
        if self.episodes%20==0:self.save_policy(f'episode-{self.episodes:06}',{})
        if reason=='completion' and not (self.ckpt/'first-training-win.pt').exists():self.save_policy('first-training-win',{})

    def training_state(self):
        return dict(signature=self.signature,brain=self.model.state_dict(),head=self.head.state_dict(),
                    optimizer=self.opt.state_dict(),replay=self.memory.state_dict(),
                    progress=(self.steps,self.episodes,self.updates),best_rank=self.best,last_eval=self.last_eval,
                    action_rng=self.rng.bit_generator.state,rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state() if self.device=='cuda' else None)
