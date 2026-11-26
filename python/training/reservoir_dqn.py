"""Double/dueling DQN on immutable connectome features; task-reward replay only."""
import copy
import hashlib
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from python.training.heads import make_head
from python.training.reservoir_ppo import graph_digest


class Replay:
    def __init__(self, capacity, width, seed):
        self.capacity, self.width = capacity, width
        self.states = np.empty((capacity, width), np.float16)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.discounts = np.empty(capacity, np.float32)
        self.priorities = np.zeros(capacity, np.float32)
        self.size = self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, discount):
        i = self.cursor
        priority = max(1., float(self.priorities[:self.size].max())) if self.size else 1.
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.discounts[i] = action, reward, discount
        self.priorities[i] = priority
        self.cursor = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch, alpha, beta):
        probabilities = self.priorities[:self.size].astype(np.float64)**alpha
        probabilities /= probabilities.sum()
        indices = self.rng.choice(self.size, batch, p=probabilities)
        # Normalize by maximum possible weight, not the sampled maximum.
        weights = (probabilities[indices]/probabilities.min())**(-beta)
        return indices, weights.astype(np.float32)

    def batch(self, indices):
        return {k:getattr(self,k)[indices] for k in ('states','next_states','actions','rewards','discounts')}

    def reprioritize(self, indices, errors):
        # Repeated sampled indices receive the maximum TD error in the batch.
        for i in np.unique(indices):
            self.priorities[i] = float(np.max(errors[indices == i])) + 1e-5

    def state_dict(self):
        return dict(size=self.size, cursor=self.cursor, rng=self.rng.bit_generator.state,
                    **{k:torch.from_numpy(getattr(self, k)[:self.size].copy()) for k in
                       ('states', 'next_states', 'actions', 'rewards', 'discounts', 'priorities')})

    def load_state_dict(self, saved):
        self.size, self.cursor = saved['size'], saved['cursor']
        if not 0 <= self.size <= self.capacity or not 0 <= self.cursor < self.capacity:
            raise ValueError('invalid replay checkpoint')
        for k in ('states', 'next_states', 'actions', 'rewards', 'discounts', 'priorities'):
            getattr(self, k)[:self.size] = saved[k].numpy()
        self.rng.bit_generator.state = saved['rng']


def nstep_transitions(trajectory, tail, terminal, gamma, n):
    """Never cross episode boundaries. Time limits bootstrap; deaths do not."""
    for i, ((state, action), _) in enumerate(trajectory):
        end = min(i+n, len(trajectory))
        reward = sum(gamma**(j-i)*trajectory[j][1] for j in range(i, end))
        next_state = trajectory[end][0][0] if end < len(trajectory) else tail
        discount = 0. if terminal and end == len(trajectory) else gamma**(end-i)
        yield state, action, reward, next_state, discount


@torch.no_grad()
def double_targets(head, target, states, rewards, discounts):
    actions = head(states).argmax(dim=1, keepdim=True)
    return rewards + discounts*target(states).gather(1, actions).squeeze(1)


class Trainer:
    algorithm = 'dqn'

    def __init__(self, run, features, device, env_config, resume=False, warm_start=None,
                 config_path='configs/dqn.toml'):
        if warm_start:
            raise ValueError('DQN must start fresh or resume its own checkpoint')
        self.cfg = tomllib.loads(Path(config_path).read_text())
        if self.cfg['algorithm'] not in ('dqn','awr') or self.cfg.get('rnd'):
            raise ValueError('DQN replay stores task reward; RND must be disabled')
        self.run, self.ckpt = Path('logs/ppo')/run, Path('checkpoints')/run
        self.run.mkdir(parents=True, exist_ok=bool(resume))
        self.ckpt.mkdir(parents=True, exist_ok=bool(resume))
        self.device, self.width = device, features
        sources = ('python/training/reservoir_dqn.py', 'python/training/heads.py',
                   'python/viewer/live.py', 'python/viewer/control.py',
                   'python/vision/connectome.py', 'python/vision/range_sensor.py',
                   'python/environment/ship.py', 'python/communication/observe.py',
                   'unity-mod/src/Bridge.cs', 'unity-mod/src/GameAdapter.cs',
                   'unity-mod/src/RangeSensor.cs', 'unity-mod/src/Combat.cs', 'unity-mod/src/CombatLedger.cs')
        self.signature = dict(config=self.cfg, environment=asdict(env_config), graph=graph_digest(),
                              width=features, selection='mean-evaluation-task-reward-v1',
                              representation=self.cfg['encoder']+';resource64;outputs+delta+ema;leak0.6;substeps3',
                              sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
        if self.cfg['encoder'] == 'rays-resources-v1':
            p = 'python/vision/radar_readout.py'
            self.signature['sources'][p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            self.signature['representation'] += ';sensor-recipients1024'
        if self.cfg['algorithm'] == 'awr':
            p = 'python/training/reservoir_awr.py'
            self.signature['sources'][p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            self.signature['return_targets'] = 'complete-episode-monte-carlo-v1'
        if self.cfg.get('replay_storage') == 'disk':
            p = 'python/training/disk_replay.py'
            self.signature['sources'][p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        torch.manual_seed(self.cfg['seed'])
        self.head = make_head(features, self.cfg, device)
        self.target = copy.deepcopy(self.head).requires_grad_(False).eval()
        self.opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg['learning_rate'])
        if self.cfg.get('replay_storage') == 'disk':
            from python.training.disk_replay import DiskReplay
            self.memory = DiskReplay(self.cfg['replay_capacity'], features, self.cfg['seed']+1,
                                     self.ckpt/'replay-data', self.cfg['priority_alpha'],
                                     uniform_fraction=self.cfg.get('replay_uniform_fraction',0.),
                                     weight_normalization=self.cfg.get('replay_weight_normalization','global'))
        else:
            self.memory = Replay(self.cfg['replay_capacity'], features, self.cfg['seed']+1)
        self.rng = np.random.default_rng(self.cfg['seed']+2)
        self.steps = self.episodes = self.updates = 0
        self.last_eval = -1
        self.best = None
        self.novelty = None
        self.explore_remaining = self.explore_action = 0
        if resume:
            path = self.ckpt/'latest.pt' if resume is True else Path(resume)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['signature'] != self.signature:
                raise ValueError('DQN checkpoint source/configuration mismatch')
            self.head.load_state_dict(saved['head'])
            self.target.load_state_dict(saved['target'])
            self.opt.load_state_dict(saved['optimizer'])
            self.memory.load_state_dict(saved['replay'])
            self.steps, self.episodes, self.updates = saved['progress']
            self.last_eval, self.best = saved['last_eval'], saved['best_rank']
            self.rng.bit_generator.state = saved['action_rng']
            self.explore_remaining, self.explore_action = saved.get('exploration_hold',(0,0))
            torch.set_rng_state(saved['rng'])
            if device == 'cuda':
                torch.cuda.set_rng_state(saved['cuda_rng'])
        else:
            (self.run/'config.json').write_text(json.dumps(self.signature, indent=2))
            self.save_policy('initial', dict(reason='fresh DQN before learning'))
        self.writer = SummaryWriter(str(self.run/'tensorboard'))

    @property
    def epsilon(self):
        fraction = min(1., self.steps/self.cfg['epsilon_decay_steps'])
        initial = self.cfg.get('epsilon_initial',1.)
        return initial + fraction*(self.cfg['epsilon_final']-initial)

    def begin_episode(self):
        self.explore_remaining = 0

    @torch.no_grad()
    def decision(self, features):
        x = F.layer_norm(features, features.shape)
        if self.explore_remaining:
            action = self.explore_action
            self.explore_remaining -= 1
        elif self.rng.random() < self.epsilon:
            action = int(self.rng.integers(36))
            self.explore_action = action
            self.explore_remaining = self.cfg.get('exploration_hold_steps',1)-1
        else:
            action = int(self.head(x).argmax())
        return action, (x.cpu().numpy().copy(), action)

    def learn(self):
        beta = min(1., self.cfg['priority_beta'] + (1-self.cfg['priority_beta'])*self.steps/self.cfg['epsilon_decay_steps'])
        indices, weights = self.memory.sample(self.cfg['batch_size'], self.cfg['priority_alpha'], beta)
        batch = self.memory.batch(indices)
        tensor = lambda a:torch.as_tensor(a, device=self.device)
        x = tensor(batch['states'].astype(np.float32))
        nx = tensor(batch['next_states'].astype(np.float32))
        expected = double_targets(self.head, self.target, nx, tensor(batch['rewards'])*self.cfg.get('reward_scale',1.), tensor(batch['discounts']))
        actual = self.head(x).gather(1, tensor(batch['actions'])[:, None]).squeeze(1)
        errors = expected-actual
        loss = (tensor(weights)*F.smooth_l1_loss(actual, expected, reduction='none')).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite DQN loss')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg['max_grad_norm'], error_if_nonfinite=True)
        self.opt.step()
        self.memory.reprioritize(indices, errors.detach().abs().cpu().numpy())
        self.updates += 1
        if self.updates % self.cfg['target_sync_updates'] == 0:
            self.target.load_state_dict(self.head.state_dict())
        return (float(loss.detach()), float(errors.detach().abs().mean()),
                float(weights.mean()), float(weights.max()),
                float(actual.detach().abs().mean()), float(expected.abs().mean()))

    def add_trajectory(self, trajectory, tail, reason):
        terminal = reason != 'time_limit'
        for transition in nstep_transitions(trajectory, tail, terminal, self.cfg['gamma'], self.cfg['n_step']):
            self.memory.add(*transition)

    def end_episode(self, trajectory, tail, reason, elapsed_ticks, seconds, details=None):
        self.explore_remaining = 0
        self.add_trajectory(trajectory, tail, reason)
        self.episodes += 1
        self.steps += len(trajectory)
        metrics = []
        if self.memory.size >= self.cfg['learning_starts']:
            for _ in range(max(1, len(trajectory)//self.cfg['train_every'])):
                metrics.append(self.learn())
        reward = sum(r for _, r in trajectory)
        self.log('episodes', dict(episode=self.episodes, steps=self.steps, length=len(trajectory), reward=reward,
                                 reason=reason, elapsed_ticks=elapsed_ticks, survived_seconds=seconds, **(details or {})))
        row = dict(update=self.updates, episodes=self.episodes, steps=self.steps, mean_return=reward,
                   epsilon=self.epsilon, replay_size=self.memory.size,
                   mean_loss=float(np.mean([x[0] for x in metrics])) if metrics else 0.,
                   mean_td_error=float(np.mean([x[1] for x in metrics])) if metrics else 0.,
                   mean_replay_weight=float(np.mean([x[2] for x in metrics])) if metrics else 0.,
                   max_replay_weight=float(np.max([x[3] for x in metrics])) if metrics else 0.,
                   mean_abs_q=float(np.mean([x[4] for x in metrics])) if metrics else 0.,
                   mean_abs_target=float(np.mean([x[5] for x in metrics])) if metrics else 0.)
        self.log('updates', row)
        for k,v in row.items():
            self.writer.add_scalar('train/'+k, v, self.steps)
        self.writer.flush()
        print(f'{self.cfg["algorithm"].upper()} episode {self.episodes}: {reason}, {seconds:.2f}s, reward={reward:.4f}, epsilon={self.epsilon:.3f}, replay={self.memory.size}, updates={self.updates}', flush=True)
        if self.episodes % 4 == 0 or self.finished():
            self.save_policy('latest', {})
        if self.episodes % 20 == 0:
            self.save_policy(f'episode-{self.episodes:06}', {})
        if reason == 'completion' and not (self.ckpt/'first-training-win.pt').exists():
            self.save_policy('first-training-win', {})

    def should_eval(self):
        return self.last_eval != self.episodes and (self.episodes % self.cfg['eval_every_episodes'] == 0 or self.finished())

    def finish_eval(self, rows):
        mean = float(np.mean([r['reward'] for r in rows]))
        if not np.isfinite(mean):
            raise FloatingPointError('invalid evaluation reward')
        selected = self.best is None or mean > self.best
        self.last_eval = self.episodes
        record = dict(update=self.updates, episode=self.episodes, rank=[mean], mean_reward=mean,
                      completion_rate=sum(r['reason']=='completion' for r in rows)/len(rows),
                      selected=selected, trials=rows)
        if selected:
            self.best = mean
            self.save_policy('best', record)
        # Resume must retain evaluation scheduling and selection, even if not best.
        self.save_policy('latest', {})
        self.log('evaluations', record)
        print(f'{self.cfg["algorithm"].upper()} evaluation episode {self.episodes}: mean task reward={mean:.4f}, selected={selected}', flush=True)

    def finished(self):
        limit = self.cfg['episodes']
        return bool(limit and self.episodes >= limit) or (self.run/'STOP').exists()

    def training_state(self):
        return dict(signature=self.signature, head=self.head.state_dict(), target=self.target.state_dict(),
                    optimizer=self.opt.state_dict(), replay=self.memory.state_dict(),
                    progress=(self.steps,self.episodes,self.updates), best_rank=self.best, last_eval=self.last_eval,
                    action_rng=self.rng.bit_generator.state, rng=torch.get_rng_state(),
                    exploration_hold=(self.explore_remaining,self.explore_action),
                    cuda_rng=torch.cuda.get_rng_state() if self.device=='cuda' else None)

    def save_policy(self, name, selection):
        saved = self.training_state()
        saved['selection'] = selection
        temporary = self.ckpt/(name+'.tmp')
        torch.save(saved, temporary)
        temporary.replace(self.ckpt/(name+'.pt'))

    def log(self, name, row):
        with (self.run/(name+'.jsonl')).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
