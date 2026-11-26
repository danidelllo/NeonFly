"""Advantage-weighted replay with measured complete-episode return targets.

No critic/target-network prediction appears in a return target. Training uses
the agent's own full-level episodes, never human or hand-selected trajectories.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from python.training.reservoir_dqn import Trainer as BaseTrainer


def monte_carlo_returns(rewards, gamma):
    result = np.empty(len(rewards),np.float32)
    carry = 0.
    for i in reversed(range(len(rewards))):
        carry = float(rewards[i])+gamma*carry
        result[i] = carry
    return result


def reconstruct_returns(rewards, discounts, n):
    """Recover full measured returns from nonoverlapping n-step reward blocks.

    R_t(full) = R_t(n-step) + gamma**n R_{t+n}(full). Terminal
    tails have zero discount. Reject tails whose outcome was not observed.
    """
    result = np.empty(len(rewards),np.float32)
    for i in reversed(range(len(rewards))):
        if discounts[i] == 0:
            result[i] = rewards[i]
        elif i+n < len(rewards):
            result[i] = rewards[i]+discounts[i]*result[i+n]
        else:
            raise ValueError('cannot reconstruct unobserved truncated tail')
    return result


def regression_weights(advantages, temperature, max_weight):
    # Centering/normalization and mean-one weights prevent a negative baseline
    # from shrinking every actor gradient toward zero.
    standardized = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-4)
    weights = (standardized/temperature).clamp(-10,math.log(max_weight)).exp()
    return weights/weights.mean()


class Trainer(BaseTrainer):
    algorithm = 'awr'

    @torch.no_grad()
    def decision(self, features):
        x = F.layer_norm(features,features.shape)
        if self.explore_remaining:
            action = self.explore_action
            self.explore_remaining -= 1
        elif self.rng.random() < self.epsilon:
            action = int(self.rng.integers(36))
            self.explore_action = action
            self.explore_remaining = self.cfg.get('exploration_hold_steps',1)-1
        else:
            probability = torch.softmax(self.head(x)[:36],dim=-1).cpu().numpy().astype(np.float64)
            probability /= probability.sum()
            action = int(self.rng.choice(36,p=probability))
        return action,(x.cpu().numpy().copy(),action)

    def add_trajectory(self, trajectory, tail, reason):
        returns = monte_carlo_returns([r for _,r in trajectory],self.cfg['gamma'])
        zero = np.zeros(self.width,np.float32)
        for ((state,action),_),value in zip(trajectory,returns):
            self.memory.add(state,action,float(value),zero,0.)

    def learn(self):
        indices,_ = self.memory.sample(self.cfg['batch_size'],self.cfg['priority_alpha'],1.)
        batch = self.memory.batch(indices)
        tensor = lambda x:torch.as_tensor(x,device=self.device)
        x = tensor(batch['states'].astype(np.float32))
        returns = tensor(batch['rewards'])*self.cfg['reward_scale']
        output = self.head(x)
        value = output[:,36]
        weights = regression_weights(returns-value.detach(),self.cfg['temperature'],self.cfg['max_actor_weight'])
        policy = Categorical(logits=output[:,:36])
        actor_loss = -(weights*policy.log_prob(tensor(batch['actions']))).mean()
        value_loss = F.smooth_l1_loss(value,returns)
        loss = actor_loss + .5*value_loss - self.cfg['entropy_coef']*policy.entropy().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite advantage-weighted regression loss')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(),self.cfg['max_grad_norm'],error_if_nonfinite=True)
        self.opt.step()
        self.updates += 1
        # Replay remains uniform: no circular TD-error prioritization.
        return (float(loss.detach()),float((value.detach()-returns).abs().mean()),
                float(weights.mean()),float(weights.max()),float(value.detach().abs().mean()),float(returns.abs().mean()))

    def log(self,name,row):
        if name == 'updates':
            row = dict(row)
            for source,target in (('mean_td_error','mean_return_error'),('mean_abs_q','mean_abs_value'),
                                  ('mean_abs_target','mean_abs_measured_return'),
                                  ('mean_replay_weight','mean_actor_weight'),('max_replay_weight','max_actor_weight')):
                row[target] = row.pop(source)
        super().log(name,row)
