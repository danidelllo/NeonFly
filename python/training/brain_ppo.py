"""On-policy recurrent PPO through intrinsic neuron dynamics, linear readout."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from python.training.brain_awr import Trainer as BaseTrainer
from python.training.reservoir_awr import monte_carlo_returns
from python.training.reservoir_ppo import episode_targets


def clipped_actor_loss(new_logp,old_logp,advantage,clip):
    ratio=(new_logp-old_logp).exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)


class Trainer(BaseTrainer):
    algorithm='brain_ppo'

    def __init__(self,*args,**kwargs):
        self.pending=[]
        super().__init__(*args,**kwargs)
        resume=kwargs.get('resume',args[4] if len(args)>4 else False)
        if resume:
            from pathlib import Path
            path=self.ckpt/'latest.pt' if resume is True else Path(resume)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            self.pending=[{k:v.numpy().copy() for k,v in r.items()} for r in saved['pending_rollouts']]

    @torch.no_grad()
    def decision(self,features):
        output=self.head(F.layer_norm(features,features.shape))
        policy=Categorical(logits=output[:36])
        p=policy.probs.cpu().numpy().astype(np.float64);p/=p.sum()
        action=int(self.rng.choice(36,p=p))
        return action,(self.model.last_drive.cpu().numpy().copy(),action,
                       float(policy.log_prob(torch.tensor(action,device=self.device))),float(output[36]))

    def training_state(self):
        saved=super().training_state()
        saved['pending_rollouts']=[{k:torch.from_numpy(v.copy()) for k,v in r.items()} for r in self.pending]
        return saved

    @torch.no_grad()
    def policy_statistics(self,rollouts):
        logps=[];entropies=[]
        for episode in rollouts:
            state=None
            for start in range(0,len(episode['actions']),self.cfg['bptt_steps']):
                stop=start+self.cfg['bptt_steps']
                raw=torch.as_tensor(episode['drives'][start:stop],device=self.device)
                features,state=self.model.forward_sequence(raw,state)
                policy=Categorical(logits=self.head(F.layer_norm(features,(self.width,)))[:,:36])
                actions=torch.as_tensor(episode['actions'][start:stop],device=self.device)
                logps.append(policy.log_prob(actions));entropies.append(policy.entropy())
                self.keep_alive()
        return torch.cat(logps),torch.cat(entropies)

    def update_policy(self):
        rollouts=self.pending
        old=torch.as_tensor(np.concatenate([r['logp'] for r in rollouts]),device=self.device)
        advantage=torch.as_tensor(np.concatenate([r['advantage'] for r in rollouts]),device=self.device)
        returns=torch.as_tensor(np.concatenate([r['target'] for r in rollouts]),device=self.device)
        advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
        total=len(old)
        initial,_=self.policy_statistics(rollouts)
        initial_error=float((initial-old).abs().max())
        if initial_error>1e-4:raise RuntimeError('recurrent acting/replay log probabilities disagree')
        params=list(self.model.parameters())+list(self.head.parameters())
        original={k:p.detach().clone() for k,p in self.model.named_parameters()}
        accepted=0;scales=[];last_kl=0.;gradient={};losses=[]
        for epoch in range(self.cfg['epochs']):
            self.opt.zero_grad(set_to_none=True)
            position=0
            for episode in rollouts:
                state=None
                for start in range(0,len(episode['actions']),self.cfg['bptt_steps']):
                    stop=min(start+self.cfg['bptt_steps'],len(episode['actions']));count=stop-start
                    features,state=self.model.forward_sequence(torch.as_tensor(episode['drives'][start:stop],device=self.device),state)
                    output=self.head(F.layer_norm(features,(self.width,)))
                    policy=Categorical(logits=output[:,:36])
                    actions=torch.as_tensor(episode['actions'][start:stop],device=self.device)
                    sl=slice(position,position+count)
                    actor=clipped_actor_loss(policy.log_prob(actions),old[sl],advantage[sl],self.cfg['clip'])
                    loss=(actor.sum()+self.cfg['value_coef']*F.smooth_l1_loss(output[:,36],returns[sl],reduction='sum')
                          -self.cfg['entropy_coef']*policy.entropy().sum())/total
                    if not torch.isfinite(loss):raise FloatingPointError('non-finite recurrent PPO loss')
                    loss.backward();losses.append(float(loss.detach()))
                    state=state.detach();position+=count;self.keep_alive()
            gradient={k:float(p.grad.norm()) if p.grad is not None else 0. for k,p in self.model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(params,self.cfg['max_grad_norm'],error_if_nonfinite=True)
            before=[p.detach().clone() for p in params];optimizer_before=copy.deepcopy(self.opt.state_dict())
            self.opt.step()
            proposed=[p.detach().clone() for p in params]
            # Check the ACTUAL recurrent policy after the joint brain/head step.
            # Backtrack excessive changes; never accept a step merely because
            # a pre-update KL estimate looked small.
            step_accepted=False
            for attempt in range(self.cfg['max_backtracks']+1):
                scale=.5**attempt
                with torch.no_grad():
                    for p,a,b in zip(params,before,proposed):p.copy_(a+scale*(b-a))
                current,_=self.policy_statistics(rollouts)
                logratio=current-old
                kl=float(((logratio.exp()-1)-logratio).mean())
                if np.isfinite(kl) and kl<=self.cfg['target_kl']:
                    last_kl=kl;scales.append(scale);accepted+=1;step_accepted=True;break
            if not step_accepted:
                with torch.no_grad():
                    for p,a in zip(params,before):p.copy_(a)
                self.opt.load_state_dict(optimizer_before)
                break
        self.updates+=1
        current,entropy=self.policy_statistics(rollouts)
        delta={k:float((p.detach()-original[k]).norm()) for k,p in self.model.named_parameters()}
        row=dict(update=self.updates,episodes=self.episodes,steps=self.steps,algorithm=self.algorithm,
                 rollout_episodes=len(rollouts),sequence_actions=total,initial_logp_error=initial_error,
                 accepted_epochs=accepted,step_scales=scales,approx_kl=last_kl,
                 entropy=float(entropy.mean()),loss=sum(losses),
                 positive_advantages=int((advantage>0).sum()),negative_advantages=int((advantage<0).sum()),
                 neuron_gradient_norms=gradient,neuron_parameter_deltas=delta)
        self.log('updates',row)
        for k,v in delta.items():self.writer.add_scalar('brain-delta/'+k,v,self.steps)
        self.writer.add_scalar('ppo/kl',last_kl,self.steps);self.writer.flush()
        print(f'BRAIN PPO update {self.updates}: episodes={self.episodes}, accepted={accepted}, KL={last_kl:.5f}, neuron deltas={delta}',flush=True)
        self.pending=[]
        return row

    def end_episode(self,trajectory,tail,reason,elapsed_ticks,seconds,details=None):
        drives=np.stack([sample[0] for sample,_ in trajectory])
        actions=np.array([sample[1] for sample,_ in trajectory],np.int64)
        logp=np.array([sample[2] for sample,_ in trajectory],np.float32)
        values=np.array([sample[3] for sample,_ in trajectory],np.float32)
        rewards=np.array([reward for _,reward in trajectory],np.float32)
        bootstrap=0.
        if reason=='time_limit':
            with torch.no_grad():bootstrap=float(self.head(torch.as_tensor(tail,device=self.device))[36])
        advantage,target=episode_targets(rewards*self.cfg['reward_scale'],values,bootstrap,self.cfg['gamma'],self.cfg['gae_lambda'])
        # Disk archive is for audit/checkpoints. Old-policy episodes are not
        # sampled into PPO updates, which use only this fresh pending batch.
        self.memory.add(drives,actions,monte_carlo_returns(rewards,self.cfg['gamma']))
        self.pending.append(dict(drives=drives,actions=actions,logp=logp,values=values,advantage=advantage,target=target))
        self.episodes+=1;self.steps+=len(rewards)
        self.log('episodes',dict(episode=self.episodes,steps=self.steps,length=len(rewards),reward=float(rewards.sum()),
                                reason=reason,elapsed_ticks=elapsed_ticks,survived_seconds=seconds,**(details or {})))
        if len(self.pending)>=self.cfg['episodes_per_update']:self.update_policy()
        print(f'BRAIN PPO episode {self.episodes}: {reason}, {seconds:.2f}s, reward={rewards.sum():.4f}, batch={len(self.pending)}/{self.cfg["episodes_per_update"]}',flush=True)
        self.save_policy('latest',{})
        if self.episodes%20==0:self.save_policy(f'episode-{self.episodes:06}',{})
        if reason=='completion' and not (self.ckpt/'first-training-win.pt').exists():self.save_policy('first-training-win',{})
