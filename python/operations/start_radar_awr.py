"""Recover measured episode returns from all of radar-014's own replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from python.environment.ship import ShipConfig
from python.training.disk_replay import DiskReplay
from python.training.reservoir_awr import Trainer, reconstruct_returns


def main():
    torch.set_num_threads(2)
    parent = Path('checkpoints/flywire-radar-014/latest.pt')
    saved = torch.load(parent,map_location='cpu',weights_only=True)
    sig = saved['signature']
    if sig['config']['encoder'] != 'rays-resources-v1' or sig['config']['algorithm'] != 'dqn' or 'return_targets' in sig:
        raise ValueError('expected original radar DQN n-step replay')
    snapshot = Path('logs/ppo/flywire-radar-014/source')
    for p,digest in sig['sources'].items():
        if hashlib.sha256((snapshot/p).read_bytes()).hexdigest() != digest:
            raise ValueError('parent source snapshot mismatch: '+p)
    rows = [json.loads(x) for x in Path('logs/ppo/flywire-radar-014/episodes.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['episode'] <= saved['progress'][1]]
    if sum(r['length'] for r in rows) != saved['replay']['total'] or saved['replay']['total'] != saved['replay']['size']:
        raise ValueError('episode boundaries do not match complete unwrapped replay')
    old_cfg = sig['config']
    old = DiskReplay(old_cfg['replay_capacity'],sig['width'],0,saved['replay']['directory'],
                     old_cfg['priority_alpha'],uniform_fraction=old_cfg['replay_uniform_fraction'],
                     weight_normalization=old_cfg['replay_weight_normalization'])
    old.load_state_dict(saved['replay'])
    t = Trainer('flywire-awr-015',sig['width'],'cuda',ShipConfig.load('configs/heavy_combat_ship.toml'),
                config_path='configs/radar_awr.toml')
    if any(t.signature[k] != sig[k] for k in ('graph','width','representation','environment')):
        raise ValueError('sensor representation or task reward changed')
    zero = np.zeros(sig['width'],np.float32)
    position = 0
    for row in rows:
        if row['reason'] not in ('death','completion'):
            raise ValueError('unobserved time-limit tail; migration needs explicit finite-horizon handling')
        length = row['length']
        batch = old.batch(np.arange(position,position+length))
        returns = reconstruct_returns(batch['rewards'],batch['discounts'],old_cfg['n_step'])
        for i,value in enumerate(returns):
            t.memory.add(batch['states'][i],int(batch['actions'][i]),float(value),zero,0.)
        position += length
    old.close()
    record = dict(parent=str(parent),checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                  old_progress=saved['progress'],recovered_transitions=position,all_episodes_used=len(rows),
                  optimizer_and_head_fresh=True,reason='DQN divergence; measured Monte Carlo returns from every stored full episode',
                  old_signature=sig,new_signature=t.signature)
    (t.run/'parent.json').write_text(json.dumps(record,indent=2))
    # Replay-only optimization on the agent's own observations; no Unity actions.
    for i in range(512):
        metrics = t.learn()
        if (i+1)%64 == 0:
            t.log('pretraining',dict(update=t.updates,loss=metrics[0],return_error=metrics[1],
                                    mean_value=metrics[4],mean_measured_return=metrics[5]))
            print('Measured-return replay pretraining',i+1,'value/return',metrics[4:6],flush=True)
    t.save_policy('initial',record)
    t.save_policy('latest',record)
    t.writer.close();t.memory.close()
    print('Prepared AWR with',position,'transitions from ALL',len(rows),'episodes; no Q bootstrap.')


if __name__ == '__main__':
    main()
