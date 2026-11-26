"""Audited recovery from run 012: best actor, all replay, corrected learning."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from python.environment.ship import ShipConfig
from python.training.reservoir_dqn import Trainer


def main():
    torch.set_num_threads(2)
    source = Path('checkpoints/flywire-dqn-012')
    best = torch.load(source/'best.pt',map_location='cpu',weights_only=True)
    latest = torch.load(source/'latest.pt',map_location='cpu',weights_only=True)
    if best['signature'] != latest['signature']:
        raise ValueError('parent checkpoints differ in representation/config')
    audited = {
        'python/viewer/live.py':'ab5b71d06ed360d5af13a7687e63077f297acec14dabae1556e4d7cc864ffc65',
        'python/training/reservoir_dqn.py':'2bba470b672e4de792dac67cdb39adc13ae504fe51d189f8a3fe5b296abfd473',
        'python/training/disk_replay.py':'ba5f71a03a98be13e5b51ad359e4e8a1815396d6b76135ae4fb66804b2e23103'}
    for p,digest in latest['signature']['sources'].items():
        expected = audited.get(p,hashlib.sha256(Path(p).read_bytes()).hexdigest())
        if digest != expected:
            raise ValueError('parent is not the audited run 012: '+p)
    t = Trainer('flywire-dqn-013',latest['signature']['width'],'cuda',
                ShipConfig.load('configs/heavy_combat_ship.toml'),config_path='configs/repaired_dqn.toml')
    for k in ('environment','graph','width','representation','selection'):
        if t.signature[k] != latest['signature'][k]:
            raise ValueError('unexpected representation/reward change')
    weights = {k:v.clone() for k,v in best['head'].items()}
    for name in ('value.weight','value.bias','advantage.weight','advantage.bias'):
        weights[name] *= t.cfg['reward_scale']
    t.head.load_state_dict(weights)
    t.target.load_state_dict(weights)
    replay = dict(latest['replay'])
    replay['directory'] = str(t.memory.directory)
    replay['uniform_fraction'] = t.memory.uniform_fraction
    replay['weight_normalization'] = t.memory.weight_normalization
    replay['priorities'] = torch.ones(replay['size'],dtype=torch.float32)
    replay['max_priority'] = 1.
    for chunk in replay['chunks']:
        shutil.copy2(Path(latest['replay']['directory'])/chunk['name'],t.memory.directory/chunk['name'])
    t.memory.load_state_dict(replay)
    # Verify Q rescaling preserves the parent's greedy policy on actual replay.
    from python.training.heads import make_head
    parent = make_head(t.width,latest['signature']['config'],'cuda')
    parent.load_state_dict(best['head'])
    batch = t.memory.batch(np.arange(min(512,t.memory.size)))
    with torch.no_grad():
        x = torch.as_tensor(batch['states'].astype(np.float32),device='cuda')
        torch.testing.assert_close(t.head(x),parent(x)*.01,rtol=1e-4,atol=1e-5)
        assert torch.equal(t.head(x).argmax(1),parent(x).argmax(1))
    t.best = best['best_rank']
    record = dict(reason='fix vanishing PER loss weights; same brain/inputs/task rewards; best actor with all experience',
                  parent_best=str(source/'best.pt'),parent_latest=str(source/'latest.pt'),
                  best_progress=best['progress'],latest_progress=latest['progress'],replay_size=t.memory.size,
                  optimizer_reset=True,target_reset_to_online=True,q_scale=.01,priority_reset=True,
                  old_signature=latest['signature'],new_signature=t.signature,
                  old_checkpoint_sha256=hashlib.sha256((source/'latest.pt').read_bytes()).hexdigest())
    (t.run/'parent.json').write_text(json.dumps(record,indent=2))
    t.save_policy('initial',record)
    t.save_policy('latest',record)
    t.save_policy('best',best['selection'])
    t.writer.close();t.memory.close()
    print('Prepared repaired learner with',replay['size'],'transitions; initial greedy policy verified; no episode limit.')


if __name__ == '__main__':
    main()
