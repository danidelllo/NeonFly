"""Audited one-time RAM-DQN -> disk replay migration, preserving learning state."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from python.environment.ship import ShipConfig
from python.training.reservoir_dqn import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--config', default='configs/disk_dqn.toml')
    parser.add_argument('--env-config', default='configs/heavy_combat_ship.toml')
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.source, map_location='cpu', weights_only=True)
    old = saved['signature']
    if old['sources']['python/training/reservoir_dqn.py'] != '2d2078c3c1197ebe87d7764e2316cd8abe541c931a5782875a3bef9793ec4e8e':
        raise ValueError('not the audited original RAM DQN trainer')
    trainer = Trainer(args.run, old['width'], args.device, ShipConfig.load(args.env_config), config_path=args.config)
    new = trainer.signature
    strip = lambda c:{k:v for k,v in c.items() if k not in ('replay_capacity','replay_storage')}
    if (strip(old['config']) != strip(new['config']) or
            any(old[k] != new[k] for k in ('environment','graph','width','selection','representation')) or
            any(new['sources'].get(p) != digest for p,digest in old['sources'].items()
                if p != 'python/training/reservoir_dqn.py')):
        raise ValueError('migration changes something other than replay storage/capacity')
    if new['config'].get('replay_storage') != 'disk':
        raise ValueError('target must use disk replay')
    trainer.head.load_state_dict(saved['head'])
    trainer.target.load_state_dict(saved['target'])
    trainer.opt.load_state_dict(saved['optimizer'])
    trainer.memory.load_state_dict(saved['replay'])
    trainer.steps, trainer.episodes, trainer.updates = saved['progress']
    trainer.last_eval, trainer.best = saved['last_eval'],saved['best_rank']
    trainer.rng.bit_generator.state = saved['action_rng']
    torch.set_rng_state(saved['rng'])
    if args.device == 'cuda':
        torch.cuda.set_rng_state(saved['cuda_rng'])
    record = dict(source=str(args.source.resolve()), checkpoint_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  progress=saved['progress'], previous_signature=old, current_signature=new,
                  reason='user-requested one-million disk replay; all collected experience and learner state retained')
    (trainer.run/'parent.json').write_text(json.dumps(record, indent=2))
    trainer.save_policy('initial', record)
    trainer.save_policy('latest', record)
    previous_best = args.source.parent/'best.pt'
    if previous_best.exists():
        best = torch.load(previous_best, map_location='cpu', weights_only=True)
        if best['signature'] != old:
            raise ValueError('best checkpoint signature differs from migration parent')
        from python.training.disk_replay import DiskReplay
        memory = DiskReplay(trainer.memory.capacity, old['width'], trainer.cfg['seed']+1,
                            trainer.ckpt/'replay-data', trainer.cfg['priority_alpha'])
        memory.load_state_dict(best['replay'])
        best['replay'] = memory.state_dict()
        best['signature'] = new
        temporary = trainer.ckpt/'best.tmp'
        torch.save(best, temporary)
        temporary.replace(trainer.ckpt/'best.pt')
        memory.close()
    # Retain evaluation history so the inherited reward ranking stays interpretable.
    previous_logs = Path('logs/ppo')/args.source.parent.name
    for name in ('episodes','updates','evaluations'):
        source = previous_logs/(name+'.jsonl')
        if source.exists():
            lines = [json.loads(line) for line in source.read_text().splitlines()]
            lines = [r for r in lines if r.get('episode',r.get('episodes',0)) <= trainer.episodes]
            (trainer.run/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in lines))
    trainer.writer.close()
    trainer.memory.close()
    print('Migrated', saved['progress'], 'with',trainer.memory.size,'transitions; capacity',trainer.memory.capacity)


if __name__ == '__main__':
    main()
