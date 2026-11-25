"""One audited continuation across the 0.7 -> 0.8 visible-ray clipping fix.

Creates a separate run. Preserves learning state, but resets best-score ranking
because observations now differ. Does not weaken normal resume validation.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch


def main():
    old_run, new_run = 'flywire-sensor-004', 'flywire-sensor-005'
    source = Path('checkpoints') / old_run / 'latest.pt'
    saved = torch.load(source, map_location='cpu', weights_only=True)
    if saved['signature']['sources']['unity-mod/src/RangeSensor.cs'] != '7d01c37a8435a08f7c7e30768a8eccbcc98d5f0ae17f0d58169306660e91e4a4':
        raise ValueError('expected the audited pre-fix sensor checkpoint')
    if saved.get('pending_batch') or saved.get('pending_returns'):
        raise ValueError('migration requires a completed policy update')
    previous = copy.deepcopy(saved['signature'])
    changed = []
    allowed = {'unity-mod/src/RangeSensor.cs', 'unity-mod/src/Bridge.cs',
               'python/communication/observe.py', 'python/environment/ship.py',
               'python/vision/range_sensor.py'}
    for name, digest in previous['sources'].items():
        relative = name.removeprefix('unity-mod/')
        snapshot = Path('logs/ppo') / old_run / 'source' / relative
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError(f'original source snapshot mismatch: {name}')
        current = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if current != digest:
            if name not in allowed:
                raise ValueError(f'unaudited source change: {name}')
            changed.append(name)
        saved['signature']['sources'][name] = current
    if set(changed) != allowed:
        raise ValueError(f'unexpected change set: {changed}')
    # Task/representation dimensions, network, optimizers, replay, and RNG stay
    # intact. Previous best scores belong to the old sensor behavior only.
    saved['best_rank'] = None
    saved['last_eval'] = -1
    saved.pop('selection', None)
    record = dict(parent_run=old_run, checkpoint=str(source),
                  checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  progress=saved['progress'], changed_sources=changed,
                  reason='Bridge 0.8: clip each ray to visible entry/exit instead of blanking all off-screen rays',
                  previous_signature=previous, current_signature=saved['signature'],
                  best_rank_reset=True, learning_state_preserved=True)
    saved['sensor_migration'] = record
    checkpoint_dir, log_dir = Path('checkpoints') / new_run, Path('logs/ppo') / new_run
    if checkpoint_dir.exists() or log_dir.exists():
        raise FileExistsError('continuation run already exists')
    checkpoint_dir.mkdir(); log_dir.mkdir()
    torch.save(saved, checkpoint_dir / 'latest.pt')
    torch.save(saved, checkpoint_dir / 'sensor-fix-start.pt')
    (log_dir / 'sensor-migration.json').write_text(json.dumps(record, indent=2))
    (log_dir / 'config.json').write_text(json.dumps(saved['signature'], indent=2))
    print(f'{new_run}: preserved progress {saved["progress"]}, policy/value, both optimizers, RND/replay and RNG')


if __name__ == '__main__':
    main()
