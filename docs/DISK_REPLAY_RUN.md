# Disk replay: flywire-dqn-012

User requested a larger disk replay, finally specifying **1,000,000 transitions**
(supersedes the earlier 20-million request). Same DQN, fixed connectome, input
mapping and heavy rewards: destruction +5, death -100. Same 512-episode budget;
episode counters continue from 011, not from zero. No reward/learning retuning.

DQN-011 stopped cleanly at episode 20 / 593 decisions / zero gradient updates.
An explicit audited migration retained all 593 transitions, online/target Q heads,
optimizer, action/replay/Torch RNG states, best ranking and evaluation schedule.
Previous best and evaluation history are retained. Original checkpoints remain.
Migration provenance and old/new source signatures are in parent.json. Normal
resume still requires exact signatures; this migration does not relax it.

Use configs/disk_dqn.toml and configs/heavy_combat_ship.toml. Replay stores
immutable append-only binary feature chunks under
checkpoints/flywire-dqn-012/replay-data. Float16 state and next-state features,
int64 action, float32 reward/discount: **16,996 bytes per transition**, about
16.996 GB (15.83 GiB) for one million. Files grow with collected experience;
there is no full-size preallocation. Only sampled rows are read into batches.

RAM holds raw priorities and sum/min segment trees (about 38 MB at this capacity)
plus a bounded memory-map cache. Sampling and priority updates use O(log N)
trees rather than scanning a million priorities for every batch. Replay capacity
is the active million-transition ring window; oldest entries leave that window.

Checkpoints store priority snapshots/RNG/counters and immutable chunk-prefix
manifests, NOT duplicate feature arrays. Flush/fsync precedes checkpoint saves.
Restore validates chunk continuity and file lengths. Appending after restoring
an older checkpoint creates a new UUID-named chunk, preserving later and older
branches. Old chunk files are deliberately retained: they may be referenced by
permanent checkpoints. Total history disk use can therefore exceed active-buffer
size after wrapping or repeated rollbacks. No unaudited automatic deletion.

**Keep replay-data with checkpoints**, including when copying or backing them up.
Manifests currently contain the absolute directory path; moving to another path
needs an explicit audited relocation. A .pt file alone is not a full disk-replay
backup. Permanent model snapshots remain every 20 episodes regardless of score.

Launch/resume (bridge healthy in Level_1, no other controller):

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-dqn-012 --config configs/disk_dqn.toml --env-config configs/heavy_combat_ship.toml --resume --device cuda
# Graceful stop:
New-Item -ItemType File -Force 'logs/ppo/flywire-dqn-012/STOP' | Out-Null
```

Remove STOP only when intentionally resuming. Observer 47653 follows controller
47652. Launch logs: logs/launch/flywire-dqn-012-{out,err}.txt.

One-off migration command already executed; do not rerun on this run name:

```powershell
.\.venv\Scripts\python.exe -m python.operations.migrate_disk_replay --source checkpoints/flywire-dqn-011/latest.pt --run flywire-dqn-012 --config configs/disk_dqn.toml --env-config configs/heavy_combat_ship.toml --device cuda
```

Validation: 56 Python tests including disk growth, wraparound, branch/old-checkpoint
restore, PER probabilities/weights and RNG, RAM migration, truncated file rejection,
actual DQN learning and optimizer resume with disk replay, 20-episode archival,
and reward-only best selection. Live migration/launch evidence is recorded in
logs/ppo/flywire-dqn-012/launch-validation.json.

A full-width CUDA check caught packed int64 field alignment before gradient
learning. Batch fields now become contiguous arrays. Run 012 stopped at episode
25 / 678 transitions for the repair, then resumed unchanged. Original checkpoint
files remain in before-contiguous-fix; contiguous-fix-audit.json documents the
exact old/new source digests. Storage layout and transition contents unchanged.
