# Replay experiment: flywire-dqn-010

User approved DQN with experience replay and reward-only best selection. Start
fresh; no PPO weights or optimizer transfer. Clean-009 stopped at episode 278,
update 69; latest resumable PPO checkpoint is episode 276. Logs include the two
later episodes, whose incomplete four-episode PPO batch was not checkpointed.
All previous checkpoints and the source snapshot are preserved. Historical PPO
requires its matching source snapshot; do not rewrite checkpoint signatures.

The fixed FlyWire graph, input mapping, 126-ray layout, 4,245 readout features,
action interval and task reward coefficients remain unchanged. No neuron or
recurrent connection is trained. The readout is a dueling network with shared
256/128 hidden units, a scalar value and 36 advantages producing 36 Q values.

Double DQN selects bootstrap actions with the online head and evaluates them
with a separate target head. Five-step discounted returns, gamma 0.999. Death
and completion have zero bootstrap; time limits retain bootstrap. Transitions
never cross resets. One gradient update per four collected decisions, batched
at episode boundaries; batch 128, learning rate 1e-4, warmup 1,024 transitions,
target synchronization every 250 gradient updates, gradient norm limit 10.

Prioritized replay: capacity 20,000, exponent 0.6, importance beta 0.4 -> 1 over
25,000 decisions. Sample with replacement and correct the weighted loss;
duplicate indices receive their largest TD error. States and next states are
the actual layer-normalized frozen reservoir features, stored as float16 on
CPU. Thus replay retains temporal context without reconstructing old frames.
The buffer occupies about 340 MB at capacity. Full permanent checkpoints can
use roughly 9 GB over this budget; disk capacity was checked before launch.

Replay stores task rewards only. RND and per-episode coverage bonuses are
disabled for this experiment to avoid replaying expired novelty. Epsilon-greedy
exploration decreases from 1.0 to 0.05 over 25,000 training decisions. Evaluation
is greedy and never writes replay or trains the network. No human examples,
selected sections, scripted movement or gate-specific rewards.

Budget: 512 full-level training episodes. Five full-level evaluations initially,
every 20 training episodes, and at the end. Best selection uses ONLY highest
mean evaluation task reward; ties retain the earlier model. Completion rate
and survival are diagnostics and never override reward ranking. The PPO helper
also now ranks by mean evaluation reward for future PPO experiments.

`initial.pt`, `latest.pt`, `best.pt`, and every `episode-XXXXXX.pt` contain online
and target networks, optimizer, replay data/priorities/cursor/RNG, action RNG,
Torch RNGs, counters, evaluation schedule/ranking, config and source signatures.
Permanent snapshots every 20 training episodes regardless of score. Latest is
saved every four episodes, after evaluations and on graceful STOP. Resume
begins an ordinary new level episode; no exact Unity scene restoration.

With bridge 0.10 in Level_1 and no controller listening on 47652:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-dqn-010 --config configs/dqn.toml --env-config configs/combat_ship.toml --device cuda
# Resume: append --resume, never --warm-start.
# Stop at next episode boundary:
New-Item -ItemType File -Force 'logs/ppo/flywire-dqn-010/STOP' | Out-Null
# Remove that marker only when intentionally resuming.
```

The existing Observatory observer on 47653 can follow the new controller. Its
assets need restarting to display DQN labels. Never launch a demo or second
controller. Start-Observatory.ps1 is safe only after confirming 47652 is this run.
Launch logs: logs/launch/flywire-dqn-010-{out,err}.txt. Model/log paths retain
the historical logs/ppo parent directory even though this is DQN.

Validation: 51 Python tests; n-step terminal/time-limit targets, Double-DQN
action selection, PER wrap/duplicates/importance weights, deterministic replay
and optimizer resume, target synchronization, 20-episode archival, reward-only
best selection. Full-width CUDA preflight trained from a synthetic replay
without contacting Unity; this is a mechanics check, not gameplay evidence.
Both edited viewer modules pass node --check. Live launch checks are recorded
separately under the run's launch-validation.json.
