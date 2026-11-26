# Repaired learning: flywire-dqn-013

User requested action after 012 stopped at its configured 512 episodes without
passing the gate. The stop was a configured budget, not a crash. New run has
**episodes=0: no episode limit**. It continues until a STOP marker, explicit
interruption, game/worker failure or machine shutdown. No promise of unattended
recovery from crashes. Checkpoints and evaluations remain every 20 episodes.

## Evidence and design fixed before launch

012 finished with 30,914 decisions / 7,272 updates. No frozen evaluation completed
the level. Best mean task reward was -84.6664 at episode 60; final mean -100.9339.
Last-50 training episodes killed 0.56 targets on average versus 1.10 in first 50.

Sampling its final replay at beta=1 showed importance-weight median 0.000341,
10th percentile 0.0000944, maximum 0.0131 in a 2,048-row sample. Priorities ranged
from 0.0000176 to 113.5. A global minimum priority scaled loss weights toward zero
even on high-error transitions (sample mean absolute TD error ~24.37). This is
a concrete optimization weakness, not proof of the only cause of failed play.

A synthetic full-graph diagnostic after eight decisions found motor-output RMS
change 0.01037 for a new front obstacle versus 0.00427 for a visual stripe change.
It establishes radar information reaches outputs; it does not establish that
all required information is separable or that visuals never distract. Recipient
interneurons had stronger radar response (0.03321), but readout population and
mapping are left unchanged to assess the learning repair first. No cropped or
selected gameplay starts, human demos, scripts or gate-specific reward terms.

## Changes

- Prioritized replay mixes 20% uniform samples, with importance probabilities
  calculated for the actual mixture. Normalize weights by the largest weight in
  each sampled batch rather than the entire buffer's rare minimum. At least one
  weight per batch is 1; the uniform component also limits sampling imbalance.
- Scale ALL task rewards by 0.01 inside TD targets. Raw reward logging, +5
  destruction, -100 death and mean-task-reward model selection are unchanged.
  Scale the parent's final value/advantage weights/biases by 0.01. Greedy action
  choices were verified unchanged on 512 actual saved feature vectors.
- One gradient update per collected decision, batched at episode boundaries,
  rather than one per four decisions. Target sync remains every 250 updates.
- Generic epsilon exploration holds its randomly selected action for three
  decisions (~0.48 game seconds). Epsilon is the chance to start a burst when no
  burst is active: 0.30 -> 0.10 over 100,000 new decisions. This is not a gate
  route or a safety controller. Greedy evaluation has no exploratory bursts.
- Retain the one-million disk replay and all 30,914 previous transitions, resetting
  priorities to one because TD values use a new scale. Preserve previous files.
- Start from 012's evaluation-selected best online head; synchronize the target
  to it and reset optimizer moments. New episode/step/update counters start zero.
  Inherit the previous best reward ranking. New baseline evaluations are greedy.
- Log mean/max replay weights and mean absolute Q/target values to distinguish
  optimization progress from repeated behavior. No claim of gate success at launch.

## Operations

Config: configs/repaired_dqn.toml, configs/heavy_combat_ship.toml. Same fixed
connectome, visual maps, radar, input mapping and readout feature population.

```powershell
# Only when bridge is healthy in Level_1 and there is no competing controller:
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-dqn-013 --config configs/repaired_dqn.toml --env-config configs/heavy_combat_ship.toml --resume --device cuda
# Graceful stop at the next episode boundary (Play first if deliberately paused):
New-Item -ItemType File -Force 'logs/ppo/flywire-dqn-013/STOP' | Out-Null
```

No --continuous flag is needed: episodes=0 explicitly disables the budget. Remove
STOP only to intentionally resume. Observer on 47653 follows the controller.
Keep checkpoints/flywire-dqn-013/replay-data with checkpoint backups. Standard
resume source/config checks stay strict. The one-off audited setup command was
python -m python.operations.start_repaired_dqn; do not rerun on this run name.
parent.json records the source checkpoints, signatures and deliberate resets.

Validation: 57 Python tests including mixture probabilities and weight scale,
resume with new sampling settings, infinite-budget semantics, exploration holds,
disk replay restoration and DQN learning. Runtime checks must show actual updates,
finite metrics and nonvanishing replay weights; record in launch-validation.json.

Judge full-level frozen evaluations by reward and report actual completion events.
Do not call longer survival or a repaired gradient proof that the gate is solved.
