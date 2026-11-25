# Fixed-connectome overnight protocol: flywire-motion-003

Designed and frozen before the training launch on 25 November 2026. No human
demonstrations, teacher policy, selected-section starts, curriculum, or route
scripts. Each training and evaluation episode reloads the ordinary Level_1
start. Functional bridge checks are software validation, not performance tuning.
Success is an explicit LevelEndTrigger event; survival or a cutoff is not a win.

## Representation

The FlyWire v783 graph is unchanged: 139,255 cells and 15,091,983 directed
connections. Original synapse counts, incoming normalization, assumed GABA
signs, three leaky tanh substeps and reset of activity at episode boundaries are
retained. No gradient enters this graph.

Four RGB 84x84 screenshots produce six fixed 42x42 feature maps: luminance,
red-minus-green, blue-minus-yellow, signed luminance change since the previous
decision, signed change across the four-frame window, and local contrast edges.
Motion includes the entire screen, without selected obstacle regions. All six
maps are mapped to visual cells through a seeded permutation, with full spatial
coverage and balanced duplicate assignments. This is an artificial interface,
not a claim about biological receptive fields.

192 visual neurons carry explicit Unity resource inputs (64 each for normalized
health, bullets and currently selected rocket ammunition). These are clamped
to [-1, 1]. No position, enemy location, route map, future frame, or reward is
fed to the policy. The raw game state remains in environment diagnostics.

The 1,415 descending/motor output activities, their first differences, and their
exponential moving averages form 4,245 features. After layer normalization,
a trainable 256-unit tanh layer and 128-unit tanh layer feed 36 action logits and
one value estimate. Only these readout parameters train. This run starts fresh;
old linear checkpoints have a different encoder and feature shape.

## Fixed reward

- +0.01 per simulated second alive, capped at 300 seconds (+3 total).
- -1 per full health bar of actual damage, using a monotonic damage-event counter.
- -2 on death; +20 on confirmed completion.
- No reward for position, money, ammunition pickups, firing, or saving rockets.
  Currency could encourage farming and a rocket cost could discourage necessary
  combat. Health and ammunition availability are observations instead.
- The episode cuts off after 600 simulated seconds or 8,192 decisions. Neither
  is completion. Cutoffs bootstrap the value; death and completion do not.

The completion bonus exceeds all available survival reward. Paused computation,
training, loading and post-death time do not earn survival reward. Damage is
recorded by the health method so a destroyed Player object cannot erase it.

## Learning and evaluation

PPO collects four full episodes per update and uses four minibatch epochs,
batch size 256, learning rate 0.0003, gamma 0.999, GAE lambda 0.99, clip 0.2,
entropy coefficient 0.02, gradient clipping 0.5, and an approximate-KL early
stop above 0.03. See configs/motion_ppo.toml and configs/motion_ship.toml.
This uses PPO's recent-batch reuse, not a long-term off-policy replay buffer.

Five greedy full-level evaluations run before learning, after update 1 and then
every 16 updates. They do not train the policy. Best checkpoint selection uses
completion rate first, speed of successful completion second, mean survival
third. This is repeated same-level evaluation, not a held-out-level test.
The continuous launch bypasses the 4,096-update limit and runs until stopped.

Latest resumable checkpoint is saved atomically after every update. Best
evaluation policy and the first stochastic training win are saved separately.
Checkpoint-policy update: persistent `episode-000100.pt`, `episode-000120.pt`,
etc. are now saved every 20 training episodes, regardless of score (after the
four-episode PPO update). They include optimizer state and CPU/CUDA sampling
RNG state. New best/first-win saves also include full training state, including
any pending episode batch. Evaluation episodes do not count toward the 20.
The running trainer was stopped cleanly at episode 92/update 23 and resumed
with unchanged learning configuration. The old best at update 16 was upgraded
using its exactly matching numbered checkpoint, preserving its learned weights.
Earlier linear-run checkpoint folders were moved to `.deps/retired-checkpoints`,
outside the active checkpoint directory; current-run history remains.
Configuration, graph fingerprint, representation version and source hashes are
stored in checkpoints. The live graph shows actual MLP action scores; linear
per-neuron contribution lines are omitted for this nonlinear model.

## Control and operation

Bridge 0.6 adds authenticated opcodes 9 (guarded full action) and 10 (normal
Level_1 reset). Simulation runs for the requested action and is then held at the
captured observation boundary. It stays held during Python inference and PPO
updates. A 30-second controller inactivity watchdog releases the hold. Existing
opcodes remain available. Background rendering is enabled; keep the game open
and unminimized. The trainer inhibits idle system sleep while running.

Live dashboard: http://127.0.0.1:47652/

Logs: logs/ppo/flywire-motion-003/{episodes,updates,evaluations}.jsonl

Checkpoints: checkpoints/flywire-motion-003/{initial,latest,best}.pt

Console: logs/launch/flywire-motion-003-{out,err}.txt

Graceful stop (PowerShell, project root):

```powershell
New-Item logs/ppo/flywire-motion-003/STOP -ItemType File
```

Stops at the episode boundary; only completed four-episode PPO updates are
checkpointed. For a resume, first ensure the previous controller has exited,
remove that STOP marker if present, open Level_1, then run:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-motion-003 --config configs/motion_ppo.toml --env-config configs/motion_ship.toml --resume --continuous --device cuda
```

Do not run a second controller against the game. Closing the dashboard does
not stop training. There is no automatic restart after reboot or a failed game
connection. Training performance remains to be established by the saved results.

To restore a particular archive, replace `--resume` with
`--resume-from checkpoints/flywire-motion-003/episode-000100.pt`. The checkpoint
restores the model, optimizer and sampling RNG, then starts a new game episode;
Unity game state is not restored. Use a new `--train-run` name when branching
from an older archive so future logs and checkpoints remain separate.
