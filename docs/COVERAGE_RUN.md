# Bounded episodic coverage: flywire-coverage-006

This design was fixed before launching its ordinary full-level training run.
No demonstrations, selected sections, gate coordinates, forced downward actions,
action overrides, or extra game-state access are used.

## Motivation and decision

Run 005 showed no completions and regressing evaluation survival. At episode
530, its latest 40 training episodes averaged 11.07 seconds versus 15.35 seconds
in its first 40. RND reward averaged only about 0.00003 per recent episode, and
its old offscreen-origin exclusion suppressed bonuses during ceiling flight.
Policy entropy was still around 3.24 nats out of log(36)=3.58. Simply increasing
randomness is therefore not the selected intervention.

The new run retains reactive decisions every eight physics ticks, existing
health/damage/death/completion rewards, all inputs, the fixed connectome, and
the nonlinear head. It combines RND with a small, persistent-within-training
episodic position-coverage incentive and lowers entropy regularization from
0.02 to 0.005. There is no forced action repetition, so the model can react to
walls at every existing decision boundary.

The general idea of episodic novelty alongside lifelong novelty is supported
by [Never Give Up](https://arxiv.org/abs/2002.06038). This implementation uses a
simple explicit screen-position grid, not that paper's learned embeddings,
k-nearest-neighbor memory, multiple policies, or complete algorithm. Screen
position already enters the agent through its range/body channels.

## Exact incentive rules

- Divide the viewport into 12 columns by 8 rows, uniformly and without any
  obstacle-specific labels. Mark the episode's initial cell visited at reset.
- Award up to 0.006 for first arrival in another cell that episode. Hovering,
  repeated visits and bouncing between already visited cells earn no additional
  coverage reward. The visited set resets at each ordinary episode reset.
- Combine it with the existing RND reward. Cap their **combined** reward at
  0.008 per decision and 0.25 per episode. A full health-bar loss still costs
  1.0 and death costs 2.0, so the entire exploration budget cannot offset either.
- No exploration bonus is paid on a damaging transition or a death. A cell
  entered on such a transition is still marked visited, preventing delayed
  credit for the same unsafe arrival. Predictor updates can still learn it.
- With bridge 0.8, a slightly offscreen ship with visible ray segments remains
  eligible. Its position is clamped to the existing boundary cell, creating no
  extra offscreen reward locations. No visible ray segment means no bonus for
  an outside origin. Rays remain limited to visible geometry.
- Evaluations have **zero intrinsic reward** and do not update the predictor.

The grid is an exploration aid, not a progress measure. It may encourage
unnecessary movement or collecting coverage early; bounded rewards and normal
damage costs limit this incentive but do not prove it will solve the level.

## Initialization and predetermined assessment

Run 005 was stopped at a completed PPO batch (episode 552 / update 138). Its
checkpoints remain intact. Run 006 transfers the checkpoint already selected by
the existing frozen-evaluation ranking (`checkpoints/flywire-sensor-005/best.pt`),
not a hand-selected training episode. Its exact hash, parent progress, signature
and transferred state are recorded in the new run's `parent.json`.

The nonlinear policy/value weights, policy optimizer moments, RND target,
predictor, predictor optimizer, replay and sampling RNG are transferred. New
episode/update counters, best ranking, and coverage visits start afresh. Source
and configuration signatures describe the new design; this is an explicitly
different run, not a claim of unchanged checkpoint compatibility. The normal
strict resume path is preserved. New source snapshots live under the run logs.

Budget: **512 training episodes (128 updates)**. Launch without `--continuous`.
Perform five greedy full-level evaluations initially, after update 1, every
16 updates, and at the final update. Keep all outcomes. Rank checkpoints by
completion rate, successful completion speed, then mean survival. Report task
return, completion counts and survival separately from intrinsic reward.
Inspect the full evaluation trend, not only best.pt; a survivor is not a win.
Do not adjust parameters or choose isolated sections mid-budget based on the gate.

Permanent snapshots still save every 20 training episodes; latest saves every
four. Coverage visits, component reward totals and the remaining bonus budget
are checkpointed. Each training episode logs `exploration.new_cells`,
`coverage_reward`, `rnd_reward`, and `blocked_bonus_steps` so failures of the
incentive are observable rather than inferred from total return alone.

## Commands

With bridge 0.8 running in Level_1 and the old controller stopped:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-coverage-006 --config configs/coverage_ppo.toml --env-config configs/sensor_ship.toml --warm-start checkpoints/flywire-sensor-005/best.pt --device cuda
```

Resume the same budget after interruption, using its own checkpoint:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-coverage-006 --config configs/coverage_ppo.toml --env-config configs/sensor_ship.toml --resume --device cuda
```

After the controller is serving 47652, open `scripts/Start-Observatory.ps1`.
Pause/Step/Play are unchanged. Before modifying older runs, restore their exact
source snapshot/commit; the new novelty source intentionally fails their strict
normal-resume signature check. Frozen replay still uses saved configuration.

## Launch verification (not an effectiveness claim)

34 Python tests passed, including no repeated-cell bonus, no bonus on unsafe
arrivals, combined caps, checkpointed visits, and nonlinear warm-start state.
Initial checkpoint policy/optimizer/RND/replay/RNG fields matched the parent
exactly, and current source hashes matched the checkpoint signature. The initial
five greedy evaluations had no completions and mean survival 13.48 seconds.
The first ordinary training episode survived 18.74 seconds and earned 0.048
coverage + 0.027855 RND reward; eight unsafe/terminal bonus opportunities were
blocked. These validate operation, not improvement. No parameters were adjusted
after these observations. Full evidence remains in local run logs.
