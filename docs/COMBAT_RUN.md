# Active uncapped combat-reward run: flywire-combat-007

The user first authorized generic damage/destruction rewards after coverage-006
remained unsuccessful around episode 256. They then explicitly requested an
immediate new run and removal of the combat and other reward caps. That request
superseded the queued 512-episode handoff: coverage-006 stopped cleanly at episode
340 / update 85, and its checkpoints remain intact. Its planned 512-episode
assessment was not completed. Combat-007 starts from its evaluation-selected best.

## Reward design fixed before launch

- +0.05 for removing a target's full maximum health, prorated by actual health
  removed. A single hit gets immediate credit; repeated calls with no health
  change get none. Overkill is clipped at zero health. Each target contributes
  at most one full bar of damage credit per scene, including if healed.
- +0.10 once for a player-weapon-caused positive-health -> dead transition.
  Every `EnemyHealth` target follows the same rule: enemies, red gates and
  other destructible targets. No gate name, coordinate or special action rule.
- There is **no aggregate combat reward cap**, **no per-step exploration reward
  cap**, and **no per-episode exploration reward cap**. Zero in the three cap
  settings disables clipping; coefficients remain 0.05 damage, 0.10 kill, 0.006
  new coverage cell, and 0.002 RND scaling. Every distinct credited kill is paid.
  Duplicate-kill prevention, per-target health accounting and once-per-cell
  coverage remain; these prevent repeated credit for the same event.
- Existing player damage cost (1.0 per full bar), death cost (-2), completion
  reward (+20), simulation survival reward and reactive action interval stay
  fixed. The existing 300-second survival-reward limit is unchanged. PPO ratio
  clipping and gradient clipping are algorithm safeguards, not reward caps.
- Combat is task shaping and is included in evaluation reward. Exploration
  bonuses are excluded in evaluation. Checkpoint ranking remains completion
  rate, completion speed, then survival; combat totals do not select best.pt.
- Logs separate `base_task_reward` (the prior task reward without combat),
  `combat_damage_fraction`, `combat_kills`, `reward_terms.combat_damage`,
  `reward_terms.combat_kill`, and exploration totals. Do not compare the larger
  shaped total with old returns as evidence of improvement.

## Attribution and limits

Bridge 0.9 marks the GameObjects instantiated by normal `Player.Shoot` and
`Player.ShootRocket`. It observes actual `EnemyHealth.TakeDamage` calls from
their projectile components, preserving the game's original damage method.
Direct bullets, Rocket direct/AOE/radial damage and ElectromagneticWave damage
are instrumented. The installed game's HomingRocket is an **enemy turret
weapon**: its damage is observed through the same wrapper but its unmarked
projectiles earn no credit. It can otherwise kill enemies/walls through friendly
fire; merely counting all EnemyHealth deaths would be incorrect.

Objects disappearing offscreen or being destroyed by collision do not produce
reward. Known death/explosion paths mark already-destroying targets so a later
projectile in the same frame does not earn credit against them. Chain reactions
from unmarked environmental explosions are conservatively not attributed.
Separate BossEye health is not part of this EnemyHealth reward or the current
sensor's damageable category. The ledger counts real health loss rather than
inferring events from a target vanishing from the range sensor.

Telemetry is cumulative across scenes; the environment subtracts its reset
baseline. Counter/schema failures invalidate the transition rather than
inventing a reward. The counters do not become extra policy inputs. The
connectome, visual/range mapping and nonlinear policy architecture are unchanged.

## Budget, transfer and preservation

Combat-007 uses `configs/combat_ppo.toml` and `configs/combat_ship.toml`:
512 training episodes / 128 updates, five greedy evaluations initially, after
update 1, every 16 updates and at the final update. No `--continuous`. Keep all
outcomes and do not retune selected sections based on the red gate.

The starting point is coverage-006's `best.pt`, selected by its existing frozen
evaluation rule before the user-requested early stop (episode 192 / update 48,
mean evaluated survival 17.632 seconds, zero completions). Transfer retains policy/value,
optimizer, RND/replay and RNG state; new run counters and ranking start fresh.
The new reward configuration and source signature are explicit. The combat
warm-start exception checks the exact audited bridge-0.8 and old environment
source hashes; it is not an arbitrary normal-resume bypass. All other encoder
and bridge-control dependencies must match. Permanent snapshots remain every
20 training episodes; latest saves every four. Old runs/checkpoints remain.

## Deployment and cancelled handoff

Bridge 0.9 is now installed. The tested DLL also remains staged in
`.deps/combat-release/NeonFly.dll` with release hashes. The old installed DLL was
backed up under `.deps/backups/before-combat-*`. The old handoff helper was
cancelled by explicit user request; do not requeue it or wait for coverage-006
to reach 512. The remaining historical helper describes its original plan.

```powershell
.\.venv\Scripts\python.exe -m python.operations.combat_handoff --check-only
# Do not run the old coverage-512 handoff for the active combat run.
```

The original handoff checked both update 128 / episode 512 and its final evaluation,
waits for the old controller to exit, backs up the old installed DLL, restarts
the game with bridge 0.9, enters Level_1 through the guarded normal menus and
opening cutscenes, then launches combat-007. It refreshes only the observer.
It never deletes saves, teleports the ship, or starts another controller while
one is attached. A successful old evaluation cancels the proposed replacement.

Safety stops: a STOP marker, failed/replaced old controller, changed release,
foreign listener, existing combat run, failed bridge load, or menu/save guard
blocks the transition and records the reason. An existing save that prevents
the guarded NewGame path is preserved; manually enter Level_1 through normal
menus before using the launch command below. No automated save overwrite.

Historical status/logs: `logs/launch/combat-handoff-{out,err}.txt`,
`logs/launch/combat-handoff-result.json`, and the handoff PID file. A singleton
lock in `.deps/combat-release/handoff.lock` prevents competing handoffs. If a
helper was killed, verify its recorded PID is gone before removing the stale
lock. The helper holds Windows awake while waiting and hands that responsibility
to the new trainer after launch. It is not a recurring automation or retry loop.

Manual launch after installation, with Level_1 and no existing controller:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-combat-007 --config configs/combat_ppo.toml --env-config configs/combat_ship.toml --warm-start checkpoints/flywire-coverage-006/best.pt --device cuda
```

For an existing combat-007 run use `--resume` instead of `--warm-start ...`.
An older coverage checkpoint needs its frozen original source to resume normally;
do not silence the new source/config mismatch. Never use the CPU-only
`combat-transfer-validation` test checkpoint for real CUDA training.

## Validation completed before queuing

45 Python tests pass, including reward deltas, optional caps, disabled-cap
payments above the old budgets, unchanged death
penalties, counter rewind/schema rejection, checkpoint transfer, and handoff
completion/hash guards. C# tests exercise the actual built combat ledger and
rewrite the installed game's real IL for both player spawn methods and seven
damage call sites. All 10,012 range-clipping cases still pass. A CPU-only transfer
check accepted the actual coverage parent and verified preserved weights/replay.
Bridge 0.9 loaded successfully at runtime. Positive gameplay events are checked
through its live counters and ordinary run logs, without selected sections.
No claim of successful level completion is made.

## Immediate-launch validation

The explicit user override cancelled the waiting helper and ended coverage-006
at a saved update, episode 340 / update 85. Bridge 0.9 loaded successfully.
Initial combat-run policy, optimizer and sampling RNG fields matched the selected
parent exactly, source hashes matched, and all three aggregate/step exploration
cap settings were verified zero in initial.pt. The five initial greedy episodes
completed: three earned 0.15 combat reward each from normal gameplay, and all
combat amounts were separately logged from base task returns. These are runtime
reward checks, not evidence of improved performance or level completion.
