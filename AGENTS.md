# NeonFly agent handoff

Read this file before operating the game or trainer. Commands below use
PowerShell from the checkout root. Current working setup was verified on
2026-11-27; inspect live state rather than assuming a process/run still exists.

## Objective and user constraints

**Permanent user constraint (2026-11-27):** The fly circuit itself must control
the game. Never add planners, safety controllers, external world models or
teacher policies. Keep preprocessing simple. User explicitly approved training
one gain, bias and time constant per neuron while preserving anatomical wiring
and connection signs, with a SINGLE LINEAR action readout. This supersedes the
older frozen-reservoir/nonlinear-readout objective below. No human demos or
selected-section training. New work belongs on codex/trainable-fly-brain; main
a1a4f5d preserves all earlier fixed-brain work and was pushed before changes.

- Train a policy to complete the user's Steam game **NEONSKY**, using the real
  FlyWire connectome as a **fixed reservoir**, with a learned nonlinear readout.
- No human demonstrations, manual gameplay examples, selected-section starts,
  cherry-picked obstacles, or level-specific action/reward scripts. Design
  changes first, then assess ordinary full-level runs. Unit tests for mechanics
  and protocol correctness are appropriate; they are not success benchmarks.
- Preserve the fixed brain unless the user explicitly requests otherwise.
  Anatomically structured visual mapping was discussed and **deferred**.
- Preserve resumable learning state. Permanent model snapshots are required
  **every 20 training episodes, regardless of score**.
- Do not call survival, a stochastic good episode, or a saved best model proof
  of completion. Inspect completion events and frozen evaluations.
- Do not disturb an existing training run just to inspect it or edit the UI.

## Current setup and important paths

| Item | Value |
| --- | --- |
| Existing checkout | `C:\Users\danik\Documents\GIT\NeonFly` |
| Repository | `https://github.com/danidelllo/NeonFly.git`, branch `codex/trainable-fly-brain` |
| Game root | `C:\Program Files (x86)\Steam\steamapps\common\NEONSKY` |
| Steam app ID | `4546400` |
| Engine / loader | Unity 2022.3.62f3, Mono x64; BepInEx **5.4.23.6** |
| Installed bridge | `BepInEx\plugins\NeonFly.dll`, version **0.10.0** |
| Local Python | `.venv\Scripts\python.exe`, Python 3.13 |
| Verified GPU stack | RTX 4070; `torch==2.9.1+cu128` |
| Unity bridge | `127.0.0.1:47651` |
| Sole model/controller server | `http://127.0.0.1:47652/` |
| Observer / preferred viewer | `http://127.0.0.1:47653/` |
| Most recent active run | `flywire-plastic-016` — query live state to confirm |
| Current run configuration | `configs/plastic_brain.toml`, `configs/heavy_combat_ship.toml` |
| Run state | `checkpoints/<run>/`, `logs/ppo/<run>/` |
| Launch logs | `logs/launch/flywire-plastic-016-{out,err}.txt` |

Current run 016 trains **intrinsic parameters in the fly circuit itself** with
ONE LINEAR action/value readout. See `docs/PLASTIC_BRAIN_RUN.md`. 139,255 cells,
417,765 intrinsic gain/bias/response-time parameters; anatomical weights/signs
fixed. Readout 1,415 motor cells -> 37 linear units (36 actions + training value),
52,392 parameters; no hidden head, no recipient expansion, no delta/EMA features.
Radar/resources preprocessing unchanged, no frames as input. Sequence replay
stores 889 raw sensor values/action, not stale frozen activations. Replay complete
episodes from zero with CURRENT brain parameters, detach gradients every 16
choices while preserving forward state. Train on the new episode plus one old
whole episode; measured-return targets, no bootstrapping. Heartbeat during replay
keeps Unity held. Full brain/head/optimizer/RNG/raw-replay checkpoint latest each
episode, permanent every 20; eval five greedy full-level episodes every 20.
No episode limit. Same +5 destruction/-100 death task rewards. Old 015 stopped
cleanly at 648 episodes / 5,880 updates. No old model/replay transfer. Full CUDA
preflight verified actual return-loss gradients and updates in all three intrinsic
groups with W bit-identical; 67 tests passed. First gameplay updates verified nonzero gradients and actual gain/bias/alpha
changes in saved checkpoints. No gameplay success claimed. Main a1a4f5d was pushed BEFORE branch work; do not merge into main without
user instruction. Use plastic_brain.toml / heavy_combat_ship.toml in commands.

Historical run 015 uses **advantage-weighted measured-return replay**, replacing
unstable DQN. See `docs/AWR_RUN.md`. 014 stopped at episode 760: Q magnitude grew
to ~153 with TD error ~59 despite scaled death reward -1; no recorded completion.
All 21,821 own-agent transitions recovered into full measured discounted returns
using old 5-step blocks plus episode log boundaries. Fresh actor/value/optimizer,
512 CUDA replay pretraining updates (value magnitude ~.97, measured returns ~.97).
NO critic/target prediction enters a return target. Uniform disk replay capacity
one million, unchanged radar/resources-only fixed brain and 7,317 features.
Same raw task rewards (+5 destruction/-100 death), best by mean evaluation task
reward. Train stochastic categorical actor with advantage-weighted regression,
value Huber loss, entropy .01; epsilon .1 -> .05 generic 3-action bursts. No human
demos, chosen sections or gate scripts. episodes=0, unlimited until STOP/failure;
permanent snapshots/evaluations every 20. Use radar_awr.toml / heavy_combat_ship.toml.
Keep replay-data with checkpoints. Legacy target checkpoint field is inert,
not used by AWR learning. New update logs report measured-return error and
value magnitude rather than Q/TD error. 64 tests and 512 full-width CUDA updates
passed. Initial fit stability is not evidence of gate success. Old sources/models
remain; strict resume signatures unchanged. Observer needs refresh for AWR label.

Historical run 014 is **radar/resources only**, a fresh representation after 013
stopped at episode 592 without resolving the user's reported gate problem. See
`docs/RADAR_RUN.md`. Encoder rays-resources-v1 disables all visual drive exactly;
images remain display/transport only. Direct inputs: 2,658 radar/ship cells plus
192 resource cells. Fixed connectome/synapses unchanged. Readout observes 2,439
neurons (1,415 original + 1,024 anatomically selected sensor recipients), giving
7,317 current/delta/EMA features. No raw-sensor bypass and no trainable neurons.
Fresh head/optimizer/replay; old feature replay is incompatible. Same repaired
DQN and +5 destruction / -100 death. Epsilon 1 -> .1 over 50k decisions, generic
three-action exploration holds. No episode limit; snapshots/evals every 20.
Disk capacity one million, ~29.284 GB at active capacity; keep replay-data with
checkpoints. Viewer must refresh after observer restart: visual cards hidden,
resources/radar clickable, /outputs now returns 2,439 neurons. Observer proxies
active mappings from main controller. 60 tests plus 7,317-feature CUDA replay
preflight passed. Historical runs require original configs/source snapshots.

Historical run 013 repairs replay learning after 012 exhausted 512 episodes with
no completion. See `docs/REPAIRED_DQN_RUN.md`. **No episode limit**: episodes=0;
run until explicitly stopped (or failure/shutdown). Keep 20-episode permanent
checkpoints and evaluations. Use configs/repaired_dqn.toml and heavy_combat_ship.toml.
Fixed brain, inputs and mapping stay intact. 012's best actor is rescaled by .01
(preserves action ranking); all 30,914 transitions retained, priorities reset,
optimizer fresh and target synchronized. New counters start zero. TD rewards
scaled .01, raw task reward/ranking unchanged. PER weights normalize per batch,
20% uniform replay, one update per decision, generic 3-decision exploratory
bursts. Epsilon .30 -> .10 over 100k new decisions. Actual logs showed old global
importance weights near 1e-4–1e-3, suppressing updates. Radar perturbations reached
motor outputs, so visuals were not removed blindly. This is a learning repair,
NOT evidence of gameplay success. Keep replay-data with models. Old source and
checkpoints remain; strict resume checks are unchanged. 57 tests passed.

Historical run 012 continues 011 with **1,000,000 disk-backed replay transitions**
(user superseded earlier 20-million request). See `docs/DISK_REPLAY_RUN.md`.
All 593 source transitions, networks/optimizer/RNG and counters were migrated
at episode 20. Capacity-only continuation, not a fresh experiment. Packed disk
feature data grows to about 17 GB for a million active transitions; checkpoints
share immutable chunk files. KEEP `checkpoints/flywire-dqn-012/replay-data` with
all model checkpoints. Older checkpoints remain restorable after newer writes.
No automatic chunk deletion; historical checkpoint retention can exceed active
capacity on disk. Priorities/tree indexes use about 38 MB RAM plus sampled pages.
Use `configs/disk_dqn.toml`, `configs/heavy_combat_ship.toml`, run 012 in current
commands. Exact reward structure remains destruction +5, death -100. A full-width
packed-record stride issue was fixed before learning; original checkpoints and
an explicit audit are in before-contiguous-fix/ and contiguous-fix-audit.json.
Latest launch logs are flywire-dqn-012-resume-{out,err}.txt. 56 tests and full-width
CUDA disk-read check passed. Standard resume signature checks remain strict.

Historical run 011 uses user-requested rewards: destruction +5, death -100;
all other task coefficients unchanged. Fresh learner/replay with same DQN and
fixed connectome/input mapper. See `docs/HEAVY_REWARD_RUN.md`. Old DQN-010
lost its game bridge at episode 320; checkpoint 320 / gradient update 3804 is
preserved. Never transfer old-scale replay into the new run. Current operational
commands below use 011 and heavy_combat_ship.toml. Historical experiments need
their own original environment config, not the current heavy reward config.

The DQN architecture is **Double/dueling DQN with prioritized experience replay**,
requested by the user. See `docs/DQN_RUN.md` and `configs/dqn.toml`. Fresh
readout, target network, optimizer and replay; fixed connectome/mapping intact.
Replay contains task rewards only; RND/coverage bonuses disabled. Epsilon-greedy
exploration 1.0 -> 0.05 over 25,000 decisions. Budget 512 training episodes;
five greedy evaluations every 20 episodes, initially and finally. Best selection
is **highest mean evaluation task reward only** (no survival/completion override).
Full replay/target/optimizer/RNG checkpoints every 20 episodes regardless of
score; latest every four, after evaluations and on STOP. Resume strictly with
matching source/config. UI labels Q values and DQN, not PPO probabilities.
Clean-009 stopped at episode 278 / update 69; latest checkpoint is episode 276.
Two later logged episodes were an unsaved partial PPO batch. Preserve all old
models and source snapshots. Old PPO resumptions require their old source files;
do not bypass signature checks. Current commands below use DQN-010.

Historical run 008 had a **512-training-episode budget**; do not add `--continuous` or
retune it mid-budget. See `docs/DENSE_SENSOR_RUN.md`. It inherits combat-007's
evaluation-selected readout/optimizer but resets RND for the new 506-dimensional
geometry vector. The user requested 126 rays: front 55, top 8, behind 8, below 55
in four 90-degree sectors. Every ray/body channel has exactly three neuron
destinations (2,658 mapped, 16 spare). Bridge 0.10 exports the actual angles.
Combat-007 stopped cleanly at episode 100 / update 25 for this requested change.
The user previously explicitly overrode the prior
deferred handoff and requested an immediate launch with combat/exploration
reward caps removed. Combat aggregate, exploration per-step and exploration
per-episode caps are all disabled (zero settings); duplicate credit guards stay.
Coverage-006 stopped early at episode 340 / update 85, not its planned 512.
Run 005 stopped at episode 552 / update 138. All old checkpoints are preserved.
The old `python.operations.combat_handoff` helper is **cancelled**. Do not
restart it or launch a second controller. Bridge 0.10 is installed already.

Dependencies, datasets, builds, checkpoints and logs are ignored by Git. A fresh
clone does **not** contain the trained model or FlyWire numerical data. Do not
assume a missing local checkpoint can be restored from GitHub.

## First actions in a new session: inspect, do not relaunch

```powershell
git status --short
Get-NetTCPConnection -LocalPort 47651,47652,47653 -State Listen -ErrorAction SilentlyContinue |
    Select-Object LocalPort,OwningProcess
Get-Process NeonSky -ErrorAction SilentlyContinue
# Only if controller port 47652 exists:
$progress = Invoke-RestMethod 'http://127.0.0.1:47652/progress'
$progress.run
$progress.episodes | Select-Object -Last 1
$control = Invoke-RestMethod 'http://127.0.0.1:47652/control'
$control | Select-Object state,pauseRequested,completedActions,supported,error
```

`/control` also returns a per-process token: do not print or commit it.
`/meta` describes the static connectome; a successful `/meta` response alone
does **not** prove the learning worker is alive. Check `/control`, fresh `/frame`
samples and launch logs. Rendering frame counts can rise while physics is held.
For actual freeze checks use `physicsSeconds`, `fixedUpdates` and positions.

Never attach a second Gym client, random agent, or controller to the same game.
The observer is safe to attach to an existing controller. Its Pause/Step/Play
buttons forward commands to that controller; it does not create another agent.

## Fresh machine / missing environment

The installed Steam game supplies Unity reference assemblies. The bridge builds
with Windows' .NET Framework compiler; a .NET SDK and CUDA toolkit are not needed
for this verified setup. Install Python 3.13 and the game first.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r python\requirements-training.txt
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m python.viewer.prepare
```

`prepare` downloads pinned public FAFB v783 tables, verifies their MD5 hashes,
and builds the local graph indices in `.deps/flywire783`. Run it before the full
unit suite: sensor mapping tests require these data. Let failures surface rather
than substituting another graph. Optional offline game-scene inspection uses
`UnityPy==1.25.2`, not required to run training:

```powershell
.\.venv\Scripts\python.exe -m pip install UnityPy==1.25.2
.\.venv\Scripts\python.exe scripts\inspect_scenes.py 'C:\Program Files (x86)\Steam\steamapps\common\NEONSKY\NeonSky_Data'
```

## Build and install BepInEx / the bridge

Close the game before replacing its plugin. If training is running, finish a
checkpoint boundary first (see below). Preserve existing saves and any existing
mod installation. Never overwrite another loader or modify `Assembly-CSharp.dll`.

```powershell
$gameRoot = 'C:\Program Files (x86)\Steam\steamapps\common\NEONSKY'
.\scripts\Build-Bridge.ps1 -Game $gameRoot
.\scripts\Test-RangeClip.ps1
```

The build script downloads the official BepInEx Windows x64 5.4.23.6 ZIP,
verifies its pinned SHA-256, stages it in `.deps/loader`, and builds
`unity-mod/build/NeonFly.dll`. **Building does not install it.**

For a **clean game installation only**, install the staged loader, including
hidden files such as `.doorstop_version`. The preflight below refuses all
existing destination entries; do not remove that guard to overwrite another mod:

```powershell
if (Get-Process NeonSky -ErrorAction SilentlyContinue) { throw 'Close NEONSKY first' }
$loaderItems = Get-ChildItem -LiteralPath '.deps\loader' -Force
foreach ($item in $loaderItems) {
    if (Test-Path -LiteralPath (Join-Path $gameRoot $item.Name)) {
        throw "Existing loader entry: $($item.Name). Inspect the installation first."
    }
}
$loaderItems | Copy-Item -Destination $gameRoot -Recurse
New-Item -ItemType Directory -Force "$gameRoot\BepInEx\plugins" | Out-Null
Copy-Item -LiteralPath 'unity-mod\build\NeonFly.dll' -Destination "$gameRoot\BepInEx\plugins\NeonFly.dll"
```

When BepInEx is **already installed**, skip the loader copy. With the game closed,
back up the old plugin and copy only the new NeonFly DLL:

```powershell
if (Get-Process NeonSky -ErrorAction SilentlyContinue) { throw 'Close NEONSKY first' }
$backupDir = New-Item -ItemType Directory -Force ('.deps\backups\bridge-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
Copy-Item -LiteralPath "$gameRoot\BepInEx\plugins\NeonFly.dll" -Destination $backupDir.FullName
Copy-Item -LiteralPath 'unity-mod\build\NeonFly.dll' -Destination "$gameRoot\BepInEx\plugins\NeonFly.dll"
```

Then launch normally through Steam and check
`$gameRoot\BepInEx\LogOutput.log` for bridge 0.10.0 and the listener on 47651.
The runtime authentication key is `%LOCALAPPDATA%\NeonFly\session.key`; never
print, copy to Git, or replace it manually. It changes on every game launch.
Source changes to Unity code require build, install, **and game restart**.

Saves may exist in the game's `SavesDir`,
`%USERPROFILE%\AppData\LocalLow\PlayG Studio\NeonSky`, and the
`HKCU\Software\PlayG Studio\NeonSky` preferences key. Do not clear them casually.

## Launch the game and reach the normal level start

```powershell
Start-Process 'C:\Program Files (x86)\Steam\steam.exe' -ArgumentList '-applaunch','4546400' -WindowStyle Hidden
# Once port 47651 is listening:
.\.venv\Scripts\python.exe -m python.communication.observe
```

The last command is read-only status; `--png logs/observation.png` additionally
captures the current composited screen (create the logs directory first).
Use normal game menus to enter **Level_1**. Its game-over screen is acceptable
because the trainer resets it. Do not start training during opening cutscenes.

Existing guarded menu operations are available for the known clean-save flow:

```powershell
# Only MainMenu with no existing save; the bridge rejects other conditions:
.\.venv\Scripts\python.exe -c "from python.communication.observe import request; m,_=request(3); print(m['scene'])"
# Only LevelSelect; selects the ordinary first level:
.\.venv\Scripts\python.exe -c "from python.communication.observe import request; m,_=request(4); print(m['scene'])"
```

These can return an opening cutscene. Observe again and wait/skip it through the
normal UI until status reports Level_1. Do not add teleport/scene-bypass logic
to make a launch succeed. Do not use New Game to overwrite an existing save.

## Resume training, then open the Observatory

Only when **no listener/controller exists on 47652**, the game bridge is healthy,
and status is Level_1:

```powershell
# A STOP marker is an intentional stop request. Remove it only when resuming.
Remove-Item -LiteralPath 'logs\ppo\flywire-plastic-016\STOP' -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-plastic-016 --config configs/plastic_brain.toml --env-config configs/heavy_combat_ship.toml --resume --device cuda
```

This foreground command stays running. Use another shell to open the viewer:

```powershell
# First confirm port 47652 belongs to the intended trainer:
(Invoke-RestMethod 'http://127.0.0.1:47652/progress').run
.\scripts\Start-Observatory.ps1
```

**Launcher trap:** if 47652 is absent, `Start-Observatory.ps1` launches an
**untrained demo agent** that controls/resets the game. It does not discover and
resume checkpoints. Start the intended trainer first. With an existing trainer,
the script creates/reuses observer 47653 and opens an isolated Edge app window.

To launch only the observer without Edge (after the trainer exists):

```powershell
.\.venv\Scripts\python.exe -m python.viewer.serve --observe --port 47653
```

Open `http://127.0.0.1:47653/`. The observer expires after 120 seconds without
requests; the healthy training process continues with all browser windows closed.
For an agent-launched background trainer, use `Start-Process -WindowStyle Hidden`
with the same argument list and `-WorkingDirectory` set to the checkout root;
redirect stdout/stderr to unique files under `logs/launch`. A venv launcher PID
can differ from the actual listener and multiprocessing worker PIDs. Discover
and verify command lines rather than reusing old PIDs.

Fresh training on a new checkout: use a **new unique run name** and omit
`--resume`; keep both sensor config arguments. Never reuse an existing run name
to start over. For other deliberately uncapped runs, `--continuous` bypasses the configured update limit, not checks,
evaluation or checkpointing. `Continue-Training.ps1` is a legacy capped-run
handoff using default configs; do not use it for the current sensor run.

## Pause, step, stop, and checkpoints

- Game Status: **Pause** holds physics at an action boundary; an in-flight
  action finishes first. **Step** executes one model-selected action (8 physics
  ticks, normally 0.16 simulated seconds), then pauses. **Play** resumes training.
- The old Freeze view/Resume view controls are removed. Telemetry stays live
  while paused. Brain camera orbit/pan/zoom remains usable.
- The model worker renews the hold once per second; deliberate pauses survive
  the bridge's 30-second abandoned-controller watchdog. Closing the browser does
  not resume a paused trainer. If the worker dies, the watchdog releases its hold.
- Terminal Step shows the final state and holds before reset. A subsequent Step
  can reset the episode, then execute one action. Terminal activity is from the
  last decision; bars saturate lethal negative health for display only.
- DQN `latest.pt`: full state every four training episodes, after evaluations
  and on graceful STOP. PPO historical latest was saved per four-episode update.
  `episode-000020.pt`, etc.: permanent snapshots every 20 training episodes.
  `best.pt`: selected by frozen evaluation; may not exist yet in a new run.
- DQN state includes online/target Q heads, optimizer, replay features/targets/
  priorities/cursor/RNG, action and Torch RNGs, evaluation ranking/scheduling,
  counters and source/config signatures. Historical PPO included novelty state.
- Best ranking: highest mean evaluation task reward only. Completion and
  survival are diagnostics. Evaluations do not learn or add replay.

Request a graceful stop with:

```powershell
New-Item -ItemType File -Force 'logs\ppo\flywire-plastic-016\STOP' | Out-Null
```

STOP is checked between episodes. If paused mid-episode, Play is needed to finish
that episode. DQN saves latest at the training episode boundary on STOP, including
replay and optimizer state. A STOP during evaluation retains the last completed
training state. Verify checkpoint progress before restarting; do not kill the
trainer/game. The historical PPO worker only saved complete four-episode batches.

To restore another full snapshot, use `--resume-from checkpoints/<run>/episode-XXXXXX.pt`
instead of `--resume`, keeping the matching run/config. It restores learning state,
not the exact Unity scene. A new episode is started. Keep the current checkpoint
before rolling back. Frozen three-episode best replay, with the trainer stopped:

```powershell
.\.venv\Scripts\python.exe -m python.viewer.serve --replay-best flywire-plastic-016 --device cuda
```

Source signatures are byte-sensitive. CRLF/LF conversion on another checkout can
invalidate an old checkpoint. Do not silently rewrite signatures, discard the
optimizer, or bypass compatibility checks. Audit a migration and keep the old
checkpoint/source snapshot. `migrate_sensor_checkpoint.py` is a **one-off**
004-to-005 sensor-fix migration, not a general resume tool. The current
checkpoint-only viewer-control predecessor migration is explicit in
`compatible_signature`; it does not authorize arbitrary model changes.

## Architecture and source map

- `unity-mod/src/Bridge.cs`: authenticated loopback requests and end-of-frame
  screenshots; controlled steps hold `Time.timeScale=0` between actions.
  `GameAdapter.cs`: full ship controls, normal resets, death/completion events.
- `RangeSensor.cs`: 126 world-space rays (front/below 55 each, top/back 8 each),
  12-unit range, clipped to camera entry
  **and exit**. Offscreen ship origins still sense visible segments. Never query
  offscreen geometry. Added `visibleStart` telemetry in bridge 0.8.0.
- Categories: solid; damageable via `EnemyHealth`; enumerated hostile projectile
  classes; lethal via `DeathZone`. Own ship/projectiles and unrelated triggers
  are ignored. Small objects can fall between thin rays. Background artwork is
  not collision geometry. This is a remaining coverage limitation, not a fixed bug.
- `python/communication/observe.py`: bridge protocol/guarded requests; reconstructs
  nested rays from flattened Unity JSON. `environment/ship.py`: Gym environment,
  full-level resets, action guards, simulation-time rewards and terminal handling.
- `vision/connectome.py`: four RGB 84x84 frames -> six 42x42 maps (brightness,
  red-green, blue-yellow, short/long motion, edges). Seeded artificial mapping to
  11,234 visual neurons plus 64 each for health/bullets/rockets (192 total).
- `vision/range_sensor.py`: 126x7 ray channels + four body channels = 886 scalars,
  exactly three copies onto 2,658 existing mechanosensory neurons. Sixteen spare
  cells stay in the recurrent graph without direct drive. This is artificial
  encoding, not a biological radar. All-input tracing totals **14,084 links**.
  Legacy v1 packets/mapping (32 rays, 2,674 cells) remain decodable with the old
  bridge, but cannot be silently fed the new 126-ray packet. v2 validates angles.
- `viewer/live.py`: fixed 139,255-cell recurrent graph; three leaky tanh substeps.
  1,415 readout neurons provide current/delta/EMA = 4,245 features. Learned
  nonlinear head: 256 -> 128 -> 36 action scores plus value. No trainable recurrent
  synapses. See `training/heads.py`, `training/reservoir_ppo.py`.
- `training/reservoir_dqn.py`: current Double/dueling DQN, n-step returns,
  prioritized replay, importance weighting, target network and full resumable
  state. Store immutable reservoir feature pairs and task rewards only. DQN
  updates at episode boundaries; no trainable recurrent cells.
- `training/novelty.py`: geometry RND with optional caps and predictor replay only. Run 006
  additionally rewards first visits to uniform 12x8 viewport cells per episode,
  with the combined bonus capped at 0.25 per episode. Damaging/death transitions
  earn no bonus. Visible offscreen-origin states count as boundary cells, not
  extra exploration space. PPO uses recent on-policy episodes, **not experience
  replay**. Run 006's entropy coefficient is 0.005 (formerly 0.02). Details and
  exact settings: `docs/COVERAGE_RUN.md`, `configs/coverage_ppo.toml`.
- Run 007 disables aggregate combat and exploration reward caps. Damage reward
  is 0.05 per full target health bar, plus 0.10 per distinct destruction, attributed
  only to player-owned weapons. `Combat.cs` and `CombatLedger.cs` provide the
  counters. Task return includes combat shaping; `base_task_reward` excludes it.
  Evaluation still excludes exploration, and checkpoint ranking is unchanged.
- `viewer/control.py`: shared action-boundary Pause/Step/Play state; worker alone
  contacts Unity. `viewer/serve.py`: main HTTP service and observer proxy.
  POST controls require same Origin, exact loopback Host and session token.
- `viewer/web/`: WebGL viewer, input/radar panels, structural tracing, controls.
  Assets are loaded at **server startup**. Editing JS/CSS requires restarting
  the observer, not training. Edits to worker code require a checkpointed restart.
- `viewer/prepare.py`: graph acquisition/indices. Fixed dataset: FAFB v783,
  139,255 neurons, 15,091,983 directed pairs, 54,492,922 synapses.

Recurrent signs/normalization are engineering assumptions. Coordinates are
representative points, not complete neuron morphologies or measured receptive
fields. Readout lines show structure/activity, not nonlinear causal attribution.
Fly body is a licensed NeuroMechFly v2 model from another specimen, illustratively
aligned; default off. Mapping/readout traces default off. Click a map/resource/
radar panel to filter tracing; Shift-click a visual pixel for exact assignments.

## Verification and troubleshooting

```powershell
# Offline tests: require Python dependencies and prepared FlyWire data.
.\.venv\Scripts\python.exe -m unittest discover -s python/tests
# Build first; exercises actual compiled clipping code without running Unity.
.\scripts\Test-RangeClip.ps1
# Optional full raw-data graph validation (larger/slow):
.\.venv\Scripts\python.exe -m python.tests.verify_viewer_graph
```

Last verified: 51 Python tests, full-width CUDA DQN replay preflight, viewer JS syntax; historical, combat accounting/installed-game IL checks,
126-angle sector checks and 10,012 C# clipping cases. Bridge 0.10 loaded and its
live 126-ray/886-channel/506-geometry packet was validated.
The live test scripts
`python.tests.live_ship` / `live_ship_controls` **drive and reset the game**;
never run them alongside training. Do not rerun expensive checks for docs-only edits.

| Symptom | Check / response |
| --- | --- |
| Connection refused on 47651 | Game running, correct DLL installed, BepInEx log, version, normal scene. |
| Expected Level_1, found Cutscene | Wait/skip normal cutscene; restart only the failed controller afterward. |
| Viewer serves but no new samples | `/control` may be deliberately paused; otherwise inspect launch stderr and worker. Static graph endpoints are not proof of training. |
| Bridge disconnected / key changed | Game closed/restarted. Do not retry an uncertain action. Restore a valid normal scene and resume the latest full checkpoint. |
| Empty radar near ceiling | Confirm bridge 0.8 or newer installed and restarted. 0.7 blanked every ray when the collider center was outside camera. See `docs/SENSOR_FIX_RUN.md`. |
| UI edit not visible | Restart only observer 47653, then reload. Verify owning process command line before stopping it. Never stop 47652 for CSS changes. |
| Observer missing | Reopen launcher only after confirming the intended controller exists; otherwise the launcher creates a demo. |
| Source/config mismatch | Compare checkpoint signature, source snapshot, configs and line endings. Do not suppress the error. |
| CUDA stalls | Check `nvidia-smi`, worker and launch logs. A CUDA restart resolved one observed startup stall; preserve checkpoint and avoid infinite retries. Normal inference was about 5-10 ms, but load varies. |

Never expose these services beyond loopback. Do not commit game binaries/assets,
credentials, saved games, local data, checkpoints or telemetry. The derived fly
mesh **is** tracked with its Apache license and provenance; preserve those files.

## Git and handoff

Work on the existing checkout unless a separate worktree is requested. Keep
training running during documentation/Git operations. Before pushing:

```powershell
git diff --check
git status --short
git add AGENTS.md README.md docs
git commit -m "Document agent setup and operating procedures"
git push -u origin codex/trainable-fly-brain
```

Use a stage list appropriate to the actual task. On this host Git Credential
Manager once waited for sign-in while GitHub CLI was already authenticated.
If that recurs, use the existing CLI credential without printing its token:

```powershell
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin main
```

Do not run `gh auth token` into logs. Local author identity has been configured
from the signed-in account with its GitHub noreply address. Verify remote HEAD
and a clean worktree after pushing. In the final handoff identify the live run,
whether it is playing/paused/stopped, checkpoints preserved, checks performed,
and any unresolved limitations. Older README/viewer sections document earlier
experiments; prefer this file and the actual configs/code for current operations.
