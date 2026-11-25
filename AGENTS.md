# NeonFly agent handoff

Read this file before operating the game or trainer. Commands below use
PowerShell from the checkout root. Current working setup was verified on
2026-11-25; inspect live state rather than assuming a process/run still exists.

## Objective and user constraints

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
| Repository | `https://github.com/danidelllo/NeonFly.git`, branch `main` |
| Game root | `C:\Program Files (x86)\Steam\steamapps\common\NEONSKY` |
| Steam app ID | `4546400` |
| Engine / loader | Unity 2022.3.62f3, Mono x64; BepInEx **5.4.23.6** |
| Installed bridge | `BepInEx\plugins\NeonFly.dll`, version **0.8.0** |
| Local Python | `.venv\Scripts\python.exe`, Python 3.13 |
| Verified GPU stack | RTX 4070; `torch==2.9.1+cu128` |
| Unity bridge | `127.0.0.1:47651` |
| Sole model/controller server | `http://127.0.0.1:47652/` |
| Observer / preferred viewer | `http://127.0.0.1:47653/` |
| Most recent active run | `flywire-sensor-005` — query live state to confirm |
| Current run configuration | `configs/sensor_ppo.toml`, `configs/sensor_ship.toml` |
| Run state | `checkpoints/<run>/`, `logs/ppo/<run>/` |
| Launch logs | `logs/launch/`; current controls launch uses `flywire-sensor-005-controls-final-{out,err}.txt` |

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
`$gameRoot\BepInEx\LogOutput.log` for bridge 0.8.0 and the listener on 47651.
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
Remove-Item -LiteralPath 'logs\ppo\flywire-sensor-005\STOP' -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-sensor-005 --config configs/sensor_ppo.toml --env-config configs/sensor_ship.toml --resume --continuous --device cuda
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
to start over. `--continuous` bypasses the configured update limit, not checks,
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
- `latest.pt`: full state after each policy update (four training episodes).
  `episode-000020.pt`, etc.: permanent snapshots every 20 training episodes.
  `best.pt`: selected by frozen evaluation; may not exist yet in a new run.
- State includes policy/value, policy optimizer and RNG, RND target/predictor,
  predictor optimizer, replay buffer/RNG/calibration and source/config signatures.
  Full snapshots also support pending batches where saved explicitly.
- Best ranking: completion rate, then faster successful completion, then mean
  survival. Evaluations use task reward only and do not learn.

Request a graceful stop with:

```powershell
New-Item -ItemType File -Force 'logs\ppo\flywire-sensor-005\STOP' | Out-Null
```

STOP is checked between episodes. If paused mid-episode, Play is needed to finish
that episode. Stopping mid-batch loses unsaved episodes since the last update.
For a no-loss handoff, set STOP while the active **non-evaluation** episode number
is divisible by four, before it ends; wait for the update and stop message. Verify
the checkpoint progress before restarting. Do not just kill the trainer/game.

To restore another full snapshot, use `--resume-from checkpoints/<run>/episode-XXXXXX.pt`
instead of `--resume`, keeping the matching run/config. It restores learning state,
not the exact Unity scene. A new episode is started. Keep the current checkpoint
before rolling back. Frozen three-episode best replay, with the trainer stopped:

```powershell
.\.venv\Scripts\python.exe -m python.viewer.serve --replay-best flywire-sensor-005 --device cuda
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
- `RangeSensor.cs`: 32 world-space rays, 12-unit range, clipped to camera entry
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
- `vision/range_sensor.py`: 32x7 ray channels + four body channels = 228 scalars
  mapped to 2,674 existing mechanosensory neurons. This is artificial encoding,
  not a biological radar. All-input tracing totals **14,100 neuron links**.
- `viewer/live.py`: fixed 139,255-cell recurrent graph; three leaky tanh substeps.
  1,415 readout neurons provide current/delta/EMA = 4,245 features. Learned
  nonlinear head: 256 -> 128 -> 36 action scores plus value. No trainable recurrent
  synapses. See `training/heads.py`, `training/reservoir_ppo.py`.
- `training/novelty.py`: capped geometry RND, with predictor replay only. PPO uses
  recent on-policy episodes, **not experience replay**. Offscreen ship states
  still receive no novelty bonus. Rewards/config are in the two sensor TOMLs.
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

Last verified: 30 Python tests and 10,012 C# clipping cases. The live test scripts
`python.tests.live_ship` / `live_ship_controls` **drive and reset the game**;
never run them alongside training. Do not rerun expensive checks for docs-only edits.

| Symptom | Check / response |
| --- | --- |
| Connection refused on 47651 | Game running, correct DLL installed, BepInEx log, version, normal scene. |
| Expected Level_1, found Cutscene | Wait/skip normal cutscene; restart only the failed controller afterward. |
| Viewer serves but no new samples | `/control` may be deliberately paused; otherwise inspect launch stderr and worker. Static graph endpoints are not proof of training. |
| Bridge disconnected / key changed | Game closed/restarted. Do not retry an uncertain action. Restore a valid normal scene and resume the latest full checkpoint. |
| Empty radar near ceiling | Confirm bridge 0.8 installed and restarted. 0.7 blanked every ray when the collider center was outside camera. See `docs/SENSOR_FIX_RUN.md`. |
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
git push origin main
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
