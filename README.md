Current run: **flywire-dqn-012**, continuing with **1 million disk-backed replay transitions**. See [disk replay notes](docs/DISK_REPLAY_RUN.md).

Current reward experiment: **flywire-dqn-011** — destruction **+5**, death **-100**. See [reward run notes](docs/HEAVY_REWARD_RUN.md).

Current experiment: **flywire-dqn-010**, fixed connectome + Double/dueling DQN with prioritized replay. See [DQN run notes](docs/DQN_RUN.md) and [agent handoff](AGENTS.md). Best models are selected by mean evaluation task reward.

# NeonFly

**Start here for current setup and agent handoff:** [AGENTS.md](AGENTS.md).
It covers dependency installation, BepInEx/bridge installation, normal game
launch, checkpoint resumption, the Observatory, Pause/Step/Play, and recovery.
The current `flywire-dense-008` run uses `configs/dense_ppo.toml` and
`configs/combat_ship.toml`. Its [sensor design and budget](docs/DENSE_SENSOR_RUN.md)
use 126 rays, with dense front/below sectors and exactly three neuron destinations
per value. [Combat rewards](docs/COMBAT_RUN.md) remain active and reward caps
remain disabled at the user's request. Historical commands below describe earlier runs.

Windows-native experiments in visual control of the Steam game NEONSKY. The research target is a policy constrained by real fly connectivity; the CNN is an engineering control, not the research result.

From the checkout root in PowerShell:

```powershell
python -m venv .venv
.\scripts\Build-Bridge.ps1
.\.venv\Scripts\python.exe -m unittest discover -s python\tests -v
```

The build stages a BepInEx loader under `.deps` and compiles the plugin under `unity-mod\build`. It does not install either into Steam. The observation client uses only the Python standard library.

**Local offline smoke test:** close NEONSKY first. Back up `NeonSky_Data` only if modifying it (not needed here); preserve the existing game saves in `SavesDir`, `%USERPROFILE%\AppData\LocalLow\PlayG Studio\NeonSky` and the `HKCU\Software\PlayG Studio\NeonSky` preferences key if present. For a clean install only, copy `.deps\loader` **contents** (including `.doorstop_version`) into the game root, without replacing existing files. Copy `unity-mod\build\NeonFly.dll` into `BepInEx\plugins`. Do not mix loaders. Launch normally through Steam in offline mode. Then:

```powershell
.\.venv\Scripts\python.exe python\communication\observe.py
.\.venv\Scripts\python.exe python\communication\observe.py --png observation.png
```

Check that scene/frame metadata changes and compare the PNG with the visible game. A connection refusal is a failure, not an empty observation; check `BepInEx\LogOutput.log`. To roll back a previously clean install, close the game, remove only the recorded new loader/plugin files and their generated BepInEx directory; do not remove Steam content or save data.

See [architecture and protocol](docs/ARCHITECTURE.md) and [milestone gates](docs/IMPLEMENTATION_PLAN.md).

**Survival scoring and best replay:** The `flywire-survival-002` run inherited the actor from `flywire-ppo-001` update 10, with a fresh critic and optimizer. Plugin 0.5 timestamps terminal events on Unity's physics clock. Reward is +0.02 per simulated second alive (maximum +1 per episode), -1 on death, and +10 on confirmed completion. Paused time and the post-death animation do not count. The 4,096-decision cutoff is not a win. Compare runs with matched inference timing; the old three-action and action-tick returns aren't comparable to these returns. The Observatory sidebar plots training episode number against return or survived seconds, with individual episodes, a trailing ten-episode mean and separate greedy evaluation points.

At initialization, after update 1, then every four updates and at the final update, the agent plays three greedy evaluation episodes on the same level without learning. `checkpoints\<run>\best.pt` is replaced atomically only on a better evaluation: completion rate first, faster wins on ties, then longer mean survival. `latest.pt` is for training resumption. A first stochastic training win additionally saves its readout separately; neither that nor best-so-far guarantees repeatable wins. Best checkpoints contain the learned head, graph fingerprint, environment settings and selection evidence. Replay with the matching local FlyWire graph and game build. After stopping the current controller, run this from the project root for three live episodes without weight updates:

```powershell
.\.venv\Scripts\python.exe -m python.viewer.serve --replay-best flywire-survival-002 --device cuda
```

**Full-connectome viewer:** With NEONSKY in `Level_1` and no other controller attached, run `.\scripts\Start-Observatory.ps1`. This opens a separate Windows graph window and starts an **untrained** full-size reservoir agent that controls and restarts the real ship. See [setup, graph provenance, performance and limitations](docs/VIEWER.md).

**Connectome learning mode:** Install `python\requirements-training.txt` in the same venv. With `Level_1` open and the old demo producer stopped, run `.\.venv\Scripts\python.exe -m python.viewer.serve --train-run flywire-ppo-001 --device cuda`. Open or refresh `http://127.0.0.1:47652/` for live PPO telemetry. The recurrent graph stays fixed; the policy and value weights learn from 1,415 descending/motor outputs. Each update uses four complete real episodes, clipped PPO and GAE. Truncations bootstrap the final observation; deaths/completions do not. Training continues with the window closed, stops at 128 updates, and saves `checkpoints\<run>\latest.pt` after every update. Create `logs\ppo\<run>\STOP` to stop at the next episode boundary (only completed updates are saved). Remove that marker before continuing with the same command plus `--resume`. Resume starts a fresh game episode and restores the readout, optimizer and sampling RNG; Unity itself is not deterministic. TensorBoard data, episode outcomes, optimizer metrics and the run configuration are in `logs\ppo\<run>`. Online training returns are **not** held-out evidence of improvement.

**Live ship regression (opt in):** From `Level_1` or its game-over screen, run `.\.venv\Scripts\python.exe -m python.tests.live_ship` in the repo root. Requires Pillow. This deliberately restarts the current level, exercises horizontal input, waits for natural death and restarts again. It writes decoded PNG evidence and a JSONL trace under `logs\smoke`. `Level_1` is a ship scene, not a jump test.

**Gymnasium ship environment (requires installed plugin 0.4):** With `Level_1` open, run the commands below in PowerShell. Each episode reloads the level. Only one client should control the game; leave manual input neutral.

```powershell
.\.venv\Scripts\python.exe -m pip install -r python\requirements-env.txt
.\.venv\Scripts\python.exe -m python.evaluation.random_ship --episodes 5 --seed 100
```

`python.environment.ship.ShipEnv` implements `reset` and `step`. Default policy input is four 84x84 grayscale frames, channels first (`uint8`); position and event counters are diagnostic `info`, not observations. Configure durations, RGB, stacking, crop `[left, top, right, bottom]`, normalization, episode length and rewards via `configs\ship.toml`. The [36 ship actions](docs/CONTROLS.md) cover two-axis movement and two independent fire buttons. This is not an on-foot environment. Gymnasium is MIT-licensed; NumPy is BSD-licensed and Pillow uses MIT-CMU.
