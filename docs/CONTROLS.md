# NEONSKY controls: Steam build 26830330

Audited the game's `ControlAction` enum, `SettingsData` defaults and the input reads in `Player`, `RickController`, and the UI controllers. Confirmed the twelve binding values against the running game's settings using bridge status (`python\communication\observe.py`), not inferred from the on-disk JSON, which omitted the key fields.

| Input | Ship (`Player`) | On foot (`RickController`) |
|---|---|---|
| A / D | Left / right, held | Left / right, held |
| W | Up, held | Jump, press edge; buffered and ground-checked |
| S | Down, held | Crouch, held |
| Space | Primary fire, held | Shoot, press edge |
| R | Selected rocket, held | Grenade: press, hold to charge, release |

Escape toggles pause. Menu and level-select navigators also read WASD and arrows; the dialogue controller reads Space and the cutscene controller checks any key when skipping is allowed. These are UI actions, not ship policy outputs. On-foot bindings and semantics are verified from the game, **not** claimed as live on-foot agent control.

## Ship policy and transport

`ShipEnv.action_space` is `Discrete(36)`. Each axis independently uses digit 0 (neutral), 1 (negative: left/down), or 2 (positive: right/up). **ID = horizontal + 3 * vertical + 9 * primary + 18 * rocket**, with each weapon flag either 0 or 1. Examples: 0 release all, 3 down, 6 up, 9 primary only, 18 rocket only, 27 both weapons stationary, 35 up-right with both weapons. The original action IDs 0/1/2 still mean neutral/left/right; three-output checkpoints are not full-action policies.

Plugin 0.4 opcode 8 uses the same credential and episode guard as opcode 7 but replaces its biased horizontal byte with the full action ID. The four-byte `NFL1` header magic, opcode, 32-byte session credential, action byte, duration byte and three little-endian signed int32 guard fields are sent in order. Legacy horizontal opcodes are unchanged. The bridge overrides the two ship movement reads and only the two key reads in `Player.Update`, not Unity's global keyboard state. Outside the action the game's physical input path remains active. Native cooldowns, ammo and selected rocket type remain authoritative; requesting fire does not guarantee a projectile. Status reports live key bindings and ammo in diagnostics, never in policy observations.

## Reproduce the control audit

From `Level_1`, run `.\.venv\Scripts\python.exe -m python.tests.live_ship_controls`. This opt-in regression reloads the level before each action and checks both displacement signs, both ammo deltas and acknowledged physics ticks. In run `logs\controls\1795484161143508600`, all 36 combinations passed, including independent fire and release. In the up-right/both case the ship moved +0.4400 X and +0.2150 Y, spent one bullet and one rocket over four ticks. A separate single-tick test also consumed one of each. Three full-action random episodes (100 decisions) then ended in natural deaths. Those are control checks, not evidence of learning or completion.
