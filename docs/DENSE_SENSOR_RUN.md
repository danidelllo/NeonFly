# Four-sector sensor run: flywire-dense-008

User-requested sensor allocation, fixed before this run starts:

| Sector (world degrees, right = forward) | Rays | Internal spacing |
| --- | ---: | ---: |
| Front, -45 to +45 | 55 | 90/55 = 1.63636 degrees |
| Top, +45 to +135 | 8 | 11.25 degrees |
| Behind, +135 to +225 | 8 | 11.25 degrees |
| Below, +225 to +315 | 55 | 90/55 = 1.63636 degrees |

Angles are sector-bin centers, avoiding duplicate boundary rays. Sectors are
fixed relative to the right-facing ship, not its instantaneous velocity. Arrays
are ordered front, top, behind, below. Maximum range remains 12 world units and
ray entry/exit remains clipped to the current camera rectangle. There are 126
physics ray queries per scan; nearest-hit and stable-geometry returns share them.

Each ray produces seven channels. 126x7 + four body-state channels = 886 values.
Each value drives exactly **three distinct mechanosensory neurons**, using
2,658 of the 2,674 available cells. The remaining 16 stay in the recurrent graph
but receive no direct sensor drive. No neurons or recurrent connections are
removed or trained. There is no fourth copy silently assigned to the first slots.
All-input tracing therefore contains 14,084 links: 11,234 visual + 192 resource
+ 2,658 range/body. The range role overlay displays its actual mapped count.

Bridge 0.10 exports the actual angle array and `quadrants-126-v2` layout label.
The encoder validates them instead of assuming uniformly spaced rays. The radar
draws those same angles, shows the ray count, and exposes group counts in its
tooltip. `color-motion-rays-v2` is a distinct representation; old 32-ray data and
the legacy seeded mapping remain decodable when their matching bridge is used.
Do not silently feed new 126-ray data to an old 32-ray policy.

## Transfer and budget

Combat-007 was stopped at episode 100 / update 25; all checkpoints are preserved.
Dense-008 initializes its readout/value and policy optimizer from combat-007's
evaluation-selected best checkpoint. The audited v1-to-v2 warm-start path checks
the old sensor/bridge/environment source hashes and records the representation
change. Normal resume remains strict. The source graph, visual maps, reward
coefficients, unclipped combat/exploration rewards, action rate and PPO settings
are unchanged.

The RND geometry vector changes from 130 to 506 dimensions (126x4 + screen XY).
Its target/predictor, optimizer, replay and calibration therefore start fresh;
loading the old RND weights would be incompatible. The readout remains 4,245 ->
256 -> 128 -> 36 plus value. New episode/update counters and best ranking start
at zero. Initial evaluation assesses the transferred policy under the new inputs.

Budget: 512 full-level training episodes / 128 updates. Five greedy full-level
evaluations initially, after update 1, every 16 updates and at the end. No demos,
selected sections, scripted movement or changed reward terms. Keep every outcome.
Permanent snapshots remain every 20 training episodes, latest every four.

```powershell
# With bridge 0.10 in Level_1 and no existing controller:
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-dense-008 --config configs/dense_ppo.toml --env-config configs/combat_ship.toml --warm-start checkpoints/flywire-combat-007/best.pt --device cuda
# Later, resume this run with --resume instead of --warm-start ... .
```

Checkpoints and run logs use `flywire-dense-008`; launch stdout/stderr are in
`logs/launch/`. Old bridge binaries are backed up under `.deps/backups/before-dense-*`.
Older training resumes/replays need their matching source/bridge representation.
The prior combat handoff is cancelled and must not be restarted.

Validation before launch: 47 Python tests, compiled 126-angle counts/spacing
checks, 10,012 clipping cases, and the compiled combat-accounting/real-game IL
checks. A CPU-only transfer check retained the actual parent's readout and
confirmed a new empty 506-dimensional novelty buffer. Runtime validation must
confirm the exported angles, exact three-copy mapping, radar and live stepping.
