# Visible-ray clipping continuation: flywire-sensor-005

Bridge 0.8 corrects the ceiling blind spot identified in
`logs/sensor-diagnostic/REPORT.md`. It intersects each ray with the camera
viewport and queries only the interval from visible entry to visible exit.
Offscreen origins no longer disable inward-pointing rays. Outward rays with no
visible segment remain empty. Hit distances still originate at the actual ship;
no offscreen collider hits, scene map, or level-specific logic are supplied.

The packet adds `visibleStart` for displaying the clipped segment. The radar
draws that interval and distinguishes an outside ship with visible sensing from
one with no visible ray segments. The policy retains 32 rays, 228 scalar inputs,
the same neuron mapping, and the same network dimensions. Small-object gaps
between thin rays are unchanged. Novelty reward still excludes offscreen ship
states; rewards, PPO settings, and full-level episode starts are unchanged.

The old game had disconnected and stopped learning before the fix. Its latest
completed update was episode 152 / update 38 / 9,742 decisions. The continuation
preserves policy/value weights, policy optimizer, RND target/predictor and
optimizer, 5,225 replay entries, calibration, and random states. Episode/update
counters continue from that checkpoint. The previous run and its checkpoints
remain intact. Episode 153 had been logged but was not part of a saved update.

`scripts/migrate_sensor_checkpoint.py` validates the old source snapshot and
limits the migration to five audited sensor/protocol/version-check files. It
writes a separate checkpoint and migration record without relaxing normal
resume validation. Best-score ranking is reset because sensor observations
changed. Historical scores from run 004 do not compete for run 005's best model.

Checkpoint seed: `checkpoints/flywire-sensor-005/sensor-fix-start.pt`.
Migration record: `logs/ppo/flywire-sensor-005/sensor-migration.json`.
Source snapshot: `logs/ppo/flywire-sensor-005/source`.
Permanent checkpoints remain every 20 training episodes; latest every 4.

Validation: `scripts/Test-RangeClip.ps1` exercises the compiled bridge with
12 boundary/regression cases and 10,000 randomized segments, checking that
visible segments are retained and stay within the camera/range. The Python
suite covers offscreen visible-hit encoding, interval validation and the
existing training/protocol/checkpoint behavior.

Resume, with the game in Level_1 and no other controller:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-sensor-005 --config configs/sensor_ppo.toml --env-config configs/sensor_ship.toml --resume --continuous --device cuda
```

Stop at an episode boundary with `logs/ppo/flywire-sensor-005/STOP`.

Live verification captured an ordinary policy-driven state in episode 154 with
screenY=1.0154753 (ship center above the camera), 15 visible ray segments and
11 collider hits, including solid and damageable categories. The old sensor
returned no segments in such states. Evidence is under
`logs/sensor-diagnostic/fixed/`; no selected-section starts or manual gameplay
were used. The radar displayed "Ship outside · sensing visible scene" with
those returns and no browser errors. All learning-state fields were compared
exactly against the source checkpoint after migration. Source signatures match.

A startup attempt encountered the normal opening cutscene before Level_1 and
was restarted after it finished. A later CUDA worker stalled before a saved
update; its partial episode log was archived under the diagnostic folder and
the unchanged continuation checkpoint was reloaded. The replacement worker was
verified producing advancing samples. Launch logs are in `logs/launch/`.
