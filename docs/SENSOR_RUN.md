# Sensor and novelty run: flywire-sensor-004

This is a fresh, separately checkpointed run. The previous motion-only run was
stopped cleanly at episode 148 / update 37 and its checkpoints were preserved.
No demonstrations, route scripts, selected-section starts, or obstacle-specific
rewards are used. Every training/evaluation episode starts at normal Level_1.

## Fixed observations

The recurrent FlyWire graph, its normalization, assumed signs and neuron dynamics
are unchanged. The six RGB/motion/edge maps and three resource inputs remain.
Bridge 0.7 additionally casts 32 uniformly spaced rays, starting at the ship's
collider center. Angles start at world-right and increase counter-clockwise.
Each ray is limited to 12 world units and clipped to the current orthographic
camera rectangle. No off-screen object positions or level map enter the policy.

Collider categories are solid, damageable (EnemyHealth), hostile projectile
(recognized game projectile component types), or lethal (DeathZone). The ship,
its projectiles and unrelated triggers are ignored. Collider categories are
engineering labels, not image recognition or biological sensory measurements.

For each ray the encoder supplies proximity, closing speed, four category-weighted
proximities and visible-range fraction. Normalized ship velocity and screen
position add four more values: **228 scalar channels**. A fixed seeded mapping
drives all **2,674 measured mechanosensory neurons**, with channels repeated
evenly. No neurons or recurrent edges are fabricated. Readout features remain
the 4,245 current/difference/average features from 1,415 descending/motor cells.

An off-screen ship is a legal game state. It reports no visible returns, zero
visible range, and its clipped screen position; it receives no novelty bonus
for the missing geometry. A missing/invalid sensor on an active visible ship
invalidates the transition rather than inventing an observation.

Actions are held for eight physics ticks (normally 0.16 simulated seconds),
instead of four. The game remains held during inference and optimization.

## Reward and exploration

Task rewards are unchanged: +0.01 per simulated second up to +3 total, -1 per
full health bar damaged, -2 on death and +20 on confirmed completion. There is
no reward for firing, money or saving ammunition. A cutoff is not completion.

Random Network Distillation (RND) adds a small training-only novelty reward.
Its target is a fixed 130→128→32 tanh network. A separate predictor of the same
shape learns those features using Adam at 0.0001. Inputs are per-ray solid,
damageable and lethal proximity, visible range, and the ship's screen position.
A separate geometry ray return ignores projectile occlusion, so bullets do
not create artificial geometric novelty. No image, HUD, clock, projectile
distance, or velocity is used for novelty.

Prediction error is divided by its mean over the first 256 observed training
states, then that scale is frozen. The coefficient is 0.002, with at most
**+0.004 per decision and +0.25 per episode**. The predictor retains an 8,192-state
geometry replay buffer across episodes and updates on batches of up to 32.
This replay is for the novelty predictor; PPO still uses recent on-policy batches.
The connectome and RND target never learn. Evaluation gives no intrinsic reward
and never trains either the policy or novelty predictor.

The PPO settings otherwise match motion-003. Both task and novelty returns are
logged separately. The viewer offers Task return and Survival in addition to
the training return that includes novelty. Best-model selection remains based
on task completion rate, then successful completion speed, then mean survival.

## Checkpoints and operation

`latest.pt` is atomically saved every four training episodes. Persistent
`episode-000020.pt`, `episode-000040.pt`, etc. are saved every 20 training episodes
regardless of score. These and new best/first-win checkpoints include actor/value
weights, policy optimizer/RNG, RND target and predictor, predictor optimizer,
replay buffer and RNG, calibration state, and run/source signatures.

Live viewer: http://127.0.0.1:47653/

The radar beside the six feature maps shows the same inference sample as the
brain. Blue = solid; amber = damageable; red = hostile projectile; violet =
lethal; faint rays are visible space with no relevant hit. Violet rings in the
3D graph mark the mechanosensory input population. Both trace overlays and the
fly body still default to off.

Configuration: `configs/sensor_ship.toml`, `configs/sensor_ppo.toml`.

Graceful stop:

```powershell
New-Item logs/ppo/flywire-sensor-004/STOP -ItemType File
```

Resume with the game in Level_1 and no other controller running (remove the
STOP marker first):

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-sensor-004 --config configs/sensor_ppo.toml --env-config configs/sensor_ship.toml --resume --continuous --device cuda
```

Stopping mid-batch still preserves only the most recent completed policy update.
The source snapshot and launch logs are retained with this run. The first
pre-training launch exposed off-screen sensor handling and was archived under
`.deps/failed-sensor-launch`; no learning updates from it were used.
