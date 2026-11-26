# Intrinsic-neuron learning: flywire-plastic-016

User approved this architecture on 2026-11-27 after explicitly rejecting planners,
safety controllers, external world models, teacher policies and substantial
networks above the brain. Preprocessing must stay simple. Main a1a4f5d was pushed
before this work; implementation branch is codex/trainable-fly-brain.

## What learns

Real FAFB783 topology, incoming-normalized synapse counts and transmitter signs
are fixed. All 139,255 neurons have three trainable intrinsic scalars:

- gain = .25 + 1.75 sigmoid(raw_gain), initialized to 1;
- bias = .5 tanh(raw_bias), initialized to 0;
- update fraction alpha = .05 + .9 sigmoid(raw_alpha), initialized to .6.

Each substep: state <- (1-alpha)*state + alpha*tanh(gain*(W*state + input)+bias).
Three substeps per decision, as before. Alpha controls the effective response
time constant; smaller alpha retains activity longer. Positive gains and fixed
W preserve excitatory/inhibitory signs. No synapse or edge is added or removed.
417,765 trainable neuron parameters. This is an artificial dynamical model
constrained by anatomy, not a claim of matching measured biological physiology.

Read only the original 1,415 motor/descending neurons' current activity. No
expanded recipient taps, delta or EMA features. Fixed layer normalization is
followed by ONE linear layer: 1,415 -> 36 action logits plus one training-only
value estimate (52,392 parameters). No hidden readout network. No controller
bypasses or corrects the chosen actions.

## Inputs and learning

Same simple radar/resource inputs: 126 rays x 7 + 4 ship values + 3 resources =
889 raw scalars, assigned to the same 2,850 anatomical input cells. Frames have
no neural drive. Image transport/display remains available.

Advantage-weighted measured-return training, now through the recurrent brain.
Completed full-level episodes provide measured discounted returns. No critic
prediction enters return targets. Fit value with Huber regression; regress
actions with standardized advantage weights and entropy .01. Separate Adam
rates: intrinsic parameters 1e-4; linear readout 3e-4; global gradient limit 1.

Replay stores immutable raw sensor/action/return episodes on disk, capacity one
million active decisions. Unlike frozen-feature replay, each training pass
recomputes activity from episode start with CURRENT brain parameters. Forward
memory persists through the complete replayed episode. Gradients truncate every
16 decisions (~2.56 simulated seconds), with a single accumulated optimizer
step after the sequences. Full measured-return targets give earlier actions
credit for later consequences beyond the gradient truncation horizon.

Each new episode trains on itself and one uniformly sampled whole past episode.
No human demos, cropped starts, selected sections, route scripts or teacher.
Raw replay costs roughly 3.6 GB per million decisions plus per-episode headers;
older checkpoint references keep their files. Keep replay-data with backups.

No frozen-feature replay or readout weights from old runs are reused. New brain
parameters start at the old circuit dynamics, with a fresh linear readout and
optimizer. Rewards unchanged: destruction +5, death -100, other coefficients in
heavy_combat_ship.toml. Internal return scale .01; selection uses mean RAW task
reward from five greedy full-level evaluations.

## Operations

No episode limit. Evaluate initially and every 20 training episodes. Save full
brain/readout/optimizer/RNG/raw-replay manifests each episode (latest), every 20
episodes permanently regardless of reward, and after evaluation. No exact Unity
scene restoration; resume begins a normal new full-level episode.

```powershell
# Only healthy bridge 0.10 in Level_1 and no competing controller on 47652:
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-plastic-016 --config configs/plastic_brain.toml --env-config configs/heavy_combat_ship.toml --device cuda
# Resume with the same arguments plus --resume.
New-Item -ItemType File -Force 'logs/ppo/flywire-plastic-016/STOP' | Out-Null
```

Remove STOP only to intentionally resume. If paused, Play is required to complete
the active episode for graceful stop. During neural training the game stays
held; status heartbeats between replay chunks prevent the abandoned-controller
watchdog from releasing it. Pause/Step requests apply at action boundaries.

Launch logs logs/launch/flywire-plastic-016-{out,err}.txt. Observer 47653 proxies
active roles and shows 1,415 readouts, zero visual maps, radar/resources, and a
single LINEAR decoder. Frozen best replay restores BOTH brain and head.

Source and graph signatures remain strict. Main preserves old fixed-brain source
and old runs have source snapshots; do not rewrite their checkpoint signatures.

## Verification

67 Python tests cover temporal gradients, parameter updates driven by actual
return losses, unchanged anatomical weights, linear-only decoder, permanent
snapshots, raw-sequence retention and deterministic optimizer resume.

Full 139,255-neuron CUDA test: gradients reached more than 135k neurons in each
intrinsic group. A complete measured-return sequence update changed gain, bias
and alpha, with graph values bit-identical. 417,765 neural + 52,392 linear
parameters; about 1,086 MB peak GPU allocation in the short preflight. This is
mechanics verification, not gameplay evidence. Live first-update gradients and
parameter deltas must be recorded in launch-validation.json. Do not claim gate
success without observed full-level completion/progress evidence.
