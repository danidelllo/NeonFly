# Recurrent PPO through the fly circuit: flywire-brain-ppo-017

User approved replacing AWR with PPO while keeping the actual trainable brain,
radar/resources preprocessing, single linear readout and rewards unchanged.
Run 016 stopped cleanly at episode 2,631 / 82,623 decisions / 2,631 AWR updates.
All old checkpoints and source snapshots remain. No old activations, replay or
trained parameters are transferred into this fresh on-policy experiment.

## Unchanged architecture

139,255 neurons; one bounded positive gain, bounded bias and bounded update
fraction/response time per cell (417,765 parameters). Anatomical topology,
counts-based weights and transmitter sign convention fixed. Three recurrent
substeps per action. 889 raw radar/ship/resource values drive 2,850 cells. No
visual frame drive, planner, safety controller, teacher or learned input encoder.
Only current activity of 1,415 motor/descending neurons is read, with fixed layer
normalization and one linear layer to 36 action logits plus a training value.
Same +5 destruction / -100 death task rewards; internal scale .01; best model
selected solely by mean raw task reward across five frozen full-level evaluations.

## Learning rule

Collect EIGHT fresh complete full-level episodes with unchanged brain/head
parameters, storing raw drives, actions, behavior log-probabilities and values.
Compute GAE (gamma .999, lambda .95) once using those behavior values; time limits
bootstrap the final value, deaths/completions do not. Normalize signed advantages
across the fresh batch. Negative advantages lower selected-action probabilities;
positive advantages raise them. Clipped PPO ratio .2, value Huber coefficient .5,
entropy .01, gradient norm limit 1, two maximum epochs. Learning rates unchanged
from 016: brain 1e-4, readout 3e-4.

Recompute brain state from each episode start under current parameters, preserving
complete forward memory and truncating gradients every 16 actions. Initial replay
log-probabilities must match behavior log-probabilities within 1e-4, or fail loudly.
No stale hidden states or old replay episodes are sampled into policy updates.

After each proposed joint brain/head step, recompute the current recurrent policy
over the full fresh batch. Sample-based KL estimate must be <= .01. Backtrack
parameter displacement by halves up to eight times. If no step is acceptable,
restore both parameters and Adam state and stop that update. Accepted fractional
steps retain gradient moments; this is an adaptive displacement along Adam's
proposed direction. This is a sampled KL safeguard, not a bound at every possible
unvisited game state. Record accepted epochs/scales/KL and neural gradients/deltas.

## Checkpoints, archive and running

The million-transition disk raw-episode archive remains for auditing and old
snapshots. It is NOT experience replay for PPO. Save pending fresh rollout data,
old probabilities/values and frozen advantages/targets with every checkpoint.
A resume halfway through a batch retains it exactly and collects the remaining
episodes under the same model before updating. Latest each completed episode;
permanent snapshots every 20 regardless of reward; evaluations initially and
every 20 episodes. Evaluation can occur midway through a batch but does not
change parameters. No episode limit. Game remains held during neural training,
with heartbeats between chunks so the abandoned-controller watchdog stays quiet.

```powershell
# Healthy bridge 0.10 in Level_1, no competing controller on 47652:
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-brain-ppo-017 --config configs/plastic_ppo.toml --env-config configs/heavy_combat_ship.toml --device cuda
# Resume: same arguments plus --resume. Stop at next episode boundary:
New-Item -ItemType File -Force 'logs/ppo/flywire-brain-ppo-017/STOP' | Out-Null
```

Remove STOP only to intentionally resume. Keep replay-data with checkpoints.
Logs: logs/launch/flywire-brain-ppo-017-{out,err}.txt. Observer 47653 proxies the
same 1,415 linear readouts and 2,850 inputs; no observer restart needed for this
algorithm change because roles/representation remain the same. Both brain and
linear head restore during frozen-best replay. Source/config checks stay strict.

## Verification

73 Python tests passed. New tests verify negative/positive advantage effects,
eight fresh episodes before an update, real recurrent gradients and neuron
changes, fixed W, exact pending-batch and optimizer resume, and rollback on
forced excessive KL. Existing sensor/input-wiring tests still pass.

Full 139,255-cell CUDA preflight: eight synthetic four-action episodes produced
an accepted PPO update, all three neural groups changed, W stayed bit-identical,
and final approximate KL .00990. Accepted step scales were 1 and .015625,
showing the post-step recurrent policy check caught an excessive displacement.
This is a mechanics check, not gameplay success. Save real first-update evidence
under logs/ppo/flywire-brain-ppo-017/launch-validation.json.
