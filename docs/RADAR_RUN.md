# Radar-only experiment: flywire-radar-014

User requested a different approach after run 013 failed to progress. Stop 013
cleanly at episode 592 / 30,401 new decisions / 30,401 updates; preserve its
checkpoints and all earlier runs. This is a NEW representation, not a resume or
checkpoint-signature migration. Start the model, target, optimizer, replay,
counters and best ranking fresh. Old frozen features cannot be reused.

## Representation

Encoder `rays-resources-v1`: no visual image values enter the connectome. The six
visual drives are disabled exactly. PNG screenshots/frame stacks remain available
for display and environment transport, not policy decisions. Tests compare black
and white frame stacks and require bit-identical brain activity/features.

Direct inputs: 126 radar rays x seven channels plus four ship velocity/position
values, each mapped to three mechanosensory neurons (2,658 total). Health, bullets
and rockets retain 64 input neurons each (192 total). Total directly driven:
2,850. Same seed-based input assignments, same fixed 139,255-cell brain/synapses.

Readout includes 1,415 original motor/descending neurons and 1,024 recipient
neurons. Select recipients deterministically by the fraction of their incoming
anatomical synapse counts originating at driven radar/resource neurons. Exclude
all directly driven cells, visual-input population and existing motor outputs.
The selection is independent of gameplay, coordinates, rewards, gate location or
success. No raw sensor bypass to the learned head. All features are actual fixed
reservoir neuron activity, differences and EMA: 2,439 x 3 = 7,317 features.

## Learning and operation

Same repaired Double/dueling DQN with 20% uniform/PER mixture, batch-normalized
importance weights, reward scale 0.01 for TD only, one update per decision,
five-step returns, +5 destruction / -100 death task rewards and reward-only best
selection. Epsilon starts at 1.0 for the fresh head, reaching .1 over 50,000 new
decisions. Generic random exploration actions last three decisions. No human
demos, picked sections, gate-specific rewards or scripted routes.

One-million-transition disk replay, about 29.284 GB of active feature records at
capacity for the larger representation. Files grow with actual experience;
checkpoints share immutable chunks. Keep replay-data with model backups.

No episode limit (episodes=0). Five greedy full-level evaluations initially and
every 20 training episodes. Full permanent snapshots every 20 episodes regardless
of reward; latest every four, after evaluations and on graceful STOP.

```powershell
# Bridge 0.10 in Level_1, no competing controller:
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-radar-014 --config configs/radar_dqn.toml --env-config configs/heavy_combat_ship.toml --device cuda
# Resume: same command plus --resume. Never --warm-start.
New-Item -ItemType File -Force 'logs/ppo/flywire-radar-014/STOP' | Out-Null
```

Launch logs: logs/launch/flywire-radar-014-{out,err}.txt. Viewer observer 47653
proxies active input/readout mappings from controller 47652; restart the observer
for updated assets. Visual cards are hidden; radar/resources remain clickable.
Readout count/traces reflect 2,439 actual readout neurons, not 1,415 hard-coded.

Validation: 60 Python tests, including exact frame invariance, sensor/health
sensitivity, recipient roster uniqueness and exclusion of direct input neurons.
Three edited JS modules pass syntax checks. Record launch/runtime verification
under logs/ppo/flywire-radar-014/launch-validation.json. A new representation is
not proof of gate success; judge full-level evaluations and completion events.
