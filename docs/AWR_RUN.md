# Measured-return replay: flywire-awr-015

User reported stagnation in radar-014. It stopped cleanly at episode 760 /
21,821 decisions / 20,799 gradient updates. Best evaluation mean reward -88.7574
at episode 100; late evaluation means around -100; no completion in logged
evaluations. More significantly, scaled absolute Q estimates grew from 0.42 at
episode 100 to 153 at 760, with absolute TD error ~59. That is value instability,
not merely insufficient time or exploration. Preserve the failed run and data.

## Replacement

Advantage-weighted regression (AWR-style), using complete measured Monte Carlo
returns, with uniform replay of the agent's own full-level experience. Reference:
https://arxiv.org/abs/1910.00177 . This is the Monte Carlo variant, not a claim
of implementing every setting in that paper. No critic prediction enters any
training return target. This removes DQN's bootstrap feedback mechanism; it
does not guarantee discovery of the gate solution or prevent every possible
optimization/generalization problem.

Same radar/resources-only fixed connectome and 2,439-neuron / 7,317-feature
readout. No trained recurrent neurons or raw-sensor bypass. Same task rewards:
destruction +5, death -100, all others unchanged. Raw rewards and reward-based
best selection unchanged. Internal return scale 0.01.

Policy/value head 256 -> 128 with 36 categorical action logits plus one value.
Uniform disk replay of states/actions/full observed discounted returns, capacity
one million. Legacy disk record next-state/discount fields are zero and unused.
Learn with replayed actions weighted by exponential standardized return advantage;
stabilize weights with log bounds and mean-one normalization. Critic is regressed
to measured returns with Huber loss. No target-network value is used; a target
field remains inert in the shared checkpoint format. Entropy coefficient .01,
temperature 1, learning rate 3e-4, gradient limit 1, one update per four decisions.

During training, sample from the categorical actor, with generic random-action
bursts starting at epsilon .1 -> .05 over 50k decisions and lasting three actions.
Greedy full-level evaluations contain neither sampling nor exploratory bursts.

## Recovery and startup

All 21,821 stored transitions from ALL 760 completed episodes were retained.
Recover full observed returns exactly up to float32 precision from the old
five-step reward blocks: G(t)=R5(t)+discount5(t)*G(t+5), with terminal tails zero.
Episode log lengths must match replay order; truncated/unobserved tails are
rejected by this one-off conversion. Unit tests compare reconstructed returns
against sums of original reward sequences for multiple episode lengths.

Fresh actor/value/optimizer, no unstable DQN weights. Before gameplay, 512 replay
updates on those own-agent observations fit value magnitude around .97 to actual
scaled return magnitude around .97. These are mechanics/fit checks, not evidence
of passing the gate. New counters start at zero, gradient update count at 512.
No human demonstrations, cropped starts, selected encounters or scripted routes.

No episode limit. Five greedy evaluations initially and every 20 episodes.
Permanent full checkpoints every 20 regardless of score; latest every four,
after evaluations and on STOP. Keep replay-data with checkpoint backups.

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-awr-015 --config configs/radar_awr.toml --env-config configs/heavy_combat_ship.toml --resume --device cuda
New-Item -ItemType File -Force 'logs/ppo/flywire-awr-015/STOP' | Out-Null
```

Start only with healthy bridge in Level_1 and no competing controller. Remove
STOP only when intentionally resuming. Logs: logs/launch/flywire-awr-015-{out,err}.
Observer 47653 displays AWR and measured-return description after refresh.
Source/config compatibility checks stay strict. One-off preparation command
already executed: python -m python.operations.start_radar_awr . parent.json
records source checkpoint hashes and all episode/transition counts.

Validation: 64 Python tests including complete death-credit propagation, return
reconstruction, equal/negative advantage weights, disk optimizer resume and
immutable targets under arbitrary critic predictions. Full-size CUDA replay
pretraining completed 512 updates. Live gameplay evidence is separate; record
launch checks in launch-validation.json. Report completion events honestly.
