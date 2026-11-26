# Clean experiment: flywire-clean-009

Fresh start requested by the user after dense-008. No warm start or checkpoint
loading. The mapper has no learned state: rebuild the fixed visual/resource
mapping (seed 783) and range/body mapping (seed 784) unchanged. Keep the fixed
connectome and 126-ray layout. Reservoir activity and temporal state reset.

Policy/value head, policy optimizer, RND target/predictor, novelty optimizer,
replay, calibration, coverage visits, counters and best ranking start fresh.
Use configs/clean_ppo.toml and configs/combat_ship.toml. Both transfer flags are
false. Budget: 512 training episodes / 128 updates with existing evaluations.
Keep the uncapped rewards and permanent checkpoints every 20 episodes.
No demonstrations, selected starts or gate-specific scripts.

With bridge 0.10 in Level_1 and no existing controller on 47652:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-clean-009 --config configs/clean_ppo.toml --env-config configs/combat_ship.toml --device cuda
```

Resume using that same command plus --resume. Never add --warm-start.
Launch logs: logs/launch/flywire-clean-009-{out,err}.txt.
Viewer: http://127.0.0.1:47653/ . Initial five baseline evaluations do not count
as training episodes. All earlier checkpoints are preserved.
