# Reward experiment: flywire-dqn-011

User requested destruction +5 and death -100 on 2026-11-26 (supersedes +0.7).
Use configs/dqn.toml and configs/heavy_combat_ship.toml. Other task rewards and
DQN settings unchanged. Fresh Q heads, optimizer and replay avoid mixing reward
scales. Fixed connectome, input mapper and 126-ray layout unchanged. No warm
start. 512 whole-level training episodes; best by mean evaluation task reward;
permanent full-state checkpoints every 20 episodes regardless of score.

DQN-010 had already lost its game bridge after episode 320; latest checkpoint
is preserved at episode 320 / 3804 gradient updates. All old files remain.
No old replay or Q values are transferred. Game relaunched via normal Steam.

Launch with bridge 0.10 in Level_1 and no existing controller:

```powershell
.\.venv\Scripts\python.exe -u -m python.viewer.serve --train-run flywire-dqn-011 --config configs/dqn.toml --env-config configs/heavy_combat_ship.toml --device cuda
```

Resume with the same arguments plus --resume. Stop marker:
logs/ppo/flywire-dqn-011/STOP. Logs: logs/launch/flywire-dqn-011-{out,err}.txt.
Do not retune rewards mid-run or load old-scale replay. Historical 010 commands
in DQN_RUN.md require its old configs/combat_ship.toml.
