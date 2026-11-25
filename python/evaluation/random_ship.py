"""Run the real ship environment with no learned policy; output one JSON record per episode."""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from python.environment.ship import ShipConfig, ShipEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--config", type=Path, default=Path("configs/ship.toml"))
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    cfg = ShipConfig.load(args.config)
    if args.max_steps is not None:
        cfg = replace(cfg, max_steps=args.max_steps)
    env = ShipEnv(cfg)
    out = Path("logs/random") / str(time.time_ns())
    out.mkdir(parents=True)
    print(out)
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as log:
        for i in range(args.episodes):
            seed = args.seed + i
            env.action_space.seed(seed)
            obs, start = env.reset(seed=seed)
            assert env.observation_space.contains(obs)
            total, steps, ticks, elapsed = 0.0, 0, 0, 0
            begin = time.monotonic()
            while True:
                obs, reward, done, cutoff, info = env.step(env.action_space.sample())
                assert env.observation_space.contains(obs)
                total += reward
                steps += 1
                ticks += info["applied_ticks"]
                elapsed += info["elapsed_ticks"]
                if done or cutoff:
                    break
            row = dict(seed=seed, steps=steps, action_ticks=ticks, elapsed_ticks=elapsed, reward=round(total, 6),
                       reason=info["reason"], terminated=done, truncated=cutoff,
                       start_xy=[start["game"]["x"], start["game"]["y"]],
                       wall_seconds=round(time.monotonic() - begin, 3))
            line = json.dumps(row)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    main()
