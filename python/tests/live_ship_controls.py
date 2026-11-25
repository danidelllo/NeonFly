"""Exercise all 36 Gym actions in the real first level, resetting before each case."""
import itertools
import json
import time
from pathlib import Path

from python.environment.ship import ShipEnv


def main():
    env = ShipEnv()
    output = Path("logs/controls") / str(time.time_ns())
    output.mkdir(parents=True)
    print(output, flush=True)
    with (output / "actions.jsonl").open("x", encoding="utf-8") as log:
        for rocket, fire, dy, dx in itertools.product((False, True), (False, True), (0, -1, 1), (0, -1, 1)):
            action = (0, -1, 1).index(dx) + 3 * (0, -1, 1).index(dy) + 9 * fire + 18 * rocket
            _, info = env.reset()
            start = info["game"]
            _, _, term, trunc, info = env.step(action)
            end = info["game"]
            x, y = end["x"] - start["x"], end["y"] - start["y"]
            bullets, rockets = start["bullets"] - end["bullets"], start["rockets"] - end["rockets"]
            row = dict(action=action, dx=x, dy=y, bullets_used=bullets, rockets_used=rockets,
                       ticks=info["applied_ticks"], frame=end["frame"])
            log.write(json.dumps(row) + "\n")
            log.flush()
            assert not term and not trunc and info["applied_ticks"] == env.config.action_ticks, row
            for expected, actual in ((dx, x), (dy, y)):
                assert (actual * expected > 0.001 if expected else abs(actual) < 0.001), row
            assert (bullets > 0) == fire and (rockets > 0) == rocket, row
            if action == 35:
                for _ in range(5):
                    _, _, term, trunc, released = env.step(0)
                    assert not term and not trunc
                    assert (released["game"]["bullets"], released["game"]["rockets"]) == (
                        end["bullets"], end["rockets"])
    print("All 36 combinations passed; neutral released both weapons.")


if __name__ == "__main__":
    main()
