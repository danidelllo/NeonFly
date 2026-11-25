"""Explicit live regression: run `python -m python.tests.live_ship` in Level_1."""
import io
import json
import time
from pathlib import Path

from PIL import Image

from python.communication.observe import request


def main():
    output = Path("logs/smoke") / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)

    def call(label, op, *args):
        state, png = request(op, *args)
        with (output / "trace.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"label": label, **state}) + "\n")
        if png:
            with Image.open(io.BytesIO(png)) as im:
                im.load()
                assert im.size == (state["width"], state["height"])
            (output / f"{label}.png").write_bytes(png)
        return state

    if call("initial", 1)["scene"] != "Level_1":
        raise RuntimeError("Start the first ship level before running the live regression")
    first = call("reset-one", 6)
    start = call("reset-two", 6)
    assert first["sceneHandle"] != start["sceneHandle"]
    assert first["controller"] == start["controller"] == "Player"
    assert all(abs(first[k] - start[k]) < 0.01 for k in ("x", "y"))
    right = call("right", 5, 1, 12)
    left = call("left", 5, -1, 12)
    assert start["sceneHandle"] == right["sceneHandle"] == left["sceneHandle"]
    assert right["appliedTicks"] == left["appliedTicks"] == 12
    assert right["x"] > start["x"] and left["x"] < right["x"]
    deadline = time.monotonic() + 95
    while time.monotonic() < deadline:
        status, _ = request(1)
        if status["gameOver"] and status["deaths"] > start["deaths"]:
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("No natural game-over observed within 95 seconds")
    terminal = call("death", 2)
    restarted = call("restart", 6)
    assert terminal["gameOver"] and terminal["sceneHandle"] != restarted["sceneHandle"]
    assert restarted["scene"] == "Level_1" and restarted["controller"] == "Player"
    assert not restarted["gameOver"]
    print(f"Live ship regression passed. Evidence: {output}")


if __name__ == "__main__":
    main()
