"""Read-only actor snapshots in action-major order; the critic is never a control."""
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from python.viewer.prepare import DATA


def demo_weights(width):
    return np.random.default_rng(783).standard_normal((36, width), dtype=np.float32)


@lru_cache(maxsize=2)
def actor_packet(run, mode, update, width):
    if mode == "demo":
        weights, bias = demo_weights(width), np.zeros(36, dtype=np.float32)
    else:
        import torch
        if not isinstance(run, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run):
            raise ValueError("Invalid policy run")
        if update is None:
            raise ValueError("This producer does not identify its replay checkpoint")
        directory = Path("checkpoints") / run
        names = ["best"] if mode == "replay" else (["initial", "best"] if update == 0 else ["latest", f"update-{update:04}"])
        for name in names:
            path = directory / (name + ".pt")
            if not path.exists():
                continue
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["signature"]["config"].get("hidden"):
                raise ValueError("Nonlinear actor scores are included in each live frame")
            if saved["progress"][2] != update:
                continue
            if saved["signature"]["graph"] != hashlib.sha256((DATA/"manifest.json").read_bytes()).hexdigest():
                raise ValueError("Checkpoint connectome mismatch")
            weights, bias = (saved["head"][k][:36].numpy() for k in ("weight", "bias"))
            break
        else:
            raise ValueError("No saved actor matching this frame's update")
    if weights.shape != (36, width) or bias.shape != (36,):
        raise ValueError("Unexpected actor dimensions")
    return np.column_stack((weights, bias)).astype("<f4", copy=False).tobytes()
