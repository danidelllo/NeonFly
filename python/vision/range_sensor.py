"""Fixed range/velocity encoding onto measured mechanosensory neurons."""
from functools import lru_cache

import numpy as np

RAY_ENCODER = "color-motion-rays-v1"
SENSOR_WIDTH = 32*7+4


def scan_arrays(game):
    scan = game.get("rays")
    if not scan or not scan.get("valid"):
        raise ValueError("valid Unity range scan required")
    distance = np.asarray(scan["distance"], dtype=np.float32)
    visible = np.asarray(scan["visible"], dtype=np.float32)
    start = np.asarray(scan.get("visibleStart", [0.]*32), dtype=np.float32)
    kind = np.asarray(scan["kind"], dtype=np.int64)
    radius = float(scan["range"])
    if (distance.shape != (32,) or visible.shape != (32,) or start.shape != (32,) or kind.shape != (32,) or
        not np.isfinite(radius) or radius <= 0 or not np.isfinite(distance).all() or
        not np.isfinite(visible).all() or not np.isfinite(start).all() or np.any(start < 0) or
        np.any(distance < start-1e-4) or np.any(distance > visible+1e-4) or
        np.any(visible > radius+1e-4) or np.any(visible < 0) or np.any((kind < 0) | (kind > 4))):
        raise ValueError("invalid range sensor data")
    return scan, distance, visible, kind, radius


def sensor_vector(game, previous=None):
    scan, distance, visible, kind, radius = scan_arrays(game)
    proximity = np.where(kind != 0, 1-distance/radius, 0)
    closing = np.zeros(32, np.float32)
    if previous is not None and previous.get("sceneHandle") == game.get("sceneHandle"):
        _, old_distance, _, old_kind, _ = scan_arrays(previous)
        dt = game["physicsSeconds"] - previous["physicsSeconds"]
        if dt > 0:
            closing = np.where((kind != 0) & (kind == old_kind), (old_distance-distance)/(dt*8), 0)
    rays = np.stack((proximity, np.clip(closing, -1, 1),
                    *[np.where(kind == category, proximity, 0) for category in range(1, 5)],
                    visible/radius), axis=1)
    body = np.array([scan["vx"]/8, scan["vy"]/8, 2*scan["screenX"]-1, 2*scan["screenY"]-1], np.float32)
    if not np.isfinite(body).all():
        raise ValueError("invalid ship sensor state")
    return np.concatenate((rays.flatten(), np.clip(body, -1, 1))).astype(np.float32)


def geometry_vector(game):
    """Novelty excludes projectiles, their closing speeds, HUDs and clocks."""
    scan, distance, visible, kind, radius = scan_arrays(game)
    distance = np.asarray(scan.get("geometryDistance", distance), np.float32)
    kind = np.asarray(scan.get("geometryKind", kind), np.int64)
    start = np.asarray(scan.get("visibleStart", [0.]*32), np.float32)
    if (distance.shape != (32,) or kind.shape != (32,) or not np.isfinite(distance).all()
        or np.any(distance < start-1e-4) or np.any(distance > visible+1e-4) or np.any((kind < 0) | (kind > 4))):
        raise ValueError("invalid stable geometry scan")
    proximity = 1-distance/radius
    geometry = np.stack((*[np.where(kind == category, proximity, 0) for category in (1, 2, 4)],
                         visible/radius), axis=1)
    return np.concatenate((geometry.flatten(), [2*scan["screenX"]-1, 2*scan["screenY"]-1])).astype(np.float32)


@lru_cache(maxsize=1)
def sensor_mapping():
    import pandas as pd
    from python.viewer.prepare import DATA
    ids = np.load(DATA/"ids.npy")
    classification = pd.read_csv(DATA/"raw/classification.csv.gz").set_index("root_id").loc[ids]
    neurons = np.flatnonzero((classification.super_class == "sensory") & (classification["class"] == "mechanosensory"))
    if len(neurons) != 2674:
        raise ValueError("unexpected mechanosensory neuron roster")
    neurons = np.random.default_rng(784).permutation(neurons).astype(np.int64)
    slots = np.arange(len(neurons), dtype=np.int64) % SENSOR_WIDTH
    return neurons, slots
