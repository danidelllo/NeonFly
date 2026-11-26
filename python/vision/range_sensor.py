"""Fixed range/velocity encoding onto measured mechanosensory neurons."""
from functools import lru_cache

import numpy as np

LEGACY_RAY_ENCODER = "color-motion-rays-v1"
RAY_ENCODER = "color-motion-rays-v2"
RAY_ENCODERS = (LEGACY_RAY_ENCODER, RAY_ENCODER)
RAY_COUNT = 126
SENSOR_WIDTH = RAY_COUNT*7+4
RAY_LAYOUT = 'quadrants-126-v2'


def ray_angles():
    return np.deg2rad(np.concatenate([start+(np.arange(count)+.5)*90/count
        for start, count in ((-45,55),(45,8),(135,8),(225,55))])).astype(np.float32)


def scan_arrays(game):
    scan = game.get("rays")
    if not scan or not scan.get("valid"):
        raise ValueError("valid Unity range scan required")
    distance = np.asarray(scan["distance"], dtype=np.float32)
    count = len(distance)
    visible = np.asarray(scan["visible"], dtype=np.float32)
    start = np.asarray(scan.get("visibleStart", [0.]*count), dtype=np.float32)
    kind = np.asarray(scan["kind"], dtype=np.int64)
    radius = float(scan["range"])
    if (count not in (32, RAY_COUNT) or distance.shape != (count,) or visible.shape != (count,) or start.shape != (count,) or kind.shape != (count,) or
        not np.isfinite(radius) or radius <= 0 or not np.isfinite(distance).all() or
        not np.isfinite(visible).all() or not np.isfinite(start).all() or np.any(start < 0) or
        np.any(distance < start-1e-4) or np.any(distance > visible+1e-4) or
        np.any(visible > radius+1e-4) or np.any(visible < 0) or np.any((kind < 0) | (kind > 4))):
        raise ValueError("invalid range sensor data")
    if count == RAY_COUNT:
        angles = np.asarray(scan.get('angles'), np.float32)
        if scan.get('layout') != RAY_LAYOUT or angles.shape != (count,) or not np.allclose(angles, ray_angles(), atol=1e-6):
            raise ValueError('range sensor angular layout mismatch')
    return scan, distance, visible, kind, radius


def sensor_vector(game, previous=None):
    scan, distance, visible, kind, radius = scan_arrays(game)
    proximity = np.where(kind != 0, 1-distance/radius, 0)
    closing = np.zeros(len(distance), np.float32)
    if previous is not None and previous.get("sceneHandle") == game.get("sceneHandle"):
        _, old_distance, _, old_kind, _ = scan_arrays(previous)
        dt = game["physicsSeconds"] - previous["physicsSeconds"]
        if dt > 0 and old_distance.shape == distance.shape:
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
    count = len(visible)
    start = np.asarray(scan.get("visibleStart", [0.]*count), np.float32)
    if (distance.shape != (count,) or kind.shape != (count,) or not np.isfinite(distance).all()
        or np.any(distance < start-1e-4) or np.any(distance > visible+1e-4) or np.any((kind < 0) | (kind > 4))):
        raise ValueError("invalid stable geometry scan")
    proximity = 1-distance/radius
    geometry = np.stack((*[np.where(kind == category, proximity, 0) for category in (1, 2, 4)],
                         visible/radius), axis=1)
    return np.concatenate((geometry.flatten(), [2*scan["screenX"]-1, 2*scan["screenY"]-1])).astype(np.float32)


@lru_cache(maxsize=1)
def sensor_mapping(encoder=RAY_ENCODER):
    import pandas as pd
    from python.viewer.prepare import DATA
    ids = np.load(DATA/"ids.npy")
    classification = pd.read_csv(DATA/"raw/classification.csv.gz").set_index("root_id").loc[ids]
    neurons = np.flatnonzero((classification.super_class == "sensory") & (classification["class"] == "mechanosensory"))
    if len(neurons) != 2674:
        raise ValueError("unexpected mechanosensory neuron roster")
    neurons = np.random.default_rng(784).permutation(neurons).astype(np.int64)
    if encoder == LEGACY_RAY_ENCODER:
        slots = np.arange(len(neurons), dtype=np.int64) % (32*7+4)
    elif encoder == RAY_ENCODER:
        neurons = neurons[:SENSOR_WIDTH*3]
        slots = np.tile(np.arange(SENSOR_WIDTH, dtype=np.int64), 3)
    else:
        raise ValueError('unknown ray encoder')
    return neurons, slots
