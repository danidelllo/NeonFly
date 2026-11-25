"""Versioned, fixed full-screen color/motion projection. No fitted parameters."""
import numpy as np
import torch
from torch.nn import functional as F

ENCODER = "color-motion-v1"


def visual_channels(observation):
    """RGB frame stack -> six 42x42 channels; no state beyond the observation."""
    if observation.shape != (12, 84, 84):
        raise ValueError("color-motion-v1 expects four RGB 84x84 uint8 frames")
    frames = observation.reshape(4, 3, 84, 84).float() / 255
    frames = F.avg_pool2d(frames, 2)
    luminance = (frames * frames.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    r, g, b = frames[-1]
    light = luminance[-1]
    edge = (light - F.avg_pool2d(light[None, None], 3, 1, 1)[0, 0]).abs()
    return torch.stack((2*light-1, r-g, b-(r+g)/2,
                        (light-luminance[-2])*3, (light-luminance[0])*2, edge*4)).clamp(-1, 1)


def sensory_mapping(sensory):
    """Use every visual neuron; cover all pixels/channels, then duplicate evenly.

    Assignment is artificial, seeded and independent of game content. Anatomy
    does not supply validated receptive fields, so coordinates are not treated
    as a retinal map. Last 192 cells carry three explicit resource signals.
    """
    visual = np.random.default_rng(783).permutation(sensory).astype(np.int64)
    if len(visual) < 6*42*42+192:
        raise ValueError("insufficient visual input population")
    return visual[:-192], np.arange(len(visual)-192, dtype=np.int64) % (6*42*42), visual[-192:]


def resources(game):
    values = []
    for current, maximum in (("health", "maxHealth"), ("bullets", "maxBullets"), ("rockets", "maxRockets")):
        value, capacity = game[current], game[maximum]
        if not np.isfinite(value) or not np.isfinite(capacity) or capacity <= 0 or value < 0:
            raise ValueError(f"invalid resource telemetry: {current}")
        values.append(2*min(1., value/capacity)-1)
    return values


def preview(observation):
    if observation.shape[0] == 12:
        rgb = observation[-3:].astype(np.float32)
        return np.rint(.299*rgb[0]+.587*rgb[1]+.114*rgb[2]).astype(np.uint8)
    return observation[-1]
