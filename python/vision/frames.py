import io
from collections import deque

import numpy as np
from PIL import Image


class VisualStack:
    """Oldest frame first, channels first. Reset never carries episode history."""

    def __init__(self, size, depth, rgb=False, crop=None, normalize=False):
        self.size, self.depth = size, depth
        self.mode = "RGB" if rgb else "L"
        self.crop, self.normalize = crop, normalize
        self.frames = deque(maxlen=depth)

    def push(self, png, *, reset=False):
        with Image.open(io.BytesIO(png)) as source:
            if self.crop:
                l, t, r, b = self.crop
                if not (0 <= l < r <= source.width and 0 <= t < b <= source.height):
                    raise ValueError("crop must be inside the rendered frame")
                source = source.crop(self.crop)
            view = source.convert(self.mode).resize((self.size, self.size), Image.Resampling.BOX)
            array = np.asarray(view)
        frame = array[None] if array.ndim == 2 else array.transpose(2, 0, 1)
        if reset:
            self.frames.clear()
            self.frames.extend([frame] * self.depth)
        else:
            self.frames.append(frame)
        stacked = np.concatenate(self.frames)
        return stacked.astype(np.float32) / 255 if self.normalize else stacked
