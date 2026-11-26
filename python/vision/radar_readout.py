"""Deterministic anatomical recipient readout; no fitted or game-specific selection."""
from functools import lru_cache
import numpy as np

from python.viewer.prepare import DATA
from python.vision.connectome import sensory_mapping
from python.vision.range_sensor import RADAR_ENCODER, sensor_mapping


@lru_cache(maxsize=2)
def readout_neurons(encoder):
    load = lambda name:np.load(DATA/(name+'.npy'))
    outputs = load('output').astype(np.int64)
    if encoder != RADAR_ENCODER:
        return outputs
    radar, _ = sensor_mapping(RADAR_ENCODER)
    _, _, resources = sensory_mapping(load('sensory'))
    driven = np.r_[radar,resources]
    ptr, indices, counts = (load('out_'+name) for name in ('indptr','indices','data'))
    score = np.zeros(len(ptr)-1,np.float64)
    for neuron in driven:
        sl = slice(ptr[neuron],ptr[neuron+1])
        np.add.at(score,indices[sl],counts[sl])
    inptr, incoming = load('in_indptr'),load('in_data')
    cumulative = np.r_[0,np.cumsum(incoming,dtype=np.float64)]
    score /= np.maximum(1,np.diff(cumulative[inptr]))
    # Never directly read an injected input; retain the reservoir transformation.
    score[np.unique(np.r_[driven,load('sensory'),outputs])] = 0
    selected = np.argsort(-score,kind='stable')[:1024]
    if np.any(score[selected] <= 0):
        raise ValueError('insufficient anatomically connected sensor recipients')
    return np.r_[outputs,selected].astype(np.int64)
