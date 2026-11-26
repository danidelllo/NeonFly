"""Append-only disk features, RAM priority trees, immutable checkpoint references.

Checkpoints keep priorities and chunk-prefix manifests, not copies of features.
Restoring an older checkpoint starts a new chunk, preserving all older branches.
Referenced files must be kept with checkpoints. No automatic file deletion.
"""
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


class DiskReplay:
    def __init__(self, capacity, width, seed, directory, alpha=.6, chunk_size=4096,
                 uniform_fraction=0., weight_normalization='global'):
        if capacity <= 0 or width <= 0 or not 0 <= alpha <= 1:
            raise ValueError('invalid disk replay dimensions/alpha')
        self.capacity, self.width, self.alpha = capacity, width, alpha
        if not 0 <= uniform_fraction <= 1 or weight_normalization not in ('global','batch'):
            raise ValueError('invalid replay weighting configuration')
        self.uniform_fraction, self.weight_normalization = uniform_fraction, weight_normalization
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.dtype = np.dtype([('states','<f2',(width,)), ('next_states','<f2',(width,)),
                               ('actions','<i8'), ('rewards','<f4'), ('discounts','<f4')])
        self.priorities = np.zeros(capacity, np.float32)
        self.leaves = 1 << (capacity-1).bit_length()
        self.sums = np.zeros(2*self.leaves, np.float64)
        self.minimum = np.full(2*self.leaves, np.inf, np.float64)
        self.size = self.cursor = self.total = 0
        self.max_priority = 1.
        self.rng = np.random.default_rng(seed)
        self.chunks, self.starts = [], []
        self.cache = OrderedDict()
        self.file = None

    def _priority(self, index, priority):
        self.priorities[index] = priority
        self.max_priority = max(self.max_priority, priority)
        node = self.leaves+index
        self.sums[node] = self.minimum[node] = float(priority)**self.alpha
        node //= 2
        while node:
            self.sums[node] = self.sums[node*2]+self.sums[node*2+1]
            self.minimum[node] = min(self.minimum[node*2], self.minimum[node*2+1])
            node //= 2

    def add(self, state, action, reward, next_state, discount):
        if self.file is None or self.chunks[-1]['count'] >= self.chunk_size:
            self._seal()
            name = uuid.uuid4().hex+'.bin'
            self.file = (self.directory/name).open('xb', buffering=0)
            self.chunks.append(dict(name=name, start=self.total, count=0))
            self.starts.append(self.total)
        record = np.zeros(1, self.dtype)
        record['states'], record['next_states'] = state, next_state
        record['actions'], record['rewards'], record['discounts'] = action, reward, discount
        if self.file.write(record.tobytes()) != self.dtype.itemsize:
            raise OSError('incomplete replay disk write')
        self.chunks[-1]['count'] += 1
        self._priority(self.cursor, self.max_priority)
        self.total += 1
        self.size = min(self.capacity, self.total)
        self.cursor = self.total % self.capacity

    def sample(self, batch, alpha, beta):
        if alpha != self.alpha or not self.size or not 0 <= beta <= 1:
            raise ValueError('invalid replay sampling parameters')
        values = self.rng.random(batch)*self.sums[1]
        nodes = np.ones(batch, np.int64)
        while nodes[0] < self.leaves:
            left = nodes*2
            right = values >= self.sums[left]
            values -= np.where(right, self.sums[left], 0.)
            nodes = left+right
        indices = nodes-self.leaves
        if self.uniform_fraction:
            uniform = self.rng.random(batch) < self.uniform_fraction
            indices[uniform] = self.rng.integers(self.size, size=int(uniform.sum()))
        probabilities = (1-self.uniform_fraction)*self.sums[self.leaves+indices]/self.sums[1]+self.uniform_fraction/self.size
        if self.weight_normalization == 'batch':
            weights = (probabilities/probabilities.min())**(-beta)
        else:
            minimum = (1-self.uniform_fraction)*self.minimum[1]/self.sums[1]+self.uniform_fraction/self.size
            weights = (probabilities/minimum)**(-beta)
        return indices, weights.astype(np.float32)

    def batch(self, indices):
        indices = np.asarray(indices, np.int64)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError('invalid replay indices')
        sequences = (self.total//self.capacity)*self.capacity+indices
        sequences = np.where(sequences >= self.total, sequences-self.capacity, sequences)
        result = np.empty(len(indices), self.dtype)
        groups = np.searchsorted(self.starts, sequences, side='right')-1
        for group in np.unique(groups):
            chunk = self.chunks[group]
            positions = np.flatnonzero(groups == group)
            offsets = sequences[positions]-chunk['start']
            if np.any(offsets >= chunk['count']):
                raise ValueError('replay manifest has missing transitions')
            # Map only immutable prefixes, so live file growth cannot expose rows
            # absent from a restored checkpoint. Always copy sampled rows.
            key = (chunk['name'], chunk['count'])
            if key not in self.cache:
                self.cache[key] = np.memmap(self.directory/chunk['name'], dtype=self.dtype,
                                             mode='r', shape=(chunk['count'],))
                if len(self.cache) > 32:
                    _, old = self.cache.popitem(last=False)
                    old._mmap.close()
            self.cache.move_to_end(key)
            result[positions] = self.cache[key][offsets]
        # Packed records have a 16,996-byte stride at width 4,245; int64 action
        # views would have a stride not divisible by eight. Torch requires each
        # field to have valid element strides, so materialize contiguous batches.
        return {k:np.ascontiguousarray(result[k]) for k in self.dtype.names}

    def reprioritize(self, indices, errors):
        if not np.isfinite(errors).all():
            raise FloatingPointError('non-finite replay priorities')
        for i in np.unique(indices):
            self._priority(int(i), float(np.max(errors[indices == i]))+1e-5)

    def _seal(self):
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            self.file = None

    def state_dict(self):
        self._seal()
        # Drop old chunks only from this manifest; older checkpoint references
        # and files stay valid. Retention/deletion requires a separate audit.
        begin = self.total-self.size
        active = [dict(c) for c in self.chunks if c['start']+c['count'] > begin]
        return dict(storage='disk-v1', capacity=self.capacity, width=self.width,
                    directory=str(self.directory), total=self.total, size=self.size,
                    cursor=self.cursor, alpha=self.alpha, max_priority=self.max_priority,
                    uniform_fraction=self.uniform_fraction, weight_normalization=self.weight_normalization,
                    chunks=active, priorities=torch.from_numpy(self.priorities[:self.size].copy()),
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, saved):
        if saved.get('storage') != 'disk-v1':
            # Audited migration from the original in-memory replay.
            if saved['size'] > self.capacity or saved['cursor'] != saved['size']:
                raise ValueError('RAM replay migration requires an unwrapped buffer')
            for i in range(saved['size']):
                self.add(*(saved[k][i].numpy() for k in ('states','actions','rewards','next_states','discounts')))
            for i,p in enumerate(saved['priorities'].numpy()):
                self._priority(i, float(p))
            self.rng.bit_generator.state = saved['rng']
            self._seal()
            return
        if (saved['capacity'],saved['width'],saved['alpha']) != (self.capacity,self.width,self.alpha):
            raise ValueError('disk replay checkpoint dimensions mismatch')
        if (saved.get('uniform_fraction',0.),saved.get('weight_normalization','global')) != (self.uniform_fraction,self.weight_normalization):
            raise ValueError('disk replay weighting configuration mismatch')
        if Path(saved['directory']).resolve() != self.directory:
            raise ValueError('disk replay directory mismatch; retain checkpoint sidecar files')
        self.close()
        self.total, self.size, self.cursor = saved['total'],saved['size'],saved['cursor']
        if self.size != min(self.capacity,self.total) or self.cursor != self.total % self.capacity:
            raise ValueError('invalid disk replay counters')
        self.chunks = [dict(c) for c in saved['chunks']]
        self.starts = [c['start'] for c in self.chunks]
        expected = self.total-self.size
        for c in self.chunks:
            if Path(c['name']).name != c['name'] or not 0 < c['count'] <= self.chunk_size:
                raise ValueError('invalid replay chunk')
            if c['start'] > expected or c['start']+c['count'] <= expected:
                raise ValueError('noncontiguous replay manifest')
            if (self.directory/c['name']).stat().st_size < c['count']*self.dtype.itemsize:
                raise ValueError('truncated disk replay chunk')
            expected = c['start']+c['count']
        if expected != self.total:
            raise ValueError('incomplete replay manifest')
        self.priorities.fill(0)
        self.sums.fill(0)
        self.minimum.fill(np.inf)
        self.priorities[:self.size] = saved['priorities'].numpy()
        if np.any(self.priorities[:self.size] <= 0) or not np.isfinite(self.priorities[:self.size]).all():
            raise ValueError('invalid replay priorities')
        power = self.priorities[:self.size].astype(np.float64)**self.alpha
        self.sums[self.leaves:self.leaves+self.size] = power
        self.minimum[self.leaves:self.leaves+self.size] = power
        # Vectorized level rebuild: O(capacity) once on resume, O(log N) updates.
        level = self.leaves//2
        while level:
            parents = np.arange(level,level*2)
            self.sums[parents] = self.sums[parents*2]+self.sums[parents*2+1]
            self.minimum[parents] = np.minimum(self.minimum[parents*2],self.minimum[parents*2+1])
            level //= 2
        self.max_priority = saved['max_priority']
        self.rng.bit_generator.state = saved['rng']

    def close(self):
        self._seal()
        for mapping in self.cache.values():
            mapping._mmap.close()
        self.cache.clear()
