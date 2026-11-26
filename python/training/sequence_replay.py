"""Immutable disk episodes of raw sensors, actions and measured returns."""
import hashlib
import os
import uuid
from pathlib import Path

import numpy as np


class SequenceReplay:
    def __init__(self,directory,capacity,seed):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.capacity,self.size = capacity,0
        self.episodes = []
        self.rng = np.random.default_rng(seed)

    def add(self,drives,actions,returns):
        if drives.shape != (len(actions),889) or len(returns) != len(actions) or not len(actions):
            raise ValueError('invalid raw sensor episode')
        if not np.isfinite(drives).all() or not np.isfinite(returns).all():
            raise ValueError('non-finite sequence replay')
        name = uuid.uuid4().hex+'.npz'
        path = self.directory/name
        with path.open('xb') as f:
            np.savez(f,drives=drives.astype(np.float32),actions=np.asarray(actions,np.int64),returns=np.asarray(returns,np.float32))
            f.flush();os.fsync(f.fileno())
        item = dict(name=name,length=len(actions),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.episodes.append(item);self.size += len(actions)
        while self.size > self.capacity and len(self.episodes)>1:
            self.size -= self.episodes.pop(0)['length']
        return item

    def load(self,item):
        path = self.directory/item['name']
        if path.name != item['name'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('raw replay episode missing or changed')
        with np.load(path,allow_pickle=False) as data:
            return {k:data[k].copy() for k in ('drives','actions','returns')}

    def sample(self,count):
        return [self.episodes[i] for i in self.rng.choice(len(self.episodes),size=count,replace=True)]

    def state_dict(self):
        return dict(directory=str(self.directory),capacity=self.capacity,size=self.size,
                    episodes=[dict(e) for e in self.episodes],rng=self.rng.bit_generator.state)

    def load_state_dict(self,saved):
        if saved['directory'] != str(self.directory) or saved['capacity'] != self.capacity:
            raise ValueError('raw replay directory/capacity mismatch')
        if sum(e['length'] for e in saved['episodes']) != saved['size']:
            raise ValueError('raw replay length mismatch')
        self.episodes = [dict(e) for e in saved['episodes']]
        for item in self.episodes:
            if Path(item['name']).name != item['name'] or not (self.directory/item['name']).exists():
                raise ValueError('missing raw replay episode')
        self.size = saved['size'];self.rng.bit_generator.state = saved['rng']
