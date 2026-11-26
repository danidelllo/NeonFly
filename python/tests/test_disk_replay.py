import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from python.training.disk_replay import DiskReplay
from python.training.reservoir_dqn import Replay


class DiskReplayTests(unittest.TestCase):
    def test_immutable_checkpoint_restore_after_wrap_and_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = DiskReplay(4,2,3,tmp,chunk_size=2)
            for i in range(4):
                replay.add(np.full(2,i),i,float(i),np.full(2,i+1),.9)
            saved = replay.state_dict()
            self.assertNotIn('states',saved)
            for i in range(4,10):
                replay.add(np.full(2,i),i,float(i),np.full(2,i+1),.9)
            replay.close()
            restored = DiskReplay(4,2,9,tmp,chunk_size=2)
            restored.load_state_dict(saved)
            np.testing.assert_array_equal(restored.batch(np.arange(4))['states'][:,0],np.arange(4))
            restored.add(np.full(2,100),1,1,np.full(2,101),.9)
            latest = restored.state_dict()
            np.testing.assert_array_equal(restored.batch(np.arange(4))['states'][:,0],[100,1,2,3])
            restored.close()
            earlier = DiskReplay(4,2,0,tmp,chunk_size=2)
            earlier.load_state_dict(saved)
            np.testing.assert_array_equal(earlier.batch(np.arange(4))['states'][:,0],np.arange(4))
            earlier.close()
            branch = DiskReplay(4,2,0,tmp,chunk_size=2)
            branch.load_state_dict(latest)
            self.assertEqual(branch.total,5)
            np.testing.assert_array_equal(branch.batch(np.arange(4))['states'][:,0],[100,1,2,3])
            branch.close()

    def test_priority_weights_and_rng_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = DiskReplay(8,2,3,tmp)
            for i in range(8):
                replay.add(np.zeros(2),0,0,np.ones(2),0)
            replay.reprioritize(np.array([0,0,1]),np.array([3.,8.,2.]))
            saved = replay.state_dict()
            other = DiskReplay(8,2,0,tmp)
            other.load_state_dict(saved)
            a,w = replay.sample(5000,.6,.4)
            b,v = other.sample(5000,.6,.4)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(w,v)
            self.assertGreater(np.sum(a==0),np.sum(a==1))
            np.testing.assert_allclose(w,(replay.priorities[a]/replay.priorities.min())**(-.6*.4),rtol=1e-6)
            replay.close();other.close()

    def test_ram_migration_and_truncated_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ram = Replay(8,2,3)
            for i in range(5):
                ram.add(np.full(2,i),i,float(i),np.full(2,i+1),.9)
            disk = DiskReplay(100,2,0,tmp)
            disk.load_state_dict(ram.state_dict())
            for k,v in ram.batch(np.arange(5)).items():
                np.testing.assert_array_equal(v,disk.batch(np.arange(5))[k])
            saved = disk.state_dict();disk.close()
            path = Path(tmp)/saved['chunks'][0]['name']
            with path.open('r+b') as f:
                f.truncate(1)
            broken = DiskReplay(100,2,0,tmp)
            with self.assertRaisesRegex(ValueError,'truncated'):
                broken.load_state_dict(saved)
            broken.close()

    def test_empty_checkpoint_and_lazy_disk_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            disk = DiskReplay(1000000,4245,0,tmp)
            self.assertEqual(list(Path(tmp).iterdir()),[])
            disk.load_state_dict(disk.state_dict())
            disk.add(np.zeros(4245),1,5,np.ones(4245),.9)
            saved = disk.state_dict()
            self.assertEqual(sum(p.stat().st_size for p in Path(tmp).iterdir()),16996)
            self.assertEqual(saved['priorities'].numel(),1)
            # Full model width is odd; packed int64 actions must be copied to
            # contiguous arrays before conversion to Torch, including duplicates.
            for values in disk.batch(np.array([0,0])).values():
                self.assertTrue(values.flags.c_contiguous)
                torch.as_tensor(values)
            disk.close()


if __name__ == '__main__':
    unittest.main()
