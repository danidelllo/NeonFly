import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from python.tests.test_sensors import game
from python.training.plastic_brain import PlasticBrain
from python.vision.connectome import resources
from python.vision.range_sensor import sensor_vector,SENSOR_WIDTH


class InputWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model=PlasticBrain('cpu')

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_all_ray_channels_and_ship_resource_order_against_independent_formula(self):
        current=game();previous=copy.deepcopy(current)
        current['physicsSeconds']=10.25
        for i in range(126):
            kind=i%5;distance=2+i/20
            current['rays']['kind'][i]=previous['rays']['kind'][i]=kind
            current['rays']['distance'][i]=distance
            previous['rays']['distance'][i]=distance+.5
        current['rays'].update(vx=2.,vy=-4.,screenX=.3,screenY=.7)
        current.update(health=75,maxHealth=100,bullets=25,maxBullets=100,rockets=0,maxRockets=10)
        result=np.r_[sensor_vector(current,previous),resources(current)]
        expected=[]
        for i in range(126):
            kind=i%5;proximity=1-(2+i/20)/12 if kind else 0.
            expected.extend([proximity,.25 if kind else 0.,
                             *[proximity if kind==category else 0. for category in range(1,5)],1.])
        expected.extend([.25,-.5,-.4,.4,.5,-.5,-1.])
        np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
        self.assertEqual(len(result),889)

    def test_all_889_values_injected_without_drops_overlap_or_offset_errors(self):
        b=self.model
        neurons=b.input_neurons.numpy();slots=b.input_slots.numpy()
        self.assertEqual(len(neurons),2850)
        self.assertEqual(len(np.unique(neurons)),2850)
        np.testing.assert_array_equal(np.bincount(slots,minlength=889),np.r_[np.full(886,3),[64,64,64]])
        self.assertTrue(np.all((neurons>=0)&(neurons<b.n)))
        # Every channel has a distinct nonzero value. Disable recurrence only
        # in this isolated test to measure the actual transition's injection.
        raw=torch.linspace(.05,.95,889)
        zero=torch.zeros((b.n,1))
        with patch('torch.sparse.mm',return_value=zero):
            actual=b.transition(torch.zeros(b.n),raw).detach()
        expected=torch.zeros(b.n)
        expected[neurons]=(1-.4**3)*raw[slots].tanh()
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)

    def test_online_and_sequence_replay_have_identical_forward_memory(self):
        b=self.model
        state=game();state.update(health=1,maxHealth=1,bullets=1,maxBullets=1,rockets=1,maxRockets=1)
        b.clear();drives=[];online=[]
        for i in range(4):
            state=copy.deepcopy(state);state['physicsSeconds']+=.16
            state['rays']['distance'][i]=3.+i
            state['health']=1.-i*.1
            online.append(b.advance(None,state).clone());drives.append(b.last_drive.clone())
        with torch.no_grad():
            replay,_=b.forward_sequence(torch.stack(drives))
            first,carry=b.forward_sequence(torch.stack(drives[:2]))
            second,_=b.forward_sequence(torch.stack(drives[2:]),carry.detach())
        torch.testing.assert_close(replay,torch.stack(online),rtol=0,atol=0)
        torch.testing.assert_close(torch.cat((first,second)),replay,rtol=0,atol=0)
        self.assertEqual(tuple(drives[0].shape),(SENSOR_WIDTH+3,))

    def test_every_input_channel_has_a_directed_path_to_a_motor_readout(self):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import breadth_first_order
        b=self.model
        # Reverse adjacency: postsynaptic row lists presynaptic sources. A
        # virtual root reaches all motor outputs, then all their ancestors.
        indices=np.r_[b.connectome.col_indices().numpy(),b.outputs.numpy()]
        ptr=np.r_[b.connectome.crow_indices().numpy(),len(indices)]
        graph=csr_matrix((np.ones(len(indices),np.int8),indices,ptr),shape=(b.n+1,b.n+1))
        visited=breadth_first_order(graph,b.n,directed=True,return_predecessors=False)
        reachable=np.zeros(b.n+1,bool);reachable[visited]=True
        slots=b.input_slots.numpy();neurons=b.input_neurons.numpy()
        effective=np.bincount(slots[reachable[neurons]],minlength=889)
        self.assertTrue(np.all(effective>0),f'Unreachable channels: {np.flatnonzero(effective==0)}')


if __name__=='__main__':unittest.main()
