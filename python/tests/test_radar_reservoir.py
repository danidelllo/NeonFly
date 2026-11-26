import copy
import unittest

import numpy as np
import torch

from python.tests.test_sensors import game
from python.viewer.live import Reservoir
from python.vision.radar_readout import readout_neurons
from python.vision.range_sensor import RADAR_ENCODER, sensor_mapping


class RadarReservoirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = Reservoir('cpu',RADAR_ENCODER)

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def state(self):
        state = game()
        state.update(health=1.,maxHealth=1.,bullets=1.,maxBullets=1.,rockets=1.,maxRockets=1.)
        return state

    def test_frames_have_exactly_no_effect(self):
        m=self.model
        state=self.state()
        m.clear();a=m.advance(np.zeros((12,84,84),np.uint8),state).clone()
        activity=m.state.clone()
        m.clear();b=m.advance(np.full((12,84,84),255,np.uint8),state)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(activity,m.state,rtol=0,atol=0)
        self.assertEqual(len(m.visual),0)
        self.assertEqual(int(torch.count_nonzero(m.visual_maps)),0)
        self.assertFalse(m.weights.requires_grad)

    def test_radar_and_resources_reach_expanded_readout(self):
        m=self.model
        state=self.state()
        m.clear();baseline=m.advance(None,state).clone()
        near=copy.deepcopy(state)
        near['rays']['distance'][:55]=[1.]*55
        m.clear();changed=m.advance(None,near)
        self.assertGreater(float((changed-baseline).norm()),.001)
        low=copy.deepcopy(state);low['health']=.1
        m.clear();changed=m.advance(None,low)
        self.assertGreater(float((changed-baseline).norm()),.001)
        self.assertEqual(m.feature_width,7317)
        self.assertEqual(len(m.outputs),2439)

    def test_recipients_are_not_injected_inputs(self):
        outputs=readout_neurons(RADAR_ENCODER)
        radar,_=sensor_mapping(RADAR_ENCODER)
        resources=self.model.resource_neurons.numpy()
        self.assertEqual(len(np.unique(outputs)),2439)
        self.assertEqual(len(np.intersect1d(outputs,np.r_[radar,resources])),0)


if __name__=='__main__':
    unittest.main()
