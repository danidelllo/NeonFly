"""FlyWire circuit with three intrinsic trainable parameters per neuron.

Anatomical sparse weights and signs never change. No learned input encoder,
hidden readout, planner, teacher or external recurrent state is introduced.
"""
import math
import numpy as np
import torch
from torch import nn

from python.viewer.live import Reservoir
from python.viewer.prepare import DATA
from python.vision.connectome import resources
from python.vision.range_sensor import RADAR_ENCODER,SENSOR_WIDTH,sensor_vector


class PlasticBrain(Reservoir,nn.Module):
    trainable_brain = True

    def __init__(self,device):
        nn.Module.__init__(self)
        Reservoir.__init__(self,device,RADAR_ENCODER)
        self.outputs = torch.from_numpy(np.load(DATA/'output.npy').astype(np.int64)).to(device)
        self.feature_width = len(self.outputs)
        # Nonpersistent buffer: checkpoint pins the graph by manifest/source hash.
        self.register_buffer('connectome',self.weights,persistent=False)
        self.raw_gain = nn.Parameter(torch.full((self.n,),math.log((1.-.25)/(2.-1.)),device=device))
        self.bias = nn.Parameter(torch.zeros(self.n,device=device))
        self.raw_alpha = nn.Parameter(torch.full((self.n,),math.log((.6-.05)/(.95-.6)),device=device))
        self.input_neurons = torch.cat((self.range_neurons,self.resource_neurons))
        self.input_slots = torch.cat((self.range_slots,
            torch.arange(SENSOR_WIDTH,SENSOR_WIDTH+3,device=device).repeat_interleave(64)))
        self.visual_maps = torch.zeros((6,42,42),device=device)
        self.clear()

    def intrinsic(self):
        return .25+1.75*self.raw_gain.sigmoid(),.5*self.bias.tanh(),.05+.9*self.raw_alpha.sigmoid()

    def transition(self,state,raw_drive):
        gain,bias,alpha = self.intrinsic()
        drive = torch.zeros(self.n,device=self.device).index_add(0,self.input_neurons,raw_drive[self.input_slots])
        for _ in range(3):
            recurrent = torch.sparse.mm(self.connectome,state[:,None])[:,0]
            state = (1-alpha)*state+alpha*torch.tanh(gain*(recurrent+drive)+bias)
        return state

    def forward_sequence(self,drives,state=None):
        if state is None:
            state = torch.zeros(self.n,device=self.device)
        outputs = []
        for raw in drives:
            state = self.transition(state,raw)
            outputs.append(state[self.outputs])
        return torch.stack(outputs),state

    @torch.no_grad()
    def advance(self,observation,game=None):
        self.range_drive = sensor_vector(game,self.previous_sensor)
        self.previous_sensor = game
        self.last_drive = torch.as_tensor(np.r_[self.range_drive,resources(game)].astype(np.float32),device=self.device)
        self.state = self.transition(self.state[:,0],self.last_drive)[:,None]
        return self.state[self.outputs,0]
