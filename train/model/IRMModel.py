'''
model for ideal ratio mask
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import math
from enum import Enum
from ModelDef import LayerType
from ModelDef import ActivationType
from ModelDef import HEADER_BLOCK_SIZE
from ModelDef import printNeonMatrix
from ModelDef import printNeonVector
from ModelDef import f32ToI32
from FSMN import LinearTransform
from FSMN import AffineTransform
from FSMN import Fsmn
from FSMN import _build_repeats


'''
FSMN model for ideal ratio mask estimation.
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
'''
class FSMNIRMNet(nn.Module):
    
    def __init__(self, input_dim = 40, linear_dim = 128, proj_dim = 64, 
                 lorder = 10, rorder = 1, num_syn = 40, fsmn_layers = 5):
        super(FSMNIRMNet, self).__init__()
        
        self.featmap = AffineTransform(input_dim, linear_dim)
        self.mem = _build_repeats(linear_dim, proj_dim, lorder, rorder, fsmn_layers)
        self.decision = AffineTransform(linear_dim, num_syn)
    
    
    def forward(self, input):
        out1 = F.relu(self.featmap(input))
        out2 = self.mem(out1)
        out3 = torch.sigmoid(self.decision(out2))
        return out3
    
    
    def printModel(self):
        self.featmap.printModel()
        
        for l in self.mem:
            l[0].printModel()
            l[1].printModel()
            l[2].printModel()
        
        self.decision.printModel()
    
    
    def printHeader(self):
        '''
        get FSMN params
        '''
        input_dim = self.featmap.linear.in_features
        linear_dim = self.featmap.linear.out_features
        proj_dim = self.mem[0][0].linear.out_features
        lorder = self.mem[0][1].conv_left.kernel_size[0]
        rorder = 0
        if self.mem[0][1].conv_right is not None:
            rorder = self.mem[0][1].conv_right.kernel_size[0]
        
        num_syn = self.decision.linear.out_features
        fsmn_layers = len(self.mem)
        
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 0.0
        # dimins
        header[2] = input_dim
        # dimouts
        header[3] = num_syn
        # numlayers
        header[4] = 3
        
        #
        # write each layer's header
        #
        hidx = 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = input_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_RELU.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_SEQUENTIAL_FSMN.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = lorder
        header[HEADER_BLOCK_SIZE * hidx + 5] = rorder
        header[HEADER_BLOCK_SIZE * hidx + 6] = fsmn_layers
        header[HEADER_BLOCK_SIZE * hidx + 7] = -1.0
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = num_syn
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SIGMOID.value)
        
        for h in header:
            print(f32ToI32(h))
