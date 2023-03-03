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
from ModelDef import printDense
from ModelDef import printGRU
from ModelDef import f32ToI32


'''
dimins:                 input dimension
dimproj:                projection dimension
'''
class RNNVAD(nn.Module):

    def __init__(self, dimins, dimproj):
        super().__init__()
        
        self.encoder = nn.Linear(dimins, dimproj)
        self.mem1 = nn.GRU(input_size = dimproj, hidden_size = dimproj, batch_first = True)
        self.decision = nn.Linear(dimproj, 2)
    
    
    def forward(self, input):
        out0 = torch.tanh(self.encoder(input))
        out1, _ = self.mem1(out0)
        out2 = self.decision(out1)
        return out2
    
    
    def printModel(self):
        printDense(self.encoder)
        printGRU(self.mem1)
        printDense(self.decision)
    
    
    def printHeader(self):
        dimins = self.encoder.in_features
        dimproj = self.encoder.out_features
        
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 0.0
        # dimins
        header[2] = dimins
        # dimouts
        header[3] = 2.0
        # numlayers
        header[4] = 3
        
        #
        # write each layer's header
        #
        hidx = 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = dimins
        header[HEADER_BLOCK_SIZE * hidx + 3] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_TANH.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_GRU.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 3] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 4] = float(ActivationType.ACTIVATION_TANH.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 3] = 2.0
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SOFTMAX.value)
        
        for h in header:
            print(f32ToI32(h))


'''
dimins:                 input dimension
dimproj:                projection dimension
'''
class RNNVAD2(nn.Module):

    def __init__(self, dimins, dimproj):
        super().__init__()
        
        self.encoder = nn.Linear(dimins, dimproj)
        self.mem1 = nn.GRU(input_size = dimproj, hidden_size = dimproj, batch_first = True)
        self.mem2 = nn.GRU(input_size = dimproj, hidden_size = dimproj, batch_first = True)
        self.decision = nn.Linear(dimproj, 2)
    
    
    def forward(self, input):
        out0 = torch.tanh(self.encoder(input))
        out1, _ = self.mem1(out0)
        out2, _ = self.mem2(out1)
        out3 = self.decision(out2)
        return out3
    
    
    def printModel(self):
        printDense(self.encoder)
        printGRU(self.mem1)
        printGRU(self.mem2)
        printDense(self.decision)
    
    
    def printHeader(self):
        dimins = self.encoder.in_features
        dimproj = self.encoder.out_features
        
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 5
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 0.0
        # dimins
        header[2] = dimins
        # dimouts
        header[3] = 2.0
        # numlayers
        header[4] = 4
        
        #
        # write each layer's header
        #
        hidx = 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = dimins
        header[HEADER_BLOCK_SIZE * hidx + 3] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_TANH.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_GRU.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 3] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 4] = float(ActivationType.ACTIVATION_TANH.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_GRU.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 3] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 4] = float(ActivationType.ACTIVATION_TANH.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = dimproj
        header[HEADER_BLOCK_SIZE * hidx + 3] = 2.0
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SOFTMAX.value)
        
        for h in header:
            print(f32ToI32(h))
