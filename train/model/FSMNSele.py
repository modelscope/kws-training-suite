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
use fsmn and max pooling for channel selection
dimins:                 input feature dimension
dimlinear:              linear dimension
dimproj:                fsmn output dimension
lorder:                 left ofder
rorder:                 right order
'''
class FSMNSele(nn.Module):
    
    def __init__(self, dimins = 120, dimlinear = 128, dimproj = 64, lorder = 20, rorder = 1):
        super(FSMNSele, self).__init__()
        
        self.featmap = AffineTransform(dimins, dimlinear)
        self.shrink = LinearTransform(dimlinear, dimproj)
        self.fsmn = Fsmn(dimproj, dimproj, lorder, rorder, 1, 1)
        self.expand = AffineTransform(dimproj, dimlinear)
    
    
    def forward(self, input):
        # batch, time, channel, lineardim
        fsmnout = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.expand.linear.out_features).cuda()
        
        for n in range(input.shape[2]):
            out1 = F.relu(self.featmap(input[:, :, n, :]))
            out2 = self.shrink(out1)
            out3 = self.fsmn(out2)
            fsmnout[:, :, n, :] = F.relu(self.expand(out3))
        
        pool = nn.MaxPool2d((input.shape[2], 1), stride = (input.shape[2], 1))
        poolout = pool(fsmnout)
        poolout = torch.squeeze(poolout, -2)
        
        return poolout
    
    
    def printModel(self):
        self.featmap.printModel()
        self.shrink.printModel()
        self.fsmn.printModel()
        self.expand.printModel()


# dimins = 3
# dimouts = 2
# lorder = 2
# rorder = 1
# 
# model = FSMNSele(dimins, dimouts, lorder, rorder)
# input = torch.randn(1, 5, 2, dimins)
# 
# fsmnout = torch.zeros(input.shape[0], input.shape[1], input.shape[2], model.dimouts)
# for n in range(input.shape[2]):
#     out1 = model.proj(input[:, :, n, :])
#     fsmnout[:, :, n, :] = model.fsmn(out1)
# 
# pool = nn.MaxPool2d((input.shape[2], 1), stride = (input.shape[2], 1))
# poolout = pool(fsmnout)
# poolout = torch.squeeze(poolout, -2)


'''
FSMN model with channel selection.
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
'''
class FSMNSeleNet(nn.Module):
    
    def __init__(self, input_dim = 120, linear_dim = 128, proj_dim = 64, 
                 lorder = 20, rorder = 1, num_syn = 5, fsmn_layers = 5):
        super(FSMNSeleNet, self).__init__()
        
        self.sele = FSMNSele(input_dim, linear_dim, proj_dim, lorder, rorder)
        self.mem = _build_repeats(linear_dim, proj_dim, lorder, rorder, fsmn_layers)
        self.decision = AffineTransform(linear_dim, num_syn)
    
    
    def forward(self, input):
        out1 = self.sele(input)
        out2 = self.mem(out1)
        out3 = self.decision(out2)
        return out3
    
    
    def printModel(self):
        self.sele.printModel()
        
        for l in self.mem:
            l[0].printModel()
            l[1].printModel()
            l[2].printModel()
        
        self.decision.printModel()
    
    
    def printHeader(self):
        '''
        get FSMN params
        '''
        input_dim = self.sele.featmap.linear.in_features
        linear_dim = self.sele.featmap.linear.out_features
        proj_dim = self.sele.shrink.linear.out_features
        lorder = self.sele.fsmn.conv_left.kernel_size[0]
        rorder = 0
        if self.sele.fsmn.conv_right is not None:
            rorder = self.sele.fsmn.conv_right.kernel_size[0]
        
        num_syn = self.decision.linear.out_features
        fsmn_layers = len(self.mem) + 1
        
        # no. of output channels, 0.0 means the same as numins
        # numouts = 0.0
        numouts = 1.0
        
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # numins
        header[0] = 0.0
        # numouts
        header[1] = numouts
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
        if numouts == 1.0:
            header[HEADER_BLOCK_SIZE * hidx + 7] = 0.0
        else:
            header[HEADER_BLOCK_SIZE * hidx + 7] = -1.0
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = numouts
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = num_syn
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SOFTMAX.value)
        
        for h in header:
            print(f32ToI32(h))


# torch.set_printoptions(precision = 6, sci_mode = False)
# input_dim = 1
# linear_dim = 2
# proj_dim = 1
# lorder = 2
# rorder = 2
# num_syn = 5
# fsmn_layers = 2
# # model = FSMNSeleNet(input_dim, linear_dim, proj_dim, lorder, rorder, num_syn, fsmn_layers)
# # torch.save(model, '111.pth')
# model = torch.load('111.pth')
# model.printHeader()
# model.printModel()
# 
# input = torch.randn(1, 50, 1, input_dim)
# for i in range(input.shape[1]):
#     input[0, i, 0, :] = i
# # print(input)
# 
# output = model(input)
# # print(output)
