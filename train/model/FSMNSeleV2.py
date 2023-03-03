'''
Perform multi-channel kws and channel selection by fsmn.

Copyright: 2022-03-11 yueyue.nyy
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
from FSMN import RectifiedLinear
from FSMN import Fsmn
from Attention import Attention


'''
one multi-channel fsmn unit
dimlinear:              input / output dimension
dimproj:                fsmn input / output dimension
lorder:                 left ofder
rorder:                 right order
'''
class FSMNUnit(nn.Module):
    
    def __init__(self, dimlinear = 128, dimproj = 64, lorder = 20, rorder = 1):
        super(FSMNUnit, self).__init__()
        
        self.shrink = LinearTransform(dimlinear, dimproj)
        self.fsmn = Fsmn(dimproj, dimproj, lorder, rorder, 1, 1)
        self.expand = AffineTransform(dimproj, dimlinear)
        
        self.debug = False
        self.dataout = None
    
    
    '''
    batch, time, channel, feature
    '''
    def forward(self, input):
        if torch.cuda.is_available():
            out = torch.zeros(input.shape).cuda()
        else:
            out = torch.zeros(input.shape)
        
        for n in range(input.shape[2]):
            out1 = self.shrink(input[:, :, n, :])
            out2 = self.fsmn(out1)
            out[:, :, n, :] = F.relu(self.expand(out2))
        
        if self.debug:
            self.dataout = out
        
        return out
    
    
    def printModel(self):
        self.shrink.printModel()
        self.fsmn.printModel()
        self.expand.printModel()
    
    
    def toKaldiNNet(self):
        re_str = self.shrink.toKaldiNNet()
        re_str += self.fsmn.toKaldiNNet()
        re_str += self.expand.toKaldiNNet()
        
        relu = RectifiedLinear(self.expand.linear.out_features, self.expand.linear.out_features)
        re_str += relu.toKaldiNNet()
        
        return re_str
    
    
# dimlinear = 3
# dimproj = 2
# lorder = 2
# rorder = 1
# model = FSMNUnit(dimlinear, dimproj, lorder, rorder)
# input = torch.randn(2, 10, 2, dimlinear)
# output = model(input)


'''
FSMN model with channel selection.
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of fsmn units
sele_layer:             channel selection layer index
'''
class FSMNSeleNetV2(nn.Module):
    
    def __init__(self, input_dim = 120, linear_dim = 128, proj_dim = 64, 
                 lorder = 20, rorder = 1, num_syn = 5, fsmn_layers = 5, sele_layer = 0):
        super(FSMNSeleNetV2, self).__init__()
        
        self.sele_layer = sele_layer
        
        self.featmap = AffineTransform(input_dim, linear_dim)
        
        self.mem = []
        for l in range(fsmn_layers):
            unit = FSMNUnit(linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(l), unit)
        
        self.decision = AffineTransform(linear_dim, num_syn)
    
    
    def forward(self, input):
        # multi-channel feature mapping
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.featmap.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.featmap.linear.out_features)
        
        for n in range(input.shape[2]):
            x[:, :, n, :] = F.relu(self.featmap(input[:, :, n, :]))
        
        for l, unit in enumerate(self.mem):
            y = unit(x)
            
            # perform channel selection
            if l == self.sele_layer:
                pool = nn.MaxPool2d((y.shape[2], 1), stride = (y.shape[2], 1))
                y = pool(y)
            
            x = y
        
        # remove channel dimension
        y = torch.squeeze(y, -2)
        z = self.decision(y)
        
        return z
    
    
    def printModel(self):
        self.featmap.printModel()
        
        for unit in self.mem:
            unit.printModel()
        
        self.decision.printModel()
    
    
    def printHeader(self):
        '''
        get FSMN params
        '''
        input_dim = self.featmap.linear.in_features
        linear_dim = self.featmap.linear.out_features
        proj_dim = self.mem[0].shrink.linear.out_features
        lorder = self.mem[0].fsmn.conv_left.kernel_size[0]
        rorder = 0
        if self.mem[0].fsmn.conv_right is not None:
            rorder = self.mem[0].fsmn.conv_right.kernel_size[0]
        
        num_syn = self.decision.linear.out_features
        fsmn_layers = len(self.mem)
        
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
            header[HEADER_BLOCK_SIZE * hidx + 7] = float(self.sele_layer)
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
    
    
    def toKaldiNNet(self):
        re_str = '<Nnet>\n'
        
        re_str = self.featmap.toKaldiNNet()
        
        relu = RectifiedLinear(self.featmap.linear.out_features, self.featmap.linear.out_features)
        re_str += relu.toKaldiNNet()
        
        for unit in self.mem:
            re_str += unit.toKaldiNNet()
        
        re_str += self.decision.toKaldiNNet()
        
        re_str += '<Softmax> %d %d\n' % (self.decision.linear.out_features, self.decision.linear.out_features)
        re_str += '<!EndOfComponent>\n'
        re_str += '</Nnet>\n'
        
        return re_str


# dimin = 2
# dimlinear = 3
# dimproj = 2
# lorder = 2
# rorder = 1
# model = FSMNSeleNetV2(dimin, dimlinear, dimproj, lorder, rorder, num_syn = 2, fsmn_layers = 1, sele_layer = 0)
# model = model.cuda()
# input = torch.randn(2, 3, 2, dimin).cuda()
# output = model(input)
# 
# x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], model.featmap.linear.out_features).cuda()
# for n in range(input.shape[2]):
#     x[:, :, n, :] = F.relu(model.featmap(input[:, :, n, :]))
# 
# for l, unit in enumerate(model.mem):
#     y = unit(x)
#     x = y
#             
#     # perform channel selection
#     if l == model.sele_layer:
#         pool = nn.MaxPool2d((x.shape[2], 1), stride = (x.shape[2], 1))
#         y = pool(x)


'''
one single-channel fsmn unit
dimlinear:              input / output dimension
dimproj:                fsmn input / output dimension
lorder:                 left ofder
rorder:                 right order
'''
class MonoFSMNUnit(nn.Module):
    
    def __init__(self, dimlinear = 128, dimproj = 64, lorder = 20, rorder = 1):
        super(MonoFSMNUnit, self).__init__()
        
        self.shrink = LinearTransform(dimlinear, dimproj)
        self.fsmn = Fsmn(dimproj, dimproj, lorder, rorder, 1, 1)
        self.expand = AffineTransform(dimproj, dimlinear)
    
    
    '''
    batch, time, feature
    '''
    def forward(self, input):
        out1 = self.shrink(input)
        out2 = self.fsmn(out1)
        out3 = F.relu(self.expand(out2))
        
        return out3


'''
perform multi-channel fusion by data concatenation
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of fsmn units
numins:                 no. of input channels
'''
class ConcatNet(nn.Module):
    
    def __init__(self, input_dim = 120, linear_dim = 128, proj_dim = 64, 
                 lorder = 20, rorder = 1, num_syn = 5, fsmn_layers = 5, numins = 2):
        super(ConcatNet, self).__init__()
        
        self.featmap = AffineTransform(input_dim, linear_dim)
        
        self.mem = []
        for l in range(fsmn_layers):
            unit = MonoFSMNUnit(linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(l), unit)
        
        self.decision = AffineTransform(linear_dim * numins, num_syn)
    
    
    def forward(self, input):
        # multi-channel temp space, [batch, time, feature * channel]
        if torch.cuda.is_available():
            ycat = torch.zeros(input.shape[0], input.shape[1], self.decision.linear.in_features).cuda()
        else:
            ycat = torch.zeros(input.shape[0], input.shape[1], self.decision.linear.in_features)
        
        for n in range(input.shape[2]):
            x = F.relu(self.featmap(input[:, :, n, :]))
            
            for unit in self.mem:
                y = unit(x)
                x = y
            
            # data concatenation
            ycat[:, :, y.shape[2] * n:y.shape[2] * (n + 1)] = y[:, :, :]
        
        # the final decision layer
        z = self.decision(ycat)
        
        return z


'''
QKV attention

Gong, Rong, et al. "Self-attention channel combinator frontend for 
end-to-end multichannel far-field speech recognition." arXiv preprint 
arXiv:2109.04783 (2021).

dimin:                  input dimension
dimatt:                 attention dimension
'''
class QKVAttention(nn.Module):
    
    def __init__(self, dimin = 128, dimatt = 128):
        super(QKVAttention, self).__init__()
        
        self.qnet = AffineTransform(dimin, dimatt)
        self.knet = AffineTransform(dimin, dimatt)
        self.vnet = AffineTransform(dimin, 1)
        
        self.w = None
        self.keepw = False
    
    
    '''
    input:              batch, time, channel, feature
    return:             batch, time, feature
                        channels are weighted averaged
    '''
    def forward(self, input):
        dimatt = self.qnet.linear.out_features
        if torch.cuda.is_available():
            q = torch.zeros(input.shape[0], input.shape[1], input.shape[2], dimatt).cuda()
            k = torch.zeros(input.shape[0], input.shape[1], input.shape[2], dimatt).cuda()
            v = torch.zeros(input.shape[0], input.shape[1], input.shape[2], 1).cuda()
        else:
            q = torch.zeros(input.shape[0], input.shape[1], input.shape[2], dimatt)
            k = torch.zeros(input.shape[0], input.shape[1], input.shape[2], dimatt)
            v = torch.zeros(input.shape[0], input.shape[1], input.shape[2], 1)
        
        for n in range(input.shape[2]):
            chin = input[:, :, n, :]
            q[:, :, n, :] = self.qnet(chin)
            k[:, :, n, :] = self.knet(chin)
            v[:, :, n, :] = self.vnet(chin)
        
        watt = torch.matmul(q, torch.transpose(k, 2, 3)) / (dimatt ** 0.5)
        watt = F.softmax(watt, dim = -1)
        
        w = torch.matmul(watt, v)
        w = F.softmax(w, dim = -2)
        
        if self.keepw:
            self.w = w
        
        if torch.cuda.is_available():
            retval = torch.zeros(input.shape[0], input.shape[1], input.shape[3]).cuda()
        else:
            retval = torch.zeros(input.shape[0], input.shape[1], input.shape[3])
        
        for n in range(input.shape[2]):
            retval += w[:, :, n] * input[:, :, n, :]
        
        return retval
        

# dimin = 2
# dimatt = 3
# input = torch.randn(2, 4, 2, dimin)
#
# model = QKVAttention(dimin, dimatt).cuda()
# input = input.cuda()
# output = model(input)
#
# q = torch.zeros(input.shape[0], input.shape[1], input.shape[2], dimatt).cuda()
# k = torch.zeros(input.shape[0], input.shape[1], input.shape[2], dimatt).cuda()
# v = torch.zeros(input.shape[0], input.shape[1], input.shape[2], 1).cuda()
#
# for n in range(input.shape[2]):
#     chin = input[:, :, n, :]
#     q[:, :, n, :] = model.qnet(chin)
#     k[:, :, n, :] = model.knet(chin)
#     v[:, :, n, :] = model.vnet(chin)
#
# watt = torch.matmul(q, torch.transpose(k, 2, 3)) / (dimatt ** 0.5)
# watt = F.softmax(watt, dim = -1)
#
# w = torch.matmul(watt, v)
# w = F.softmax(w, dim = -2)
#
# retval = torch.zeros(input.shape[0], input.shape[1], input.shape[3]).cuda()
# for n in range(input.shape[2]):
#     retval += w[:, :, n] * input[:, :, n, :]

'''
attention fsmn net
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of fsmn units
dimatt:                 attention dimension
'''
class QKVAttFSMNNet(nn.Module):
    
    def __init__(self, input_dim = 120, linear_dim = 128, proj_dim = 64, 
                 lorder = 20, rorder = 1, num_syn = 5, fsmn_layers = 5, dimatt = 128):
        super(QKVAttFSMNNet, self).__init__()
        
        self.featmap = AffineTransform(input_dim, linear_dim)
        
        self.mem = []
        for l in range(fsmn_layers):
            unit = FSMNUnit(linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(l), unit)
        
        self.att = QKVAttention(linear_dim, dimatt)
        self.decision = AffineTransform(linear_dim, num_syn)
    
    
    def forward(self, input):
        # multi-channel feature mapping
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.featmap.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.featmap.linear.out_features)
        
        for n in range(input.shape[2]):
            x[:, :, n, :] = F.relu(self.featmap(input[:, :, n, :]))
        
        for unit in self.mem:
            y = unit(x)
            x = y
        
        y1 = self.att(y)
        z = self.decision(y1)
        
        return z


'''
attention fsmn net
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of fsmn units
dimatt:                 attention dimension
'''
class AttFSMNNetV2(nn.Module):
    
    def __init__(self, input_dim = 120, linear_dim = 128, proj_dim = 64, 
                 lorder = 20, rorder = 1, num_syn = 5, fsmn_layers = 5, dimatt = 128):
        super(AttFSMNNetV2, self).__init__()
        
        self.featmap = AffineTransform(input_dim, linear_dim)
        
        self.mem = []
        for l in range(fsmn_layers):
            unit = FSMNUnit(linear_dim, proj_dim, lorder, rorder)
            self.mem.append(unit)
            self.add_module('mem_{:d}'.format(l), unit)
        
        self.att = Attention(linear_dim, dimatt)
        self.decision = AffineTransform(linear_dim, num_syn)
    
    
    def forward(self, input):
        # multi-channel feature mapping
        if torch.cuda.is_available():
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.featmap.linear.out_features).cuda()
        else:
            x = torch.zeros(input.shape[0], input.shape[1], input.shape[2], self.featmap.linear.out_features)
        
        for n in range(input.shape[2]):
            x[:, :, n, :] = F.relu(self.featmap(input[:, :, n, :]))
        
        for unit in self.mem:
            y = unit(x)
            x = y
        
        y1 = self.att(y)
        z = self.decision(y1)
        
        return z
