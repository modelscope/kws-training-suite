'''
Soft self-attention mechanism.
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
from FSMN import FSMNNet


'''
attention network
dimfeat:                input/output feature dimension
dimatt:                 attention projection dimension
'''
class Attention(nn.Module):
    
    def __init__(self, dimfeat, dimatt):
        super(Attention, self).__init__()
        
        self.proj = nn.Linear(dimfeat, dimatt)
        self.gvec = nn.Linear(dimatt, 1, bias = False)
        
        self.keepw = False
        self.w = None
    
    '''
    input:              batch x time x channel x feature
    '''
    def forward(self, input):
        # batch x time x channel
        weight = torch.zeros(input.shape[0], input.shape[1], input.shape[2]).cuda()
        
        # calculate attention weight
        for n in range(input.shape[2]):
            tmp1 = torch.tanh(self.proj(input[:, :, n, :]))
            weight[:, :, n] = self.gvec(tmp1)[:, :, 0]
        
        # normalize
        weight = F.softmax(weight, dim = -1)
        
        if self.keepw:
            self.w = weight
        
        # apply attention weight
        # batch x time x feature
        retval = torch.zeros(input.shape[0], input.shape[1], input.shape[3]).cuda()
        for n in range(input.shape[2]):
            retval += torch.unsqueeze(weight[:, :, n], -1) * input[:, :, n, :]
        
        return retval
    
    
    def printModel(self):
        printNeonMatrix(self.proj.weight)
        printNeonVector(self.proj.bias)
        printNeonMatrix(self.gvec.weight)


# numins = 3
# dimouts = 2
# dimatt = 5
# numbatches = 2
# batchsize = 10
# 
# proj = nn.Linear(dimouts, dimatt)
# gvec = nn.Linear(dimatt, 1, bias = False)
# 
# input = torch.randn(numbatches, batchsize, numins * dimouts)
# 
# weight = torch.zeros(input.shape[0], input.shape[1], numins)
# 
# for n in range(numins):
#     tmp1 = torch.tanh(proj(input[:, :, dimouts * n:dimouts * (n + 1)]))
#     weight[:, :, n] = gvec(tmp1)[:, :, 0]
# 
# weight = F.softmax(weight, dim = 2)
# 
# retval = torch.zeros(input.shape[0], input.shape[1], dimouts)
# for n in range(numins):
#     retval = retval + torch.unsqueeze(weight[:, :, n], 2) * input[:, :, dimouts * n:dimouts * (n + 1)]


'''
Attention based FSMN model.
ckptpath:               checkpoint path
dimatt:                 projection dimension for attention

input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
'''
class AttFSMNNet(nn.Module):
    
    def __init__(self, ckptpath = None, dimatt = 128, input_dim = 200, linear_dim = 128, 
                 proj_dim = 128, lorder = 10, rorder = 1, num_syn = 5, fsmn_layers = 4):
        super(AttFSMNNet, self).__init__()
        
        if ckptpath is None:
            self.fsmnnet = FSMNNet(input_dim, linear_dim, proj_dim, lorder, rorder, num_syn, fsmn_layers)
            dimins = input_dim
        else:
            self.fsmnnet = torch.load(ckptpath)
            dimins = self.fsmnnet.linear1.linear.in_features
        
        self.att = Attention(dimins, dimatt)
    
    
    def forward(self, input):
        wfeat = self.att(input)
        out = self.fsmnnet(wfeat)        
        return out
    
    
    def printModel(self):
        self.att.printModel()
        self.fsmnnet.printModel()
    
    
    def printHeader(self):
        '''
        get FSMN params
        input_dim:              input dimension
        linear_dim:             fsmn input dimension
        proj_dim:               fsmn projection dimension
        lorder:                 fsmn left order
        rorder:                 fsmn right order
        num_syn:                output dimension
        fsmn_layers:            no. of sequential fsmn layers
        '''
        input_dim = self.fsmnnet.linear1.linear.in_features
        linear_dim = self.fsmnnet.linear1.linear.out_features
        proj_dim = self.fsmnnet.fsmn[0][0].linear.out_features
        lorder = self.fsmnnet.fsmn[0][1].conv_left.kernel_size[0]
        rorder = 0
        if self.fsmnnet.fsmn[0][1].conv_right is not None:
            rorder = self.fsmnnet.fsmn[0][1].conv_right.kernel_size[0]
        
        num_syn = self.fsmnnet.linear2.linear.out_features
        fsmn_layers = len(self.fsmnnet.fsmn)
        
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 5
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 1.0
        # dimins
        header[2] = input_dim
        # dimouts
        header[3] = num_syn
        # numlayers
        header[4] = 4
        
        #
        # write each layer's header
        #
        hidx = 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_ATTENTION.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 3] = input_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = self.att.proj.out_features
        header[HEADER_BLOCK_SIZE * hidx + 5] = 1.0
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = input_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_RELU.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_SEQUENTIAL_FSMN.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = lorder
        header[HEADER_BLOCK_SIZE * hidx + 5] = rorder
        header[HEADER_BLOCK_SIZE * hidx + 6] = fsmn_layers
        header[HEADER_BLOCK_SIZE * hidx + 7] = float(ActivationType.ACTIVATION_RELU.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = num_syn
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SOFTMAX.value)
        
        for h in header:
            print(f32ToI32(h))


# batchsize = 10
# dimins = 200
# 
# featlist = []
# featlist.append(torch.randn(batchsize, dimins))
# featlist.append(torch.randn(batchsize, dimins))
# featlist.append(torch.randn(batchsize, dimins))
# 
# model = AttFSMNNet('/home/yueyue.nyy/data/kws_train/Sweeper/2021-07-26_changefeat/checkpoint/checkpoint_499_model_0_loss_train_0.083954356610775_loss_val_0.06602371484041214.pth', 128)
# output = model(featlist)


'''
attention in a streaming window
dimio:                  input / output dimension
dimatt:                 attention dimension
attwsize:               attention window size
'''
class StreamingAttention(nn.Module):
    
    def __init__(self, dimio, dimatt, attwsize):
        super(StreamingAttention, self).__init__()
        
        self.attwsize = attwsize
        self.proj = nn.Linear(dimio, dimatt)
        self.gvec = nn.Linear(dimatt, 1, bias = False)
        
        
    def forward(self, input):
        proj = torch.tanh(self.proj(input))
        # gvec = torch.exp(self.gvec(proj))
        gvec = torch.sigmoid(self.gvec(proj))
        
        winput = gvec * input
        retval = torch.zeros(input.shape).cuda()
        
        for tau in range(gvec.shape[1]):
            retval[:, tau, :] = torch.sum(
                winput[:, max(tau - self.attwsize + 1, 0):tau + 1, :], dim = 1) / torch.sum(
                    gvec[:, max(tau - self.attwsize + 1, 0):tau + 1, :], dim = 1)
        
        return retval
    
    
    def printModel(self):
        printNeonMatrix(self.proj.weight)
        printNeonVector(self.proj.bias)
        printNeonMatrix(self.gvec.weight)


# dimio = 2
# attwsize = 3
# numbatches = 2
# batchsize = 10
# 
# proj = nn.Linear(dimio, dimatt)
# gvec = nn.Linear(dimatt, 1, bias = False)
# 
# input = torch.randn(numbatches, batchsize, dimio)
# 
# proj1 = torch.tanh(proj(input))
# gvec1 = torch.exp(gvec(proj1))
# 
# winput = gvec1 * input
# retval = torch.zeros(input.shape)
# 
# for tau in range(gvec1.shape[1]):
#     retval[:, tau, :] = torch.sum(
#         winput[:, max(tau - attwsize + 1, 0):tau + 1, :], dim = 1) / torch.sum(
#             gvec1[:, max(tau - attwsize + 1, 0):tau + 1, :], dim = 1)


'''
streaming attention based kws
dimins:                 input dimension
dimproj:                projection dimension
dimatt:                 attention dimension
attwsize:               attention window size
dimouts:                output dimension
'''
class StreamingAttKWS(nn.Module):
    
    def __init__(self, dimins, dimproj, dimatt, attwsize, dimouts):
        super(StreamingAttKWS, self).__init__()
        
        self.dimins = dimins
        self.dimproj = dimproj
        self.dimatt = dimatt
        self.attwsize = attwsize
        self.dimouts = dimouts
        
        self.encoder = nn.GRU(dimins, dimproj, batch_first = True)
        self.att = StreamingAttention(dimproj, dimatt, attwsize)
        self.decision = nn.Linear(dimproj, dimouts)
    
    
    def forward(self, input):
        out0, _ = self.encoder(input)
        out1 = self.att(out0)
        out2 = self.decision(out1)
        return out2
    
    
    def printModel(self):
        printGRU(self.encoder)
        self.att.printModel()
        printDense(self.decision)
    
    
    def printHeader(self):        
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 1.0
        # dimins
        header[2] = self.dimins
        # dimouts
        header[3] = self.dimouts
        # numlayers
        header[4] = 3
        
        #
        # write each layer's header
        #
        hidx = 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_GRU.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.dimins
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.dimproj
        header[HEADER_BLOCK_SIZE * hidx + 4] = float(ActivationType.ACTIVATION_TANH.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_ATTENTION.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.dimproj
        header[HEADER_BLOCK_SIZE * hidx + 4] = self.dimatt
        header[HEADER_BLOCK_SIZE * hidx + 5] = self.attwsize
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.dimproj
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.dimouts
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SOFTMAX.value)
        
        for h in header:
            print(f32ToI32(h))


'''
attention with gru memory
dimfeat:                input/output feature dimension
dimatt:                 attention projection dimension
'''
class GRUAttention(nn.Module):
    
    def __init__(self, dimfeat, dimatt):
        super(GRUAttention, self).__init__()
        
        self.proj = nn.GRU(input_size = dimfeat, hidden_size = dimatt, batch_first = True)
        self.gvec = nn.Linear(dimatt, 1, bias = False)
    
    
    '''
    input:              batch x time x channel x feature
    '''
    def forward(self, input):
        # batch x time x channel
        weight = torch.zeros(input.shape[0], input.shape[1], input.shape[2]).cuda()
        
        # calculate attention weight
        for n in range(input.shape[2]):
            tmp1, _ = self.proj(input[:, :, n, :])
            tmp1 = torch.tanh(tmp1)
            
            weight[:, :, n] = self.gvec(tmp1)[:, :, 0]
        
        # normalize
        weight = F.softmax(weight, dim = -1)
        
        # apply attention weight
        # batch x time x feature
        retval = torch.zeros(input.shape[0], input.shape[1], input.shape[3]).cuda()
        for n in range(input.shape[2]):
            retval += torch.unsqueeze(weight[:, :, n], -1) * input[:, :, n, :]
        
        return retval
    
    
    def printModel(self):
        printGRU(self.proj)
        printNeonMatrix(self.gvec.weight)


# numchs = 3
# dimfeat = 2
# dimatt = 5
# numbatches = 2
# batchsize = 5
# 
# input = torch.randn(numbatches, batchsize, numchs, dimfeat)
# 
# proj = nn.GRU(input_size = dimfeat, hidden_size = dimatt, batch_first = True)
# gvec = nn.Linear(dimatt, 1, bias = False)
# 
# weight = torch.zeros(input.shape[0], input.shape[1], input.shape[2])
# 
# for n in range(input.shape[2]):
#     tmp1, _ = proj(input[:, :, n, :])
#     tmp1 = torch.tanh(tmp1)
#             
#     weight[:, :, n] = gvec(tmp1)[:, :, 0]
# 
# weight = F.softmax(weight, dim = -1)
#  
# retval = torch.zeros(input.shape[0], input.shape[1], input.shape[3])
# for n in range(input.shape[2]):
#     retval += torch.unsqueeze(weight[:, :, n], -1) * input[:, :, n, :]


'''
Attention based FSMN model.
ckptpath:               checkpoint path
dimatt:                 projection dimension for attention

input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
'''
class GRUAttFSMNNet(nn.Module):
    
    def __init__(self, ckptpath = None, dimatt = 128, input_dim = 120, linear_dim = 128, 
                 proj_dim = 64, lorder = 20, rorder = 1, num_syn = 5, fsmn_layers = 6):
        super(GRUAttFSMNNet, self).__init__()
        
        if ckptpath is None:
            self.fsmnnet = FSMNNet(input_dim, linear_dim, proj_dim, lorder, rorder, num_syn, fsmn_layers)
            dimins = input_dim
        else:
            self.fsmnnet = torch.load(ckptpath)
            dimins = self.fsmnnet.linear1.linear.in_features
        
        self.att = GRUAttention(dimins, dimatt)
    
    
    '''
    input:              batch x time x channel x feature
    '''
    def forward(self, input):
        wfeat = self.att(input)
        out = self.fsmnnet(wfeat)        
        return out
    
    
    def printModel(self):
        self.att.printModel()
        self.fsmnnet.printModel()
    
    
    def printHeader(self):
        '''
        get FSMN params
        input_dim:              input dimension
        linear_dim:             fsmn input dimension
        proj_dim:               fsmn projection dimension
        lorder:                 fsmn left order
        rorder:                 fsmn right order
        num_syn:                output dimension
        fsmn_layers:            no. of sequential fsmn layers
        '''
        input_dim = self.fsmnnet.linear1.linear.in_features
        linear_dim = self.fsmnnet.linear1.linear.out_features
        proj_dim = self.fsmnnet.fsmn[0][0].linear.out_features
        lorder = self.fsmnnet.fsmn[0][1].conv_left.kernel_size[0]
        rorder = 0
        if self.fsmnnet.fsmn[0][1].conv_right is not None:
            rorder = self.fsmnnet.fsmn[0][1].conv_right.kernel_size[0]
        
        num_syn = self.fsmnnet.linear2.linear.out_features
        fsmn_layers = len(self.fsmnnet.fsmn)
        
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 5
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 1.0
        # dimins
        header[2] = input_dim
        # dimouts
        header[3] = num_syn
        # numlayers
        header[4] = 4
        
        #
        # write each layer's header
        #
        hidx = 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_GRU_ATTENTION.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 3] = input_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = self.att.proj.out_features
        header[HEADER_BLOCK_SIZE * hidx + 5] = 1.0
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = input_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_RELU.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_SEQUENTIAL_FSMN.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = lorder
        header[HEADER_BLOCK_SIZE * hidx + 5] = rorder
        header[HEADER_BLOCK_SIZE * hidx + 6] = fsmn_layers
        header[HEADER_BLOCK_SIZE * hidx + 7] = float(ActivationType.ACTIVATION_RELU.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = num_syn
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SOFTMAX.value)
        
        for h in header:
            print(f32ToI32(h))
