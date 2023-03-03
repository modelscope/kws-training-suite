'''
FSMN implementation.

Copyright: 2022-03-09 yueyue.nyy
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


'''
function that transform as str numpy mat to standard kaldi str matrix
:param np_mat:          numpy mat
:return:                str
'''
def toKaldiMatrix(np_mat):
    np.set_printoptions(threshold = np.inf, linewidth = np.nan)
    out_str = str(np_mat)
    out_str = out_str.replace('[', '')
    out_str = out_str.replace(']', '')
    return '[ %s ]\n' % out_str


'''
print torch tensor for debug
torch_tensor:           a tensor
'''
def printTensor(torch_tensor):
    re_str = ''
    x = torch_tensor.detach().squeeze().numpy()
    re_str += toKaldiMatrix(x)
    re_str += '<!EndOfComponent>\n'
    print(re_str)


class LinearTransform(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim, bias = False)
        
        self.debug = False
        self.dataout = None
    
    
    def forward(self, input):
        output = self.linear(input)
        
        if self.debug:
            self.dataout = output
        
        return output
    
    
    def printModel(self):
        printNeonMatrix(self.linear.weight)
    
    
    def toKaldiNNet(self):
        re_str = ''
        re_str += '<LinearTransform> %d %d\n' % (self.output_dim, self.input_dim)
        re_str += '<LearnRateCoef> 1\n'
        
        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += toKaldiMatrix(x)
        re_str += '<!EndOfComponent>\n'
        
        return re_str
    

class AffineTransform(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        self.debug = False
        self.dataout = None
    
    
    def forward(self, input):        
        output = self.linear(input)
        
        if self.debug:
            self.dataout = output
        
        return output
    
    
    def printModel(self):
        printNeonMatrix(self.linear.weight)
        printNeonVector(self.linear.bias)
    
    
    def toKaldiNNet(self):
        re_str = ''
        re_str += '<AffineTransform> %d %d\n' % (self.output_dim, self.input_dim)
        re_str += '<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0\n'
        
        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += toKaldiMatrix(x)
        
        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += toKaldiMatrix(x)
        re_str += '<!EndOfComponent>\n'
        
        return re_str


class Fsmn(nn.Module):
    
    def __init__(self, input_dim, output_dim, lorder = None, rorder = None, lstride = None, rstride = None):
        super(Fsmn, self).__init__()
        
        self.dim = input_dim
        
        if lorder is None: return
        
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        
        self.conv_left  = nn.Conv2d(
            self.dim, self.dim, [lorder, 1], dilation = [lstride, 1], groups = self.dim, bias = False)
        
        if rorder > 0:
            self.conv_right = nn.Conv2d(
                self.dim, self.dim, [rorder, 1], dilation = [rstride, 1], groups = self.dim, bias = False)
        else:
            self.conv_right = None
        
        self.debug = False
        self.dataout = None
    
    
    def forward(self, input):
        x = torch.unsqueeze(input, 1)
        x_per = x.permute(0, 3, 2, 1)
        
        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])
        
        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)
        
        out1 = out.permute(0, 3, 2, 1)
        output = out1.squeeze(1)
        
        if self.debug:
            self.dataout = output
        
        return output
    
    
    def printModel(self):
        tmpw = self.conv_left.weight
        tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
        for j in range(tmpw.shape[0]):
            tmpwm[:, j] = tmpw[j, 0, :, 0]
        
        printNeonMatrix(tmpwm)
        
        if self.conv_right is not None:
            tmpw = self.conv_right.weight
            tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
            for j in range(tmpw.shape[0]):
                tmpwm[:, j] = tmpw[j, 0, :, 0]
            
            printNeonMatrix(tmpwm)
    
    
    def toKaldiNNet(self):
        re_str = ''
        re_str += '<Fsmn> %d %d\n' % (self.dim, self.dim)
        re_str += '<LearnRateCoef> %d <LOrder> %d <ROrder> %d <LStride> %d <RStride> %d <MaxNorm> 0\n' % (
            1, self.lorder, self.rorder, self.lstride, self.rstride)
        
        #print(self.conv_left.weight,self.conv_right.weight)
        lfiters = self.state_dict()['conv_left.weight']
        x = np.flipud(lfiters.squeeze().numpy().T)
        re_str += toKaldiMatrix(x)
        
        if self.conv_right is not None:
            rfiters = self.state_dict()['conv_right.weight']
            x = (rfiters.squeeze().numpy().T)
            re_str += toKaldiMatrix(x)
            re_str += '<!EndOfComponent>\n'
        
        return re_str
    

class RectifiedLinear(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__()
        self.dim = input_dim
        self.relu = nn.ReLU()
    
    
    def forward(self, input):
        return self.relu(input)
    
    
    def toKaldiNNet(self):
        re_str = ''
        re_str += '<RectifiedLinear> %d %d\n' % (self.dim, self.dim)
        re_str += '<!EndOfComponent>\n'
        return re_str
        
        # re_str = ''
        # re_str += '<ParametricRelu> %d %d\n' % (self.dim, self.dim)
        # re_str += '<AlphaLearnRateCoef> 0 <BetaLearnRateCoef> 0\n'
        # re_str += toKaldiMatrix(np.ones((self.dim), dtype = 'int32'))
        # re_str += toKaldiMatrix(np.zeros((self.dim), dtype = 'int32'))
        # re_str += '<!EndOfComponent>\n'
        # return re_str
    

def _build_repeats(linear_dim = 136, proj_dim = 68, lorder = 3, rorder = 2, fsmn_layers=5):
    repeats = [
        nn.Sequential(
            LinearTransform(linear_dim, proj_dim), 
            Fsmn(proj_dim, proj_dim, lorder, rorder, 1, 1), 
            AffineTransform(proj_dim, linear_dim), 
            RectifiedLinear(linear_dim, linear_dim))
        for i in range(fsmn_layers)
    ]
    
    return nn.Sequential(*repeats)



'''
FSMN net for keyword spotting
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
'''
class FSMNNet(nn.Module):
    
    def __init__(self, input_dim = 200, linear_dim = 128, proj_dim = 128, lorder = 10, rorder = 1, num_syn = 5, fsmn_layers = 4):
        super(FSMNNet, self).__init__()
        
        self.input_dim = input_dim
        self.linear_dim = linear_dim
        self.proj_dim = proj_dim
        self.lorder = lorder
        self.rorder = rorder
        self.num_syn = num_syn
        self.fsmn_layers = fsmn_layers
        
        self.linear1 = AffineTransform(input_dim, linear_dim)
        self.relu = RectifiedLinear(linear_dim, linear_dim)
        
        self.fsmn = _build_repeats(linear_dim, proj_dim, lorder, rorder, fsmn_layers)
        
        self.linear2 = AffineTransform(linear_dim, num_syn)
        # self.sig = nn.LogSoftmax(dim = -1)
    
    
    def forward(self, input):
        x1 = self.linear1(input)
        x2 = self.relu(x1)
        x3 = self.fsmn(x2)
        x4 = self.linear2(x3)
        
        return x4
        # x5 = self.sig(x4)
        # return x5
    
    
    def printModel(self):
        self.linear1.printModel()
        
        for l in self.fsmn:
            l[0].printModel()
            l[1].printModel()
            l[2].printModel()
        
        self.linear2.printModel()
        
    
    def printHeader(self):
        #
        # write total header
        #
        header = [0.0] * HEADER_BLOCK_SIZE * 4
        # numins
        header[0] = 0.0
        # numouts
        header[1] = 0.0
        # dimins
        header[2] = self.input_dim
        # dimouts
        header[3] = self.num_syn
        # numlayers
        header[4] = 3
        
        #
        # write each layer's header
        #
        hidx = 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.input_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_RELU.value)
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_SEQUENTIAL_FSMN.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.proj_dim
        header[HEADER_BLOCK_SIZE * hidx + 4] = self.lorder
        header[HEADER_BLOCK_SIZE * hidx + 5] = self.rorder
        header[HEADER_BLOCK_SIZE * hidx + 6] = self.fsmn_layers
        header[HEADER_BLOCK_SIZE * hidx + 7] = -1.0
        hidx += 1
        
        header[HEADER_BLOCK_SIZE * hidx + 0] = float(LayerType.LAYER_DENSE.value)
        header[HEADER_BLOCK_SIZE * hidx + 1] = 0.0
        header[HEADER_BLOCK_SIZE * hidx + 2] = self.linear_dim
        header[HEADER_BLOCK_SIZE * hidx + 3] = self.num_syn
        header[HEADER_BLOCK_SIZE * hidx + 4] = 1.0
        header[HEADER_BLOCK_SIZE * hidx + 5] = float(ActivationType.ACTIVATION_SOFTMAX.value)
        
        for h in header:
            print(f32ToI32(h))
    
    
    def toKaldiNNet(self):
        re_str = ''
        re_str += '<Nnet>\n'
        re_str += self.linear1.toKaldiNNet()
        re_str += self.relu.toKaldiNNet()
        
        for fsmn in self.fsmn:
            re_str += fsmn[0].toKaldiNNet()
            re_str += fsmn[1].toKaldiNNet()
            re_str += fsmn[2].toKaldiNNet()
            re_str += fsmn[3].toKaldiNNet()
        
        re_str += self.linear2.toKaldiNNet()
        re_str += '<Softmax> %d %d\n' % (self.num_syn, self.num_syn)
        re_str += '<!EndOfComponent>\n'
        re_str += '</Nnet>\n'
        
        return re_str


# model = FSMNNet()
# model.printHeader()
# model.printModel()


'''
one deep fsmn layer
dimproj:                projection dimension, input and output dimension of memory blocks
dimlinear:              dimension of mapping layer
lorder:                 left order
rorder:                 right order
lstride:                left stride
rstride:                right stride
'''
class DFSMN(nn.Module):
    
    def __init__(self, dimproj = 64, dimlinear = 128, lorder = 20, rorder = 1, lstride = 1, rstride = 1):
        super(DFSMN, self).__init__()
        
        self.lorder = lorder
        self.rorder = rorder
        self.lstride = lstride
        self.rstride = rstride
        
        self.expand = AffineTransform(dimproj, dimlinear)
        self.shrink = LinearTransform(dimlinear, dimproj)
        
        self.conv_left  = nn.Conv2d(
            dimproj, dimproj, [lorder, 1], dilation = [lstride, 1], groups = dimproj, bias = False)
        
        if rorder > 0:
            self.conv_right = nn.Conv2d(
                dimproj, dimproj, [rorder, 1], dilation = [rstride, 1], groups = dimproj, bias = False)
        else:
            self.conv_right = None
    
    
    def forward(self, input):
        f1 = F.relu(self.expand(input))
        p1 = self.shrink(f1)
        
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        
        y_left = F.pad(x_per, [0, 0, (self.lorder - 1) * self.lstride, 0])
        
        if self.conv_right is not None:
            y_right = F.pad(x_per, [0, 0, 0, (self.rorder) * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            out = x_per + self.conv_left(y_left) + self.conv_right(y_right)
        else:
            out = x_per + self.conv_left(y_left)
        
        out1 = out.permute(0, 3, 2, 1)
        output = input + out1.squeeze(1)
        
        return output
    
    
    def printModel(self):
        self.expand.printModel()
        self.shrink.printModel()
        
        tmpw = self.conv_left.weight
        tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
        for j in range(tmpw.shape[0]):
            tmpwm[:, j] = tmpw[j, 0, :, 0]
        
        printNeonMatrix(tmpwm)
        
        if self.conv_right is not None:
            tmpw = self.conv_right.weight
            tmpwm = torch.zeros(tmpw.shape[2], tmpw.shape[0])
            for j in range(tmpw.shape[0]):
                tmpwm[:, j] = tmpw[j, 0, :, 0]
            
            printNeonMatrix(tmpwm)


# dimproj = 1
# dimlinear = 2
# lorder = 3
# rorder = 2
# model = DFSMN(dimproj, dimlinear, lorder, rorder, 1, 1)
# 
# input = torch.randn(1, 10, dimproj)
# input[0, :, 0] = torch.tensor([1, 1, 3, 2, -1, 3, 2, 5, 4, -3])
# f1 = F.relu(model.linear(input))
# p1 = model.proj(f1)
# 
# # p1[0, :, 0] = torch.tensor([1, 1, 3, 2, -1, 3, 2, 5, 4, -3])
# x = torch.unsqueeze(p1, 1)
# x_per = x.permute(0, 3, 2, 1)
# 
# y_left = F.pad(x_per, [0, 0, (model.lorder - 1) * model.lstride, 0])
# y_right = F.pad(x_per, [0, 0, 0, (model.rorder) * model.rstride])
# y_right = y_right[:, :, model.rstride:, :]
# cl = model.conv_left(y_left)
# cr = model.conv_right(y_right)
# out = x_per + cl + cr
# 
# out1 = out.permute(0, 3, 2, 1)
# output = input + out1.squeeze(1)


'''
build stacked dfsmn layers
'''
def buildDFSMNRepeats(linear_dim = 128, proj_dim = 64, lorder = 20, rorder = 1, fsmn_layers = 6):
    repeats = [
        nn.Sequential(
            DFSMN(proj_dim, linear_dim, lorder, rorder, 1, 1))
        for i in range(fsmn_layers)
    ]
    
    return nn.Sequential(*repeats)
