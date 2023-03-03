import math
import os
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FSMNSeleV2 import FSMNSeleNetV2
from FSMNSeleV3 import FSMNSeleNetV3
from FSMN import FSMNNet
from RNNVAD import RNNVAD
from RNNVAD import RNNVAD2


def printNumParams(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

# VAD model
# NUM_CLASSES = 2
# input_dim = 40
# input_dim2 = input_dim * 2
#
# printNumParams(FSMNNet(input_dim, linear_dim = 48, proj_dim = 24, lorder = 10, 
#                        rorder = 0, num_syn = NUM_CLASSES, fsmn_layers = 2))
#
# printNumParams(FSMNNet(input_dim, linear_dim = 72, proj_dim = 24, lorder = 10, 
#                        rorder = 0, num_syn = NUM_CLASSES, fsmn_layers = 2))
#
# printNumParams(RNNVAD(input_dim, dimproj = 24))
#
# printNumParams(RNNVAD(input_dim, dimproj = 48))
#
# printNumParams(FSMNNet(input_dim2, linear_dim = 48, proj_dim = 24, lorder = 10, 
#                        rorder = 0, num_syn = NUM_CLASSES, fsmn_layers = 2))
#
# printNumParams(FSMNNet(input_dim2, linear_dim = 72, proj_dim = 24, lorder = 10, 
#                        rorder = 0, num_syn = NUM_CLASSES, fsmn_layers = 2))
#
# printNumParams(RNNVAD(input_dim2, dimproj = 24))
#
# printNumParams(RNNVAD(input_dim2, dimproj = 40))

NUM_CLASSES = 5
input_dim = 120

# soundpro
# 123k
printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 144, proj_dim = 68, lorder = 16, rorder = 1,
                             num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))

# 502k
printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 304, proj_dim = 148, lorder = 16, rorder = 1,
                             num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))

# 501k
printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 276, proj_dim = 136, lorder = 16, rorder = 1,
                             num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))

# 1008k
printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 432, proj_dim = 216, lorder = 16, rorder = 1,
                             num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))

# 1014k
printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 400, proj_dim = 196, lorder = 16, rorder = 1,
                             num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))


# add reference channel
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 128, proj_dim = 64, lorder = 20, rorder = 1, 
#                              num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))

# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 144, proj_dim = 68, lorder = 16, rorder = 1, 
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 144, proj_dim = 68, lorder = 10, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 148, proj_dim = 68, lorder = 10, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 148, proj_dim = 72, lorder = 10, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))

# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 144, proj_dim = 68, lorder = 8, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))

# # TMJL big
# # 123k
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 144, proj_dim = 68, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# # 156k
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 160, proj_dim = 80, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# # 185k
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 176, proj_dim = 88, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# # 201k
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 184, proj_dim = 92, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# # 252k
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 208, proj_dim = 104, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))
#
# # 502k
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 304, proj_dim = 148, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))

# xiao ya
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 128, proj_dim = 64, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))
#
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 152, proj_dim = 52, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))
#
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 176, proj_dim = 44, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))
#
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 144, proj_dim = 68, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))


# sweeper
# model = FSMNSeleNetV2(input_dim, linear_dim = 168, proj_dim = 84, lorder = 20, rorder = 1,
#                       num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 0)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')
# 
# model = FSMNSeleNetV2(input_dim, linear_dim = 160, proj_dim = 72, lorder = 20, rorder = 1,
#                       num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 0)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')
# 
# model = FSMNSeleNetV2(input_dim, linear_dim = 128, proj_dim = 80, lorder = 20, rorder = 1,
#                       num_syn = NUM_CLASSES, fsmn_layers = 7, sele_layer = 0)
# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')


# Tai Ling
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 128, proj_dim = 80, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 7, sele_layer = 6))
# 
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 140, proj_dim = 72, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 7, sele_layer = 6))
# 
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 156, proj_dim = 76, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))
# 
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 164, proj_dim = 72, lorder = 16, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))
# 
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 256, proj_dim = 128, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 6, sele_layer = 5))

# CMD100
# NUM_CLASSES = 80
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 360, proj_dim = 84, lorder = 20, rorder = 1, 
#                              num_syn = NUM_CLASSES, fsmn_layers = 7, sele_layer = 6))
# 
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 308, proj_dim = 100, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 7, sele_layer = 6))
# 
# printNumParams(FSMNSeleNetV2(input_dim, linear_dim = 254, proj_dim = 124, lorder = 20, rorder = 1,
#                              num_syn = NUM_CLASSES, fsmn_layers = 7, sele_layer = 6))
