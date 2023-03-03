'''
print input and output for each layer
'''

import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FSMN import FSMNNet

model = torch.load('checkpoint_483_model_1_loss_train_0.09871919453144073_loss_val_0.07424543052911758.pth', map_location = 'cpu')
input = torch.load('feat.pt')

input = torch.squeeze(input, -2)
input = input[:, :100, :]

output = model(input)
