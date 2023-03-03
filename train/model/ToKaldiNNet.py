'''
Convert to kaldi format.

Copyright: 2022-03-11 yueyue.nyy
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
from FSMNSeleV2 import FSMNSeleNetV2

model = torch.load(sys.argv[1], map_location = 'cpu')

ts = model.toKaldiNNet()
print(ts)
