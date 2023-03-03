'''
Print model.
'''

import sys
import numpy as np
import torch
import torch.nn as nn
from FSMN import FSMNNet
from FSMNSele import FSMNSeleNet
from FSMNSeleV2 import FSMNSeleNetV2
from FSMNSeleV3 import FSMNSeleNetV3
from Attention import AttFSMNNet
from Attention import StreamingAttKWS
from RNNVAD import RNNVAD
from RNNVAD import RNNVAD2


if len(sys.argv) != 2:
    sys.stderr.write("Usage:\n");
    sys.stderr.write("PrintModel <.pth file path>\n")
    exit(-1)

model = torch.load(sys.argv[1], map_location = 'cpu')
model.printHeader()
model.printModel()
