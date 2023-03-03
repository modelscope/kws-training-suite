'''
Generate test data.
'''

import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import KWSDataset


CONF_VAL_PATH = '../conf/basetrain_normal.conf'
NUM_CHANNELS = 1
NUM_CLASSES = 5
BLOCK_DEC = 2
BLOCK_CAT = 3
NUM_WORKERS = 1
BATCH_SIZE = 1


'''
generate validation set
'''
dataset = KWSDataset(CONF_VAL_PATH, NUM_CHANNELS, NUM_CLASSES, BLOCK_DEC, BLOCK_CAT, NUM_WORKERS)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)
it = iter(dataloader)

feat = next(it)[0]
torch.save(feat, 'feat.pt')
