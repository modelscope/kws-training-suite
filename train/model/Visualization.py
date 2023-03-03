'''
Data visualization.

Copyright: 2022-04-14 yueyue.nyy
'''

import numpy as np
import torch
import torch.nn.functional as F

FBANK_SIZE = 40
BLOCK_DECIMATION = 2
BLOCK_CAT = 3
NUM_CHS = 1

# load input feature
data = np.fromfile('../input_feat.f32', dtype = 'float32')
data = np.reshape(data, (data.shape[0] // (FBANK_SIZE * NUM_CHS), FBANK_SIZE * NUM_CHS))

# decimated size
size1 = int(np.ceil(data.shape[0] / BLOCK_DECIMATION)) - BLOCK_CAT + 1

# feature decimation and concatenation
# time x channel x feature
featall = np.zeros((size1, NUM_CHS, FBANK_SIZE * BLOCK_CAT), dtype = 'float32')

for n in range(NUM_CHS):
    feat = data[:, FBANK_SIZE * n:FBANK_SIZE * (n + 1)]
    
    for tau in range(size1):
        for i in range(BLOCK_CAT):
            featall[tau, n, FBANK_SIZE * i:FBANK_SIZE * (i + 1)] = feat[(tau + i) * BLOCK_DECIMATION, :]

datain = torch.from_numpy(featall)

# load model
model = torch.load('../checkpoint_0130_model_05_loss_train_0.06826995313167572_loss_val_0.0671650618314743.pth', 
                   map_location = 'cpu')

# set debug tags
model.featmap.debug = True

for m in model.mem:
    m.shrink.debug = True
    m.fsmn.debug = True
    m.expand.debug = True
    m.debug = True

model.decision.debug = True

# batch, time, channel, feature
datain = torch.unsqueeze(datain, 0)

# apply model
dataout = model(datain)
dataout = torch.squeeze(dataout, 0)

pout = F.softmax(dataout, dim = 1)

# get output of each layer
datadict = {}

datadict.update({'datain': datain})

datadict.update({'featmap': model.featmap.dataout})

for lidx, layer in enumerate(model.mem):
    datadict.update({'mem_{:d}.shrink'.format(lidx): layer.shrink.dataout})
    datadict.update({'mem_{:d}.fsmn'.format(lidx): layer.fsmn.dataout})
    datadict.update({'mem_{:d}.expand'.format(lidx): layer.expand.dataout})
    datadict.update({'mem_{:d}'.format(lidx): layer.dataout})

datadict.update({'decision': F.softmax(model.decision.dataout, dim = -1)})

# squeeze to matrices
for key in datadict.keys():
    tmpdata = datadict[key]
    
    if tmpdata.shape[0] == 1:
        tmpdata = torch.squeeze(tmpdata, 0)
    
    if tmpdata.shape[-2] == 1:
        tmpdata = torch.squeeze(tmpdata, -2)
    
    datadict.update({key: tmpdata.T})

# save data
for l, key in enumerate(datadict.keys()):
    np.savetxt('dataout/{:02d}_'.format(l) + key + '.txt', datadict[key].detach().numpy())
