'''
This is used to train tian mao jing ling kws.
'''

import math
import os
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base, 'data'))
sys.path.append(os.path.join(base, 'model'))

from KWSDataset import KWSDataset
from KWSDataLoader import KWSDataLoader
from FSMNSeleV2 import FSMNSeleNetV2
from FSMNSeleV3 import FSMNSeleNetV3


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#
# common params
#
BASETRAIN_CONF_VAL_PATH = '../conf/basetrain_normal.conf'
BASETRAIN_CONF_EASY_PATH = '../conf/basetrain_easy.conf'
BASETRAIN_CONF_NORMAL_PATH = '../conf/basetrain_normal.conf'
BASETRAIN_CONF_HARD_PATH = '../conf/basetrain_hard.conf'
FINETUNE_CONF_VAL_PATH = '../conf/finetune_normal.conf'
FINETUNE_CONF_EASY_PATH = '../conf/finetune_easy.conf'
FINETUNE_CONF_NORMAL_PATH = '../conf/finetune_normal.conf'
FINETUNE_CONF_HARD_PATH = '../conf/finetune_hard.conf'
FEAT_CAT_SIZE = 120
NUM_CLASSES = 5
BLOCK_DEC = 2
BLOCK_CAT = 3
NUM_WORKERS = 60
BASETRAIN_RATIO = 0.5
BATCH_SIZE = 20
PREFETCH_FACTOR = 20
NUM_BATCHES_TRAIN = 300
NUM_BATCHES_VAL = 150
NUM_EPOCHS = 500


'''
generate validation set
'''
dataset = KWSDataset(BASETRAIN_CONF_VAL_PATH, FINETUNE_CONF_VAL_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
                     NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
dataloader.startDataLoader()
it = iter(dataloader)

data_val = []
for bi in range(NUM_BATCHES_VAL):
    feat, label = next(it)
    label = torch.reshape(label, (-1,))
    data_val.append([feat, label])

del it
del dataloader
del dataset


'''
prepare model

FSMN model with channel selection.
input_dim:              input dimension
linear_dim:             fsmn input dimension
proj_dim:               fsmn projection dimension
lorder:                 fsmn left order
rorder:                 fsmn right order
num_syn:                output dimension
fsmn_layers:            no. of sequential fsmn layers
'''
model = []
input_dim = FEAT_CAT_SIZE

model.append(FSMNSeleNetV2(input_dim, linear_dim = 144, proj_dim = 68, lorder = 16, rorder = 1,
                           num_syn = NUM_CLASSES, fsmn_layers = 5, sele_layer = 4))


'''
build adam optimizer and corresponding cross entropy loss function
modell:                 model list
lr:                     learning rate
'''
def buildAdamCE(modell, lr):
    optl = []
    lossl = []
    
    for m in modell:
        optl.append(optim.Adam(m.parameters(), lr))
        lossl.append(nn.CrossEntropyLoss())
    
    return optl, lossl


# build corresponding optimizer and loss function
optimizer, loss_fn = buildAdamCE(model, lr = 5e-4)


usegpu = torch.cuda.is_available()
# usegpu = False

if usegpu:
    for i in range(len(model)):
        model[i] = model[i].cuda()
        loss_fn[i] = loss_fn[i].cuda()


'''
initial dataset
'''
dataset = KWSDataset(BASETRAIN_CONF_EASY_PATH, FINETUNE_CONF_EASY_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
                     NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
dataloader.startDataLoader()
it = iter(dataloader)


'''
training loop
'''
cmd = 'rm -rf ../checkpoint'
os.system(cmd)
cmd = 'mkdir ../checkpoint'
os.system(cmd)

totaltime = datetime.datetime.now()

for epoch in range(1, NUM_EPOCHS + 1):
    epochtime = datetime.datetime.now()
    
    #
    # change difficuity
    #
    if epoch == math.floor(NUM_EPOCHS * 0.1):
        del it
        del dataloader
        del dataset
        
        dataset = KWSDataset(BASETRAIN_CONF_NORMAL_PATH, FINETUNE_CONF_NORMAL_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
                             NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
        dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
        dataloader.startDataLoader()
        it = iter(dataloader)
    elif epoch == math.floor(NUM_EPOCHS * 0.7):
        del it
        del dataloader
        del dataset
        
        dataset = KWSDataset(BASETRAIN_CONF_HARD_PATH, FINETUNE_CONF_HARD_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
                             NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
        dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
        dataloader.startDataLoader()
        it = iter(dataloader)
    
    #
    # train
    #
    loss_train_epoch = np.zeros((len(model), ), dtype = 'float32')
    validbatchs = np.zeros((len(model), ), dtype = 'int')
    
    for bi in range(NUM_BATCHES_TRAIN):
        # prepare data
        feat, label = next(it)
        label = torch.reshape(label, (-1,))
        
        if usegpu:
            feat = feat.cuda()
            label = label.cuda()
        
        for mi in range(len(model)):
            # apply model
            optimizer[mi].zero_grad()
            predict = model[mi](feat)
            
            # calculate loss
            loss = loss_fn[mi](torch.reshape(predict, (-1, NUM_CLASSES)), label)
            
            if not np.isnan(loss.item()):
                loss.backward()
                optimizer[mi].step()
                
                loss_train_epoch[mi] += loss.item()
                validbatchs[mi] += 1
            
            sys.stderr.write('Epoch: {}/{}, batch: {}/{}, model: {}, loss: {}\n'.format(
                epoch, NUM_EPOCHS, bi + 1, NUM_BATCHES_TRAIN, mi, loss.item()))
    
    #
    # validation
    #
    loss_val_epoch = np.zeros((len(model), ), dtype = 'float32')
    
    with torch.no_grad():
        for feat, label in data_val:
            if usegpu:
                feat = feat.cuda()
                label = label.cuda()
            
            for mi in range(len(model)):
                # apply model
                predict = model[mi](feat)
                
                # calculate loss
                loss = loss_fn[mi](torch.reshape(predict, (-1, NUM_CLASSES)), label)
                loss_val_epoch[mi] += loss.item()
    
    # average training loss in one epoch
    loss_train_epoch /= validbatchs
    loss_val_epoch /= NUM_BATCHES_VAL
    
    for mi in range(len(model)):
        sys.stderr.write('Epoch: {:04d}, model: {:02d}, loss_train: {}, loss_val: {}\n'.format(
            epoch, mi, loss_train_epoch[mi], loss_val_epoch[mi]))
        
        # check point
        torch.save(model[mi], '../checkpoint/checkpoint_{:04d}_model_{:02d}_loss_train_{}_loss_val_{}.pth'.format(
            epoch, mi, loss_train_epoch[mi], loss_val_epoch[mi]))
    
    # time spent per epoch
    epochtime = datetime.datetime.now() - epochtime
    sys.stderr.write('Epoch {:04d} time spent: {:.2f} hours\n'.format(epoch, epochtime.total_seconds() / 3600.0))

del it
del dataloader
del dataset

# total time spent
totaltime = datetime.datetime.now() - totaltime
sys.stderr.write('Total time spent: {:.2f} hours\n'.format(totaltime.total_seconds() / 3600.0))
