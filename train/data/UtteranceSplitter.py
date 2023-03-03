'''
Used to split utterances from long audio according to its label.

Copyright: 2022-10-09 yueyue.nyy
'''

import math
import os
import sys
import random
import numpy as np
import torch
from scipy.io import wavfile


# default sample rate
DEFAULT_FS = 16000
# data block size (s)
BLOCK_SIZE = 0.04
# minimal neg utterance length (s)
MIN_NEG_LEN = 1.0
# gain in the audio label
AUDIO_LABEL_GAIN = 100.0


'''
find utterance boundary according to label
label:                  label of the training data
return:                 [[offset, offset + length]]
'''
def detectBoundary(label):
    boundaryl = []
    
    pl = 0
    idx1 = 0
    idx2 = 0
    for i in range(label.shape[0]):
        l = label[i]
        if pl == 0 and l != 0:
            idx1 = i
        elif pl != 0 and l == 0:
            idx2 = i
            boundaryl.append([idx1, idx2])
        
        pl = l
    
    # the last one
    if l != 0:
        boundaryl.append([idx1, label.shape[0]])
    
    return boundaryl


'''
convert audio label to torch label
alabel:                 audio label [0, 1]
return:                 downsampled int torch label
'''
def audio2TorchLabel(alabel):
    flabel = np.round(AUDIO_LABEL_GAIN * alabel.astype('float32') / 32768.0)
    blocksize = int(DEFAULT_FS * BLOCK_SIZE)
    tlabel = np.zeros((flabel.shape[0] // blocksize, ), dtype = 'int32')
    
    for tau in range(tlabel.shape[0]):
        l = flabel[tau * blocksize]
        # remove vad label
        if l >= AUDIO_LABEL_GAIN:
            l = 0.0
        
        tlabel[tau] = int(l)
    
    return tlabel


class UtteranceSplitter:
    
    '''
    minrelax:           min utterance boundary relax (s)
    maxrelax:           max utterance boundary relax (s)
    '''    
    def __init__(self, minrelax = 1.0, maxrelax = 3.0):
        self.minrelax = int(minrelax / BLOCK_SIZE)
        self.maxrelax = int(maxrelax / BLOCK_SIZE)
        self.minneglen = int(MIN_NEG_LEN / BLOCK_SIZE)
    
    
    '''
    split boundaries for positive utterances
    boundaryl:          boundary list [[offset, offset + length]]
    labellen:           label length
    return:             boundaries for positive utterances [[offset, offset + length]]
    '''
    def splitPosBoundary(self, boundaryl, labellen):        
        posbl = []
        
        for idx, bl in enumerate(boundaryl):
            if idx == 0:
                lidx1 = max(bl[0] - self.maxrelax, 0)
                lidx2 = max(bl[0] - self.minrelax, 0)
            else:
                pbl = boundaryl[idx - 1]
                lidx1 = max(bl[0] - self.maxrelax, pbl[1])
                lidx2 = max(bl[0] - self.minrelax, pbl[1])
            
            if idx == len(boundaryl) - 1:
                ridxin1 = min(bl[1] - 1 + self.minrelax, labellen - 1)
                ridxin2 = min(bl[1] - 1 + self.maxrelax, labellen - 1)
            else:
                nbl = boundaryl[idx + 1]
                ridxin1 = min(bl[1] - 1 + self.minrelax, nbl[0] - 1)
                ridxin2 = min(bl[1] - 1 + self.maxrelax, nbl[0] - 1)
            
            lidx = random.randint(lidx1, lidx2)
            ridxin = random.randint(ridxin1, ridxin2)
            # relax 1 at both sides
            posbl.append([lidx + 1, ridxin])
        
        return posbl
    
    
    '''
    split boundaries for negative utterances
    boundaryl:          boundary list [[offset, offset + length]]
    labellen:           label length
    maxneglen:          max neg utterance length
    return:             boundaries for negative utterances [[offset, offset + length]]
    '''
    def splitNegBoundary(self, boundaryl, labellen, maxneglen):
        tmpl = []
        
        # pure noise
        if len(boundaryl) == 0:
            tmpl.append([0, labellen])
        else:
            for i in range(0, len(boundaryl) - 1):
                bl = boundaryl[i]
                nbl = boundaryl[i + 1]
                # +1 skip pos utterance tail
                idx1 = bl[1] + 1
                # -1 to skip pos utterance head
                idxout2 = nbl[0] - 1
                
                # skip too short segments
                if idxout2 - idx1 < self.minneglen:
                    continue
                
                tmpl.append([idx1, idxout2])
        
        # split long noise
        negbl = []
        while len(tmpl) > 0:
            bl = tmpl.pop(0)
            
            neglen = bl[1] - bl[0]
            if neglen > maxneglen:
                half = neglen // 2
                
                if half >= self.minneglen:
                    tmpl.append([bl[0], bl[0] + half])
                
                if bl[1] - (bl[0] + half) >= self.minneglen:
                    tmpl.append([bl[0] + half, bl[1]])
            else:
                negbl.append(bl)
        
        return negbl
    
    
    '''
    split audio utterances
    audioin:            input audio path
    baseout:            output directory
    '''
    def splitAudioUtterances(self, audioin, baseout):
        fs, data = wavfile.read(audioin)
        
        # convert audio label to torch label
        tlabel = audio2TorchLabel(data[:, -1])
        # get utterance boundaries
        boundaryl = detectBoundary(tlabel)
        
        # split positive utterances
        posbl = self.splitPosBoundary(boundaryl, tlabel.shape[0])
        blocksize = int(DEFAULT_FS * BLOCK_SIZE)
        
        for idx, b in enumerate(posbl):
            idx1 = b[0] * blocksize
            if idx1 < 0:
                idx1 = 0
            elif idx1 > data.shape[0]:
                idx1 = data.shape[0]
            
            idxout2 = b[1] * blocksize
            if idxout2 < 0:
                idxout2 = 0
            elif idxout2 > data.shape[0]:
                idxout2 = data.shape[0]
            
            fout = os.path.join(baseout, 'pos_{:0>4d}.wav'.format(idx))
            wavfile.write(fout, fs, data[idx1:idxout2, :])
        
        # split negative utterance
        negbl = self.splitNegBoundary(
            boundaryl, tlabel.shape[0], int(10.0 / BLOCK_SIZE))
        
        for idx, b in enumerate(negbl):
            idx1 = b[0] * blocksize
            if idx1 < 0:
                idx1 = 0
            elif idx1 > data.shape[0]:
                idx1 = data.shape[0]
            
            idxout2 = b[1] * blocksize
            if idxout2 < 0:
                idxout2 = 0
            elif idxout2 > data.shape[0]:
                idxout2 = data.shape[0]
            
            fout = os.path.join(baseout, 'neg_{:0>4d}.wav'.format(idx))
            wavfile.write(fout, fs, data[idx1:idxout2, :])
    
    
    '''
    split long feat into utterances
    feat:               feature, [batch, time, channel, feat]
    label:              corresponding label, [batch, time, 1]
    return:             uttfeat, [batch, time, channel, feat]
                        uttlen, real utterance length, batch
                        uttlabel, [batch, time, 1]
    '''
    def splitFeatUtterances(self, feat, label):
        allposbl = []
        allnegbl = []
        uttbatchsize = 0
        maxlen = 0
        
        # detect pos and neg utterance boundaries
        for bi in range(label.shape[0]):
            # detect utterance boundary
            boundaryl = detectBoundary(label[bi, :, 0])
            
            # split positive utterances
            posbl = self.splitPosBoundary(boundaryl, label.shape[1])
            allposbl.append(posbl)
            uttbatchsize += len(posbl)
            
            for bl in posbl:
                tmplen = bl[1] - bl[0]
                if tmplen > maxlen:
                    maxlen = tmplen
            
            # split negative utterances
            negbl = self.splitNegBoundary(boundaryl, label.shape[1], maxlen)
            allnegbl.append(negbl)
            uttbatchsize += len(negbl)
            
            for bl in negbl:
                tmplen = bl[1] - bl[0]
                if tmplen > maxlen:
                    maxlen = tmplen
        
        # generate utterance batch
        uttfeat = torch.zeros(uttbatchsize, maxlen, feat.shape[2], feat.shape[3])
        uttlen = torch.zeros(uttbatchsize, dtype = torch.int64)
        uttlabel = torch.zeros(uttbatchsize, maxlen, label.shape[2], dtype = torch.int64)
        
        idx = 0
        for bi in range(feat.shape[0]):
            # split pos utterances and labels
            for bl in allposbl[bi]:
                tmplen = bl[1] - bl[0]
                
                uttfeat[idx, :tmplen, :, :] = feat[bi, bl[0]:bl[1], :, :]
                uttlen[idx] = tmplen
                uttlabel[idx, :tmplen, :] = label[bi, bl[0]:bl[1], :]
                
                idx += 1
            
            # split neg utterances and labels
            for bl in allnegbl[bi]:
                tmplen = bl[1] - bl[0]
                
                uttfeat[idx, :tmplen, :, :] = feat[bi, bl[0]:bl[1], :, :]
                uttlen[idx] = tmplen
                uttlabel[idx, :tmplen, :] = label[bi, bl[0]:bl[1], :]
                
                idx += 1
        
        return uttfeat, uttlen, uttlabel
    
    
    '''
    convert cross entropy label to CTC label
    label:              ce label, [batch, time, 1]
    nonkwlabel:         non keyword label used by ctc, used to replace filler
    return:             targets: target labels, [batch, max length]
                        lengths: [batch]
    '''
    def ce2ctcLabel(self, label, nonkwlabel):
        maxlen = 0
        lenl = []
        allctcl = []
        
        for bi in range(label.shape[0]):
            # convert to ctc label for each batch
            bl = label[bi, :, 0]            
            ctcl=[]
            pl = 0
            
            for l in bl:
                l = int(l)
                # convert ce filler to ctc non keyword
                if l == 0:
                    l = nonkwlabel
                
                if l == 0:
                    continue
                
                if pl != l:
                    ctcl.append(l)
                
                pl = l
            
            allctcl.append(ctcl)
            
            lenl.append(len(ctcl))
            if len(ctcl) > maxlen:
                maxlen = len(ctcl)
        
        # convert to tensor
        targets = torch.zeros(label.shape[0], maxlen, dtype = torch.int64)
        for bi in range(label.shape[0]):
            ctcl = torch.tensor(allctcl[bi], dtype = torch.int64)
            targets[bi, :ctcl.shape[0]] = ctcl[:]
        
        lengths = torch.tensor(lenl, dtype = torch.int64)
        
        return targets, lengths
    
    
# uttsplit = UtteranceSplitter()
# uttsplit.splitAudioUtterances('D:/feat.wav', 'D:/baseout')

# import torch
# from KWSDataset import KWSDataset
# from KWSDataLoader import KWSDataLoader
#
# BASETRAIN_CONF_VAL_PATH = '../conf/SoundPro/single_normal.conf'
# FINETUNE_CONF_VAL_PATH = '../conf/SoundPro/multi_normal.conf'
# FEAT_CAT_SIZE = 120
# NUM_CLASSES = 20
# BLOCK_DEC = 2
# BLOCK_CAT = 3
# NUM_WORKERS = 2
# BASETRAIN_RATIO = 0.5
# BATCH_SIZE = 2
# PREFETCH_FACTOR = 2
#
#
# dataset = KWSDataset(BASETRAIN_CONF_VAL_PATH, FINETUNE_CONF_VAL_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
#                      NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
# dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
# dataloader.startDataLoader()
# it = iter(dataloader)
#
# feat, label = next(it)
#
# uttsplit = UtteranceSplitter()
# uttfeat, uttlen, uttlabel = uttsplit.splitFeatUtterances(feat, label)
# print(uttfeat.shape)
# print(uttlen.shape)
# print(uttlabel.shape)
#
#
# ctclabel, ctcllen = uttsplit.ce2ctcLabel(uttlabel)
# for ubi in range(uttlabel.shape[0]):
#     print(uttlabel[ubi, :, 0])
#     print(ctclabel[ubi, :ctcllen[ubi]])
#     print()
#
# del it
# del dataloader
# del dataset
