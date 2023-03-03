'''
Used to prepare simulated data.
'''

import numpy as np
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
import torch


LIB_PATH = '../libSoundConnect.so'
NUM_WORKERS = 1
FBANK_SIZE = 40
LABEL_SIZE = 1
LABEL_GAIN = 100.0


'''
dataset for keyword spotting and vad
confpath:               configure file path
numchs:                 no. of front-end output channels
numclasses:             no. of nn output classes, 2 classes to generate vad label
blockdec:               block decimation
blockcat:               block concatenation
numworks:               no. of workers

'''
class KWSDataset(torch.utils.data.IterableDataset):
    def __init__(self, confpath, numchs, numclasses, blockdec, blockcat, numworkers):
        super().__init__()
        self.numchs = numchs
        self.numclasses = numclasses
        self.blockdec = blockdec
        self.blockcat = blockcat
        
        self.obj = ctypes.cdll.LoadLibrary(LIB_PATH)
        
        self.obj.TrainModePy_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.obj.TrainModePy_init(ctypes.c_char_p(confpath.encode('UTF-8')), numworkers)
        
        # get multi-channel feature vector size
        self.obj.TrainModePy_featSize.restype = ctypes.c_int
        self.featsize = self.obj.TrainModePy_featSize()
        
        # get label vector size
        self.obj.TrainModePy_labelSize.restype = ctypes.c_int
        self.labelsize = self.obj.TrainModePy_labelSize()
        
        # get minibatch size (time dimension)
        self.obj.TrainModePy_featBatchSize.restype = ctypes.c_int
        self.batchsize = self.obj.TrainModePy_featBatchSize()
        
        # print(self.featsize, self.labelsize, self.batchsize)
        
        self.obj.TrainModePy_processBatch.argtypes = [ctypes.c_int]
        # 2d array: time x (multi-channel feature + label)
        self.obj.TrainModePy_feat.argtypes = [ctypes.c_int]
        self.obj.TrainModePy_feat.restype = ndpointer(
            dtype = ctypes.c_float, 
            shape = (self.batchsize, self.featsize + self.labelsize))
    
    
    def __del__(self):
        self.obj.TrainModePy_free()
    
    
    def __iter__(self):
        return self
    
    
    '''
    return:             time x channel x feature, label
    '''
    def __next__(self):
        id = 0
        info = torch.utils.data.get_worker_info()
        if info is not None:
            id = info.id
        
        # get raw data
        self.obj.TrainModePy_processBatch(id)
        data = self.obj.TrainModePy_feat(id)
        
        # convert float label to int
        label = data[:, FBANK_SIZE * self.numchs:]
        
        if self.numclasses == 2:
            # generate vad label
            label[label > 0.0] = 1.0
        else:
            # generate kws label
            label = np.round(label * LABEL_GAIN)
            label[label > self.numclasses - 1] = 0.0
        
        # decimated size
        size1 = int(np.ceil(label.shape[0] / self.blockdec)) - self.blockcat + 1
        
        # label decimation
        label1 = np.zeros((size1, LABEL_SIZE), dtype = 'float32')
        for tau in range(size1):
            label1[tau, :] = label[(tau + self.blockcat // 2) * self.blockdec, :]
        
        # feature decimation and concatenation
        # time x channel x feature
        featall = np.zeros((size1, self.numchs, FBANK_SIZE * self.blockcat), dtype = 'float32')
        for n in range(self.numchs):
            feat = data[:, FBANK_SIZE * n:FBANK_SIZE * (n + 1)]
            
            for tau in range(size1):
                for i in range(self.blockcat):
                    featall[tau, n, FBANK_SIZE * i:FBANK_SIZE * (i + 1)] = \
                    feat[(tau + i) * self.blockdec, :]
        
        return torch.from_numpy(featall), torch.from_numpy(label1).long()


# NUM_CHANNELS = 3
# FBANK_SIZE = 2
# LABEL_SIZE = 1
# BLOCK_DECIMATION = 2
# BLOCK_CAT = 3
# size = 13
# 
# data = np.random.randn(size, NUM_CHANNELS * FBANK_SIZE)
# idx = 0
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         data[i, j] = idx
#         idx += 1
# 
# label = np.random.randn(size, LABEL_SIZE)
# 
# size1 = int(np.ceil(data.shape[0] / BLOCK_DECIMATION)) - BLOCK_CAT + 1
# 
# label1 = np.zeros((size1, LABEL_SIZE), dtype = 'float32')
# for tau in range(size1):
#     label1[tau, :] = label[(tau + BLOCK_CAT // 2) * BLOCK_DECIMATION, :]
#  
# featall = np.zeros((size1, NUM_CHANNELS, FBANK_SIZE * BLOCK_CAT), dtype = 'float32')
# for n in range(NUM_CHANNELS):
#     feat = data[:, FBANK_SIZE * n:FBANK_SIZE * (n + 1)]
#     
#     for tau in range(size1):
#         for i in range(BLOCK_CAT):
#             featall[tau, n, FBANK_SIZE * i:FBANK_SIZE * (i + 1)] = \
#             feat[(tau + i) * BLOCK_DECIMATION, :]

# dataset = KWSDataset('../conf/monovad_easy.conf', 1, 2, 1, 1, 1)
# d = dataset.__next__()
# print(d)
