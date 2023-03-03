'''
Used to generate ideal ratio mask.
'''

import numpy as np
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
import torch


LIB_PATH = '../libSoundConnect.so'


'''
dataset for keyword spotting and vad
confpath:               configure file path
isfbankmask:            true to generate fbank domain mask, false to generate frequency domain mask
numworks:               no. of workers
'''
class IRMDataset(torch.utils.data.IterableDataset):
    def __init__(self, confpath, isfbankmask, numworkers):
        super().__init__()
        self.isfbankmask = isfbankmask
        
        self.obj = ctypes.cdll.LoadLibrary(LIB_PATH)
        
        self.obj.TrainModePy_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
        self.obj.TrainModePy_init(
            ctypes.c_char_p(confpath.encode('UTF-8')), 
            ctypes.c_char_p(None), 
            numworkers, 
            1.0)
        
        # get multi-channel feature vector size
        self.obj.TrainModePy_featSize.argtypes = [ctypes.c_int]
        self.obj.TrainModePy_featSize.restype = ctypes.c_int
        self.featsize = self.obj.TrainModePy_featSize(0)
        
        # get label vector size
        self.obj.TrainModePy_labelSize.argtypes = [ctypes.c_int]
        self.obj.TrainModePy_labelSize.restype = ctypes.c_int
        self.labelsize = self.obj.TrainModePy_labelSize(0)
        
        # get minibatch size (time dimension)
        self.obj.TrainModePy_featBatchSize.argtypes = [ctypes.c_int]
        self.obj.TrainModePy_featBatchSize.restype = ctypes.c_int
        self.batchsize = self.obj.TrainModePy_featBatchSize(0)
        
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
    get label size
    '''
    def labelSize(self):
        if self.isfbankmask:
            return self.featsize
        else:
            return self.labelsize - self.featsize
    
    
    '''
    return:             time x feature, time x label
    '''
    def __next__(self):
        id = 0
        info = torch.utils.data.get_worker_info()
        if info is not None:
            id = info.id
        
        # get raw data
        self.obj.TrainModePy_processBatch(id)
        # data must be copied
        data = self.obj.TrainModePy_feat(id).copy()
        
        feat = data[:, :self.featsize]
        
        if self.isfbankmask:
            label = data[:, self.featsize:self.featsize * 2]
        else:
            label = data[:, self.featsize * 2:]
        
        return torch.from_numpy(feat), torch.from_numpy(label)


# dataset = IRMDataset('../ttt.conf', True, 1)
# feat, label = dataset.__next__()
