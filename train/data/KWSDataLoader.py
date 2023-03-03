'''
Load kws data with multi-thread.
'''

import sys
import threading
from threading import Lock
import queue
import numpy as np
import torch
from KWSDataset import KWSDataset


class Worker(threading.Thread):
    
    '''
    id:                 worker id
    lock:               used to synchronize the thread
    dataset:            the dataset
    pool:               queue as the global data buffer
    '''
    def __init__(self, id, lock, dataset, pool):
        threading.Thread.__init__(self)
        
        self.id = id
        self.lock = lock
        self.dataset = dataset
        self.pool = pool
        self.isrun = True
    
    
    def run(self):
        while self.isrun:
            # data simulation
            if self.isrun:
                self.dataset.processBatch(self.id)
            
            # get simulated minibatch
            if self.isrun:
                self.lock.acquire()
                data = self.dataset.getBatch(self.id)
                self.lock.release()
            
            # put data into buffer
            if self.isrun:
                self.pool.put(data)
        
        sys.stderr.write('KWSDataLoader: Worker {:d} stopped.\n'.format(self.id))
    
    
    '''
    stop the worker thread
    '''
    def stopWorker(self):
        self.isrun = False
    

class KWSDataLoader:
    
    '''
    dataset:            the dataset reference
    batchsize:          data batch size
    numworkers:         no. of workers
    prefetch:           prefetch factor
    '''    
    def __init__(self, dataset, batchsize, numworkers, prefetch = 10):
        self.dataset = dataset
        self.batchsize = batchsize
        self.datamap = {}
        self.isrun = True
        self.lock = Lock()
        
        # data queue
        self.pool = queue.Queue(batchsize * prefetch)
        
        # initialize workers
        self.workerlist = []
        for id in range(numworkers):
            w = Worker(id, self.lock, dataset, self.pool)
            self.workerlist.append(w)
    
    
    def __del__(self):
        self.stopDataLoader()
    
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        while self.isrun:
            # get data from common data pool
            data = self.pool.get()
            self.pool.task_done()
            
            # group minibatches with the same shape
            key = str(data[0].shape)
            
            batchl = self.datamap.get(key)
            if batchl is None:
                batchl = []
                self.datamap.update({key: batchl})
            
            batchl.append(data)
            
            # a full data batch collected
            if len(batchl) >= self.batchsize:
                featbatch = []
                labelbatch = []
                
                for feat, label in batchl:
                    featbatch.append(feat)
                    labelbatch.append(label)
                
                batchl.clear()
                
                feattensor = torch.stack(featbatch, dim = 0)
                labeltensor = torch.stack(labelbatch, dim = 0)
                
                sys.stderr.write('KWSDataLoader: Batch shape: ' + str(feattensor.shape) + '\n')
                
                return feattensor, labeltensor
        
        return None, None
    
    
    '''
    start multi-thread data loader
    '''
    def startDataLoader(self):
        for w in self.workerlist:
            w.start()
    
    
    '''
    stop data loader
    '''
    def stopDataLoader(self):
        self.isrun = False
        
        for w in self.workerlist:
            w.stopWorker()
        
        while not self.pool.empty():
            self.pool.get_nowait()
        
        # wait workers terminated
        for w in self.workerlist:
            while not self.pool.empty():
                self.pool.get_nowait()
            
            w.join()
        
    
# BASETRAIN_CONF_VAL_PATH = '../conf/basetrain_normal.conf'
# BASETRAIN_CONF_EASY_PATH = '../conf/basetrain_easy.conf'
# BASETRAIN_CONF_NORMAL_PATH = '../conf/basetrain_normal.conf'
# BASETRAIN_CONF_HARD_PATH = '../conf/basetrain_hard.conf'
# FINETUNE_CONF_VAL_PATH = '../conf/sweeper_normal.conf'
# FINETUNE_CONF_EASY_PATH = '../conf/sweeper_easy.conf'
# FINETUNE_CONF_NORMAL_PATH = '../conf/sweeper_normal.conf'
# FINETUNE_CONF_HARD_PATH = '../conf/sweeper_hard.conf'
# FEAT_CAT_SIZE = 120
# NUM_CLASSES = 5
# BLOCK_DEC = 2
# BLOCK_CAT = 3
# NUM_WORKERS = 5
# BASETRAIN_RATIO = 0.8
# BATCH_SIZE = 2
# PREFETCH_FACTOR = 2
# 
# 
# dataset = KWSDataset(BASETRAIN_CONF_VAL_PATH, FINETUNE_CONF_VAL_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
#                      NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
# dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
# it = iter(dataloader)
# 
# for bi in range(10):
#     feat, label = next(it)
#     print(feat.shape)
# 
# del it
# sys.stderr.write('delete val it.\n')
# del dataloader
# sys.stderr.write('delete val loader.\n')
# del dataset
# sys.stderr.write('delete val set.\n')
# 
# 
# dataset = KWSDataset(BASETRAIN_CONF_EASY_PATH, FINETUNE_CONF_EASY_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
#                      NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
# sys.stderr.write('init easy set.\n')
#  
# dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
# sys.stderr.write('init easy loader.\n')
# it = iter(dataloader)
# sys.stderr.write('init easy it.\n')
# 
# del it
# sys.stderr.write('delete easy it.\n')
# del dataloader
# sys.stderr.write('delete easy loader.\n')
# del dataset
# sys.stderr.write('delete easy set.\n')
# 
# 
# dataset = KWSDataset(BASETRAIN_CONF_EASY_PATH, FINETUNE_CONF_EASY_PATH, NUM_WORKERS, BASETRAIN_RATIO, 
#                      NUM_CLASSES, BLOCK_DEC, BLOCK_CAT)
# sys.stderr.write('init normal set.\n')
#  
# dataloader = KWSDataLoader(dataset, batchsize = BATCH_SIZE, numworkers = NUM_WORKERS, prefetch = PREFETCH_FACTOR)
# sys.stderr.write('init normal loader.\n')
# it = iter(dataloader)
# sys.stderr.write('init normal it.\n')
# 
# for bi in range(10):
#     feat, label = next(it)
#     print(feat.shape)
# 
# del it
# sys.stderr.write('delete normal it.\n')
# del dataloader
# sys.stderr.write('delete normal loader.\n')
# del dataset
# sys.stderr.write('delete normal set.\n')
