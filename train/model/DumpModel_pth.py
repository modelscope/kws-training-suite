'''
Dump top k pytorch model to file.
'''

import numpy as np
import math
import os
import sys
import re
import shutil


def listFilesRec(path, suffix, filelist):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].replace('.', '')
        if ext in suffix:
            filelist.append(path)
    else:
        for f in os.listdir(path):
            listFilesRec(os.path.join(path, f).replace('\\', '/'), suffix, filelist)
    
    return filelist


def listFiles(base, suffix):        
    suffix2 = []
    for ext in suffix:
        suffix2.append(ext.replace('.', ''))
    
    filelist = []
    filelist = listFilesRec(base, suffix2, filelist)
    return filelist


'''
load model paths
basein:                 checkpoint base
'''
def loadModelPaths(basein):
    paths = []
    for f in listFiles(basein, ['.pth']):
        m = re.match('.+_model_(\d+)_.+', f)
        if m:
            idx = int(m.group(1))
            
            if len(paths) < idx + 1:
                for i in range(len(paths), idx + 1):
                    paths.append([])
            
            paths[idx].append(f)
    
    return paths


'''
select top k models according to loss_val
pathl:                  path list
k:                      top k
'''
def topKModels(pathl, k):
    lossval = [0.0] * len(pathl)
    
    for i in range(len(pathl)):
        m = re.match('.+loss_val_(.+)\.pth', pathl[i])
        if m:
            lossval[i] = float(m.group(1))
    
    idxl = np.argsort(np.array(lossval))
    
    topkl = []
    for i in range(min(k, len(pathl))):
        topkl.append(pathl[idxl[i]])
    
    return topkl


if len(sys.argv) != 4:
    sys.stderr.write("Usage:\n");
    sys.stderr.write("DumpModel <checkpoint base> <baseout> <K>\n")
    exit(-1)

basein = sys.argv[1]
baseout = sys.argv[2]
topk = int(sys.argv[3])


for pathl in loadModelPaths(basein):
    topkl = topKModels(pathl, topk)
    
    for tidx in range(len(topkl)):
        _, name = os.path.split(topkl[tidx])
        name, _ = os.path.splitext(name)         
        pathout = os.path.join(baseout, 'top_' + ('%02d_' % (tidx + 1)) + name + '.pth')
        
        # cmd = 'python ./model/PrintModel.py ' + topkl[tidx] + ' >' + pathout
        # print(cmd)
        # os.system(cmd)
        shutil.copyfile(topkl[tidx], pathout)
        
        print('DONE: ' + pathout)

