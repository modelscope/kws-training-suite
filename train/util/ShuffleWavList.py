'''
Shuffle wave file lists according to a specified ratio.
'''

import os
import sys
import random
import math


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

if len(sys.argv) <= 1:
    sys.stderr.write('Shuffle wave file list according to specified ratio.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('ShuffleWavList.py <list1 ratio1> [list2 ratio2] ...\n')
    

numlists= (len(sys.argv) - 1) // 2
totallist = []

for i in range(numlists):
    path = sys.argv[1 + i * 2]
    ratio = float(sys.argv[1 + i * 2 + 1])
    
    if os.path.isdir(path):
        flist = listFiles(path, ['wav'])
    else:
        with open(path, 'r', encoding = 'UTF-8') as fd:
            flist = fd.readlines()
    
    random.shuffle(flist)
    totallist.extend(flist[:math.floor(len(flist) * ratio)])

random.shuffle(totallist)
for f in totallist:
    print(os.path.abspath(f.strip()))
