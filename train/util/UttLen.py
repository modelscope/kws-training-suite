'''
Print utterance length (second).

Copyright: 2022-07-26 yueyue.nyy
'''

import sys
import os
import math
import shutil
import numpy as np
from librosa import get_duration


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


if len(sys.argv) < 2:
    sys.stderr.write('Print utterance length (second).\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('UttLen <listin or basein> [--sort]\n')
    exit(-1)


pathin = sys.argv[1]
needsort = False
if len(sys.argv) > 2 and sys.argv[2] == '--sort':
    needsort = True

lines = []
if os.path.isdir(pathin):
    lines = listFiles(pathin, ['.wav'])
else:
    with open(pathin, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()

uttlenl = np.zeros((len(lines), ), dtype = 'float32')
for idx, ts in enumerate(lines):
    ts = ts.strip()
    
    try:
        uttlen = get_duration(filename = ts)
        uttlenl[idx] = uttlen
        
        # if uttlen > 2.5:
        #     base, name = os.path.split(ts)
        #     base = os.path.split(base)[1]
        #     fout = os.path.join('longutts_2.5', base)
        #     fout = os.path.join(fout, name)
        #     shutil.move(ts, fout)
        
    except:
        print('FAILED: ' + ts)

if needsort:
    idxl = np.argsort(-uttlenl)
    
    for idx in idxl:
        print(lines[idx] + ' ' + str(uttlenl[idx]))
else:
    for idx, ts in enumerate(lines):
        print(ts + ' ' + str(uttlenl[idx]))
