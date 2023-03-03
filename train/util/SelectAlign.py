'''
Select aligned keywords fragments.
'''

import os
import sys
import math
import numpy as np
from scipy.io import wavfile


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


if len(sys.argv) <= 3:
    sys.stderr.write('Usage:\n');
    sys.stderr.write('SelectAlign <basein> <baseout> <labels>\n')
    exit(-1)


basein = sys.argv[1]
baseout = sys.argv[2]

flabels = np.zeros((len(sys.argv) - 3, ), dtype = 'float32')
for i in range(flabels.shape[0]):
    flabels[i] = float(sys.argv[i + 3])


for fin in listFiles(basein, ['.wav']):    
    fs, data = wavfile.read(fin)
    chlabel = np.round(data[:, 1].astype('float32') / 32768.0 * 100.0) / 100.0
    
    data2 = np.zeros(data.shape, dtype = 'int16')
    idx = 0
    for t in range(data.shape[0]):
        for l in flabels:
            if l == chlabel[t]:
                data2[idx, :] = data[t, :]
                idx += 1
                break
    
    _, name = os.path.split(fin)
    fout = os.path.join(baseout, name)
    wavfile.write(fout, fs, data2[:idx, :])
