'''
Change keyword label.
'''

import os
import sys
import math
from scipy.io import wavfile
import numpy as np


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
    sys.stderr.write('Change keyword label.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('ChangeLabel.py <basein> <baseout> <old0 new0> [old1 new1...]\n')

basein = sys.argv[1]
baseout = sys.argv[2]

# get labels
numlabels = (len(sys.argv) - 3) // 2
oldlabel = []
newlabel = []
for i in range(numlabels):
    lb = float(sys.argv[2 * i + 3]) * 32768.0
    if lb >= 32768.0:
        lb = 32767.0
    oldlabel.append(int(lb))
    
    lb = float(sys.argv[2 * i + 4]) * 32768.0
    if lb >= 32768.0:
        lb = 32767.0
    newlabel.append(int(lb))
    
    
# change label
wavlist = listFiles(basein, ['wav'])

for fin in wavlist:
    fs, data = wavfile.read(fin)
    data2 = data.copy()
    
    for i in range(numlabels):
        data2[data[:, 1] == oldlabel[i], 1] = newlabel[i]

    _, name = os.path.split(fin)
    fout = os.path.join(baseout, name)
    wavfile.write(fout, fs, data2)
