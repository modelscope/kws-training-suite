'''
Split multi-channel pcm utterances into multiple single channel waves.

Copyright: 2022-05-06 yueyue.nyy
'''

import os
import sys
import math
import re
import shutil
import tempfile
import traceback
import numpy as np
from scipy.io import wavfile


FS = 16000
NUM_CHS = 3
VALID_CHS = [0, 1]


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
split one pcm file
fin:                    input file path
'''
def splitPCM(fin):
    data = np.fromfile(fin, 'int16')
    data = np.reshape(data, (data.shape[0] // NUM_CHS, NUM_CHS))
    
    base, name = os.path.split(fin)
    name = os.path.splitext(name)[0]
    
    for n in VALID_CHS:
        fout = os.path.join(base, name + '_ch{:d}.wav'.format(n))
        wavfile.write(fout, FS, data[:, n])
    
    os.remove(fin)


if len(sys.argv) != 2:
    sys.stderr.write('Split multi-channel pcm utterances into multiple single channel waves.\n');
    sys.stderr.write('Usage:\n');
    sys.stderr.write('SplitPCM <basein>\n')
    exit(-1)


basein = sys.argv[1]
for fin in listFiles(basein, ['.pcm', '.PCM']):
    try:
        splitPCM(fin)
        print('DONE: ' + fin)
    except:
        traceback.print_exc()
        print('FAILED: ' + fin)
