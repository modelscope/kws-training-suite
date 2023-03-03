'''
Append align info to utterances.
'''

import os
import sys
import math
import numpy as np
import re
from scipy.io import wavfile


if len(sys.argv) != 4:
    sys.stderr.write("Usage:\n");
    sys.stderr.write("AppendAlign <wave scp file> <align ark> <baseout>\n")
    exit(-1)


#
# load scp
#
with open(sys.argv[1], 'r', encoding = 'UTF-8') as fd:
    lines = fd.readlines()
    
scpmap = {}
for ts in lines:
    ts = ts.strip()
    sts = ts.split()
    scpmap.update({sts[0]:sts[1]})


#
# load align info
#
with open(sys.argv[2], 'r', encoding = 'UTF-8') as fd:
    lines = fd.readlines()

arkmap = {}
for ts in lines:
    ts = ts.strip()
    sts = ts.split()
    arkmap.update({sts[0]:sts[1:]})


#
# append align info
#
if not os.path.exists(sys.argv[3]):
    os.makedirs(sys.argv[3])

blocksize = 160
for key, path in scpmap.items():
    try:
        align = arkmap[key]
    except KeyError:
        continue
    
    fs, data = wavfile.read(path)
    data2 = np.zeros((data.shape[0], 2), dtype = 'int16')
    data2[:, 0] = data[:]
    
    for bi in range(len(align)):
        data2[blocksize * bi:blocksize * (bi + 1), 1] = int(align[bi]) * 32768 / 100
    
    _, fname = os.path.split(path)
    pathout = os.path.join(sys.argv[3], fname)
    wavfile.write(pathout, fs, data2)
    
    break
