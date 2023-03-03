'''
Partition utterance speed to fast, normal, slow, very slow, according to utterance length.

Copyright: 2022-07-26 yueyue.nyy
'''

import sys
import os
import math
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


if len(sys.argv) != 2:
    sys.stderr.write('Partition utterance speed to fast, normal, slow, very slow, according to utterance length.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('UttSpeedPartition <listin or basein>\n')
    exit(-1)


pathin = sys.argv[1]
name = os.path.split(pathin)[1]
name = os.path.splitext(name)[0]

fdfast = open(name + '_fast.txt', 'w', encoding = 'UTF-8')
fdnormal = open(name + '_normal.txt', 'w', encoding = 'UTF-8')
fdslow = open(name + '_slow.txt', 'w', encoding = 'UTF-8')
fdveryslow = open(name + '_veryslow.txt', 'w', encoding = 'UTF-8')


lines = []
if os.path.isdir(pathin):
    lines = listFiles(pathin, ['.wav'])
else:
    with open(pathin, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()

for ts in lines:
    ts = ts.strip()
    
    try:
        uttlen = get_duration(filename = ts)
        
        if uttlen < 0.55:
            fdfast.write(ts + '\n')
        elif uttlen >= 0.55 and uttlen < 1.4:
            fdnormal.write(ts + '\n')
        elif uttlen >= 1.4 and uttlen < 1.9:
            fdslow.write(ts + '\n')
        else:
            fdveryslow.write(ts + '\n')
    except:
        print('FAILED: ' + ts)

fdfast.flush()
fdnormal.flush()
fdslow.flush()
fdveryslow.flush()

fdfast.close()
fdnormal.close()
fdslow.close()
fdveryslow.close()
