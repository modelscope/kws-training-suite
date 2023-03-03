'''
Check wav length
'''

import sys
import os
import math
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


if len(sys.argv) != 3:
    sys.stderr.write("Usage:\n");
    sys.stderr.write("CheckWavLen <listin or basein> <duration (s)>\n")
    exit(-1)


pathin = sys.argv[1]
duration = float(sys.argv[2])


lines = []
if os.path.isdir(pathin):
    lines = listFiles(pathin, ['.wav'])
else:
    with open(pathin, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()


for ts in lines:
    ts = ts.strip()
    
    try:
        fs, data = wavfile.read(ts)
        len = float(data.shape[0]) / fs
        
        if len < duration:
            print(ts + ' ' + str(len))
            # cmd = 'rm -f ' + ts
            # os.system(cmd)
    except:
        print(ts)
        # cmd = 'rm -f ' + ts
        # os.system(cmd)
