'''
count data set length (hours)
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
    sys.stderr.write("Usage:\n");
    sys.stderr.write("DatasetLen <listin or basein>\n")
    exit(-1)


pathin = sys.argv[1]


lines = []
if os.path.isdir(pathin):
    lines = listFiles(pathin, ['.wav'])
else:
    with open(pathin, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()

totallen = 0.0
for ts in lines:
    ts = ts.strip()
    
    try:
        totallen += get_duration(filename = ts)
    except:
        print('FAILED: ' + ts)
    
print(str(totallen / 3600.0) + ' hours')
