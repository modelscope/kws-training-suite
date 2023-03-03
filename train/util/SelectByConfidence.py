'''
Select utterances by confidence.
'''

import os
import sys
import math
import re
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


if len(sys.argv) != 4:
    sys.stderr.write('Select utterances by confidence.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('SelectByConfidence <basein> <lower bound> <upper bound>\n')
    exit(-1)

basein = sys.argv[1]
lower = float(sys.argv[2])
upper = float(sys.argv[3])


for fin in listFiles(basein, ['.wav']):
    m = re.match('.+_confidence_(.+?)\.wav$', fin)
    if m is None:
        continue
    
    confidence = float(m.group(1))
    if confidence >= lower and confidence < upper:
        print(os.path.abspath(fin))
