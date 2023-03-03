'''
Cut utterances according to json annotation.
'''

import os
import sys
import math
import numpy as np
import re
import json
from scipy.io import wavfile

# boundary relax (second)
RELAX_SEC = 0.5

annotin = 'D:/annotation_json/标注结果_20210810.txt'
basein = 'D:/11111/20210810'
baseout = 'D:/11111/utterance'

outmap = {'打开灯光': '02_dkdg', 
          '关闭灯光': '03_gbdg', 
          '打开座垫': '04_dkzd', 
          '自我介绍': '05_zwjs'}


'''
cut utterances according to json annotation
jtext:                  json text
'''
def cutUtterances(jtext):
    jdata = json.loads(jtext)
    
    path = jdata['pcm']
    fname = os.path.split(path)[1]
    fname = os.path.splitext(fname)[0] + '.wav'
    label = jdata['文本']
    
    inpath = os.path.join(basein, fname)
    outpath = os.path.join(baseout, outmap[label])
    
    if not os.path.exists(inpath):
        return
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    fs, wavdata = wavfile.read(inpath)
    uttidx = 0
    uttfprefix = os.path.split(basein)[1] + '_' + os.path.splitext(fname)[0]
    
    for annot in jdata['sd_result']['items']:
        uttinterval = annot['meta']['segment_range']
        # print(uttinterval[0], uttinterval[1])
        
        for n in range(wavdata.shape[1]):
            uttfname = uttfprefix + '_utt_{:04d}'.format(uttidx) + '_ch_{:01d}'.format(n) + '.wav'
            print(uttfname)
            
            idx1 = int(fs * (uttinterval[0] - RELAX_SEC))
            idx1 = max(idx1, 0)
            idx2 = int(fs * (uttinterval[1] + RELAX_SEC))
            idx2 = min(idx2, wavdata.shape[0])
            wavfile.write(os.path.join(outpath, uttfname), fs, wavdata[idx1:idx2, n])
        
        uttidx += 1


with open(annotin, 'r', encoding = 'UTF-8') as file:
    lines = file.readlines()

for l in lines:
    cutUtterances(l)
