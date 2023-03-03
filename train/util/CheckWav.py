'''
Check wav file format.
'''

import sys
import os
import math
import traceback
from librosa import get_duration
from librosa import get_samplerate
import shutil


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
    sys.stderr.write('Check wav file format.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write("CheckWav <listin or basein> [duration (second)]\n")
    exit(-1)


pathin = sys.argv[1]

lines = []
if os.path.isdir(pathin):
    lines = listFiles(pathin, ['.wav'])
else:
    with open(pathin, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()

duration = None
if len(sys.argv) > 2:
    duration = float(sys.argv[2])


for ts in lines:
    ts = ts.strip()
    
    try:
        if duration is not None:
            len = get_duration(filename = ts)
            if len < duration:
                print(ts + ' ' + str(len))
        
        fs = get_samplerate(ts)
        if fs == 16000:
            name = os.path.split(ts)[1]
            shutil.move(ts, os.path.join('/home/yueyue.nyy/data2/dataset/2022-01-19_King-AVT/16k', name))
        elif fs == 32000:
            name = os.path.split(ts)[1]
            shutil.move(ts, os.path.join('/home/yueyue.nyy/data2/dataset/2022-01-19_King-AVT/32k', name))
        elif fs == 48000:
            name = os.path.split(ts)[1]
            shutil.move(ts, os.path.join('/home/yueyue.nyy/data2/dataset/2022-01-19_King-AVT/48k', name))
        
    except:
        traceback.print_exc()
        print('FAILED: ' + ts)
