'''
Generate scp file for garbge data.
'''

import os
import sys
import math
import numpy as np
import re
import shutil


EXPAND_TIMES = 3
FILE_EXP_LIST = 'exp_list.txt'


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
    sys.stderr.write("Usage:\n");
    sys.stderr.write("GenerateSCP_garbge <src file> <basein> <baseout>\n")
    exit(-1)


with open(sys.argv[1], 'r', encoding = 'UTF-8') as fd:
    lines = fd.readlines()

srcsize = len(lines)    
expsize = srcsize * EXPAND_TIMES
lines = []

for i in range(1, 11):
    lines.append('utt_' + str(i))
    tmplines = listFiles(os.path.join(sys.argv[2], 'utt_' + str(i)), ['.wav'])
    lines.extend(tmplines)


count = 0
flist = open(os.path.join(sys.argv[3], FILE_EXP_LIST), 'w', encoding = 'UTF-8')
prefix = 'utt_1'
for ts in lines:
    ts = ts.strip()
    
    # get prefix
    if ts.startswith('utt_'):
        prefix = ts
        continue
    
    # generate key
    # repeat = (count // 2) // srcsize + 1
    repeat = count // srcsize + 1
    
    _, sts = os.path.split(ts)
    sts, _ = os.path.splitext(sts)
    filename = sts + '_x' + str(repeat)
    key = prefix + '_' + filename
    
    # write file list
    pathdest = os.path.join(sys.argv[3], key + '.wav')
    flist.write(key + ' ' + os.path.abspath(pathdest) + '\n')
    
    # copy file
    pathsrc = os.path.join(sys.argv[2], prefix + '/' + sts + '.wav')
    shutil.copyfile(pathsrc, pathdest)
    
    count += 1
    # if count // 2 >= expsize:
    if count >= expsize:
        break

flist.flush()
flist.close()
