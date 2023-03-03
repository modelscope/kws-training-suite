'''
Generate scp file and annotation file from source.
'''

import os
import sys
import math
import numpy as np
import re
import shutil


EXPAND_TIMES = 3
FILE_ANNOT = 'annot.txt'
FILE_EXP_LIST = 'exp_list.txt'
FILE_EXP_ANNOT = 'exp_annot.txt'


if len(sys.argv) != 4:
    sys.stderr.write("Usage:\n");
    sys.stderr.write("GenerateSCP <src file> <basein> <baseout>\n")
    exit(-1)


with open(sys.argv[1], 'r', encoding = 'UTF-8') as fd:
    lines = fd.readlines()

srcsize = len(lines)    
expsize = srcsize * EXPAND_TIMES
lines = []


with open(os.path.join(sys.argv[2], FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
    lines = fd.readlines()

count = 0
flist = open(os.path.join(sys.argv[3], FILE_EXP_LIST), 'w', encoding = 'UTF-8')
fannot = open(os.path.join(sys.argv[3], FILE_EXP_ANNOT), 'w', encoding = 'UTF-8')
for ts in lines:
    # generate key
    repeat = (count // 2) // srcsize + 1
    
    ts = ts.strip()
    sts = ts.split()
    key = sts[0] + '_x' + str(repeat)
    
    # write annotation
    fannot.write(key + ' ')
    for i in range(1, len(sts)):
        if i < len(sts) - 1:
            fannot.write(sts[i] + ' ')
        else:
            fannot.write(sts[i])
    fannot.write('\n')
    
    # write file list
    pathdest = os.path.join(sys.argv[3], key + '.wav')
    flist.write(key + ' ' + os.path.abspath(pathdest) + '\n')
    
    # copy file
    pathsrc = os.path.join(sys.argv[2], sts[0] + '.wav')
    shutil.copyfile(pathsrc, pathdest)
    
    count += 1
    if count // 2 >= expsize:
        break

flist.flush()
flist.close()
fannot.flush()
fannot.close()
