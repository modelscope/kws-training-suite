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

lines.append('utt_1')
with open(os.path.join(sys.argv[2], 'utt_1/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
    tmplines = fd.readlines()
lines.extend(tmplines)

lines.append('utt_2')
with open(os.path.join(sys.argv[2], 'utt_2/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
    tmplines = fd.readlines()
lines.extend(tmplines)

lines.append('utt_3')
with open(os.path.join(sys.argv[2], 'utt_3/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
    tmplines = fd.readlines()
lines.extend(tmplines)

# lines.append('utt_4')
# with open(os.path.join(sys.argv[2], 'utt_4/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
#     tmplines = fd.readlines()
# lines.extend(tmplines)
# 
# lines.append('utt_5')
# with open(os.path.join(sys.argv[2], 'utt_5/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
#     tmplines = fd.readlines()
# lines.extend(tmplines)
# 
# lines.append('utt_6')
# with open(os.path.join(sys.argv[2], 'utt_6/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
#     tmplines = fd.readlines()
# lines.extend(tmplines)
# 
# lines.append('utt_7')
# with open(os.path.join(sys.argv[2], 'utt_7/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
#     tmplines = fd.readlines()
# lines.extend(tmplines)
# 
# lines.append('utt_8')
# with open(os.path.join(sys.argv[2], 'utt_8/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
#     tmplines = fd.readlines()
# lines.extend(tmplines)
# 
# lines.append('utt_9')
# with open(os.path.join(sys.argv[2], 'utt_9/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
#     tmplines = fd.readlines()
# lines.extend(tmplines)
# 
# lines.append('utt_10')
# with open(os.path.join(sys.argv[2], 'utt_10/' + FILE_ANNOT), 'r', encoding = 'UTF-8') as fd:
#     tmplines = fd.readlines()
# lines.extend(tmplines)

count = 0
flist = open(os.path.join(sys.argv[3], FILE_EXP_LIST), 'w', encoding = 'UTF-8')
fannot = open(os.path.join(sys.argv[3], FILE_EXP_ANNOT), 'w', encoding = 'UTF-8')
prefix = 'utt_1'
for ts in lines:
    ts = ts.strip()
    
    # get prefix
    if ts.startswith('utt_'):
        prefix = ts
        continue
       
    # generate key
    repeat = (count // 2) // srcsize + 1
    
    sts = ts.split()
    filename = sts[0] + '_x' + str(repeat)
    key = prefix + '_' + filename
    
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
    pathsrc = os.path.join(sys.argv[2], prefix + '/' + sts[0] + '.wav')
    shutil.copyfile(pathsrc, pathdest)
    
    count += 1
    if count // 2 >= expsize:
        break

flist.flush()
flist.close()
fannot.flush()
fannot.close()
