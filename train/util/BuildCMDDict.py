'''
Build dictionary for commands model training.

Copyright: 2022-05-16 yueyue.nyy
'''

import os
import sys
import math
import tempfile
import numpy as np
import re
import traceback


'''
load commands list and corresponding modeling units
path:                   file path
return:                 {cmd: comments, unit list}
'''
def loadCMDList(path):
    with open(path, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
    
    kwdict = {}
    
    id = 0
    for ts in lines:
        ts = ts.strip()
        if len(ts) <= 0:
            continue
        if ts.startswith('#'):
            continue
        
        sts = ts.split()
        if len(sts) != 3:
            continue
        
        strid = sts[0]
        cmd = sts[1]
        units = sts[2];
        
        kw = str(id) + '_' + units
        comments = strid + ' ' + cmd
        uus = units.split('_')
        kwdict.update({kw: [comments, uus]})
        
        id += 1
    
    return kwdict


'''
build modeling units dictionary
kwdict:                 keyword dict
return:                 {unit: [classid, count]}
'''
def buildUnitDict(kwdict):
    tmpdict = {}
    
    for kw in kwdict.keys():
        units = kwdict[kw][1]
        
        for u in units:
            val = tmpdict.get(u)
            if val is None:
                tmpdict.update({u: 0})
            
            tmpdict[u] += 1
    
    # sort
    unitl = []
    for u in tmpdict.keys():
        unitl.append(u)
    
    unitl.sort()
    
    # build unit dict
    unitdict = {}
    for i, u in enumerate(unitl):
        unitdict.update({u: [i + 1, tmpdict[u]]})
    
    return unitdict


'''
print kws_decode_desc parameter
kwdict:                 commands dict
unitdict:               modeling units dict
'''
def printDecodeDesc(kwdict, unitdict):
    print('kws_decode_desc = ')
    
    for kw in kwdict.keys():
        # comments
        print('# ' + kwdict[kw][0])
        
        units = kwdict[kw][1]
        desc = ''
        for i in range(len(units)):
            u = units[i]
            desc += str(unitdict[u][0])
            if i < len(units) - 1:
                desc += ','
        
        print(kw + ',' + desc)


if len(sys.argv) != 2:
    sys.stderr.write('Build dictionary for commands model training.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('BuildCMDDict <path>\n')
    sys.stderr.write('Input format example:\n')
    sys.stderr.write('0 天猫精灵 tian_mao_jin_lin\n')
    sys.stderr.write('1 你好天猫 ni_hao_tian_mao\n')
    sys.stderr.write('...\n')
    exit(-1)

fin = sys.argv[1]

# load origin cmds
kwdict = loadCMDList(fin)
# build modeling units dictionary
unitdict = buildUnitDict(kwdict)

# print unit dict
print('#')
print('# dictionary:')
print('#')
print('# Label Unit Count')
for u in unitdict.keys():
    print('# ' + str(unitdict[u][0]) + ' ' + u + ' ' + str(unitdict[u][1]))
print('#')
print()

# print results
printDecodeDesc(kwdict, unitdict)
