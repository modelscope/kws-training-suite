'''
Build train_pos_list parameter for commands model training.

Copyright: 2022-05-16 yueyue.nyy
'''

import os
import sys
import math
import tempfile
import re
import traceback


'''
load keyword descriptions
fin:                    input file path
return:                 {Chinese_keyword: [id, kw, [descs]]}
'''
def loadKWSDesc(fin):
    with open(fin, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
    
    kwsdesc = {}
    
    isdesc = False
    for ts in lines:
        ts = ts.strip()
        if len(ts) <= 0:
            continue
        
        if ts.startswith('kws_decode_desc ='):
            isdesc = True
            continue
        
        if not isdesc:
            continue
        
        if ts.startswith('#'):
            sts = ts.split()
            id = sts[1]
            chskw = sts[2]
        else:
            sts = ts.split(',')
            kw = sts[0]
            desc = sts[1:]
            
            kwsdesc.update({chskw: [id, kw, desc]})
    
    return kwsdesc
 

'''
load keyword data source lists
basecmd:                data source directory
return:                 {chskws: listpath}
'''
def loadKWList(basecmd):
    listdict = {}
    
    for f in os.listdir(basecmd):
        if os.path.isdir(os.path.join(basecmd, f)):
            continue
        
        name = os.path.split(f)[1]
        name, suffix = os.path.splitext(name)
        if suffix != '.txt':
            continue
        
        listdict.update({name: os.path.abspath(os.path.join(basecmd, f))})
    
    return listdict


'''
print train_pos_list parameter
'''
def printTrainPosList(kwsdesc, dsdict):
    print('train_pos_list = ')
    for chskw in kwsdesc.keys():
        ts = '# '
        ts += kwsdesc[chskw][0] + ' ' + chskw + ' ' + kwsdesc[chskw][1]
        desc = kwsdesc[chskw][2]
        for u in desc:
            ts += ',' + u
        print(ts)
        
        labelmap = ''
        for i in range(len(desc)):
            labelmap += desc[i]
            if i < len(desc) - 1:
                labelmap += '_'
        
        path = dsdict.get(chskw)
        if path is not None:
            print(path + ',1.0,' + labelmap)
    

if len(sys.argv) != 3:
    sys.stderr.write('Build train_pos_list parameter for commands model training.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('BuildCMDTrainPosList <dict path> <base cmd>\n')
    exit(-1)

dictpath = sys.argv[1]
basecmd = sys.argv[2]

# load keyword descriptions
kwsdesc = loadKWSDesc(dictpath)
# load data source lists
dsdict = loadKWList(basecmd)
# print result
printTrainPosList(kwsdesc, dsdict)
