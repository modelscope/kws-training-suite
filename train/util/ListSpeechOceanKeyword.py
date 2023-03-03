'''
List speech ocean listed keyword.
'''

import os
import sys
import math
import numpy as np
import re


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


'''
load annotation from file
fannot:                 annotation file path
return:                 {id, kw, duration}
'''
def loadAnnot(fannot):
    retl = []
    
    with open(fannot, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
    
    for ts in lines:
        sts = ts.strip().split()
        retl.append({'id': sts[0], 'kw': sts[1]})
    
    return retl


'''
load wave files
basewav:                wave file base
return:                 {id, [files]}
'''
def loadWavs(basewav):
    wavdict = {}

    for fwav in listFiles(basewav, ['.wav']):
        id = os.path.split(fwav)[1]
        id = os.path.splitext(id)[0]
        
        fl = wavdict.get(id)
        if fl is None:
            fl = []
            wavdict.update({id: fl})
        
        fl.append(fwav)
    
    return wavdict
        

if len(sys.argv) != 3:
    sys.stderr.write('List speech ocean listed keyword.\n');
    sys.stderr.write('Usage:\n');
    sys.stderr.write('ListSpeechOceanKeyword <base> <keyword>\n')
    exit(-1)

base = sys.argv[1]
kw = sys.argv[2]

baseann = os.path.join(base, 'script')
basewav = os.path.join(base, 'wave')
wavdict = loadWavs(basewav)

for fannot in listFiles(baseann, ['.txt', '.TXT']):
    for annot in loadAnnot(fannot):
        if annot['kw'] != kw:
            continue
        
        fl = wavdict.get(annot['id'])
        if fl is not None:
            for l in fl:
                print(os.path.abspath(l))
