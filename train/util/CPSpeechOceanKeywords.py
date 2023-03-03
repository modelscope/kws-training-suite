'''
Copy keywords to individual directories according to speech ocean annotation format.

Copyright: 2022-05-05 yueyue.nyy
'''

import os
import sys
import math
import re
import shutil
import tempfile
import traceback


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
return:                 {id, kw}
'''
def loadAnnot(fannot):
    retl = []
    
    with open(fannot, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
    
    for i, ts in enumerate(lines):
        if i == 0:
            continue
        
        sts = ts.strip().split()
        retl.append({'id': sts[0], 'kw': sts[1]})
    
    return retl


'''
find wave directory by annotation file
fannot:                 annotation file
return:                 wave directory
'''
def findWavDirByAnnot(fannot):
    wavdir = os.path.split(fannot)[0]
    wavdir = wavdir.replace('script', 'wave')
    return wavdir


'''
copy keywords in an annotation file
fannot:                 annotation file
baseout:                output directory
'''
def cpKeywords(fannot, baseout):
    dirin = findWavDirByAnnot(fannot)
    
    for entry in loadAnnot(fannot):
        dirout = os.path.join(baseout, entry['kw'])
        if not os.path.exists(dirout):
            os.mkdir(dirout)
        
        suffix = '.wav'
        fin = os.path.join(dirin, entry['id'] + suffix)
        if not os.path.exists(fin):
            suffix = '.pcm'
            fin = os.path.join(dirin, entry['id'] + suffix)
        
        fout = os.path.join(dirout, entry['id'] + suffix)
        # avoid duplicate name
        if os.path.exists(fout):
            tmpfd, tmppath = tempfile.mkstemp(prefix = entry['id'] + '_', suffix = suffix, dir = dirout)
            os.close(tmpfd)
            os.remove(tmppath)
            fout = tmppath
        
        try:
            shutil.copyfile(fin, fout)
            print('DONE: ' + fin + ' -> ' + fout)
        except:
            traceback.print_exc()
            print('FAILED: ' + fin)
    

if len(sys.argv) != 3:
    sys.stderr.write('Copy keywords to individual directories according to speech ocean annotation format.\n');
    sys.stderr.write('Usage:\n');
    sys.stderr.write('CPSpeechOceanKeywords <basein> <baseout>\n')
    exit(-1)

basein = sys.argv[1]
baseout = sys.argv[2]
baseann = os.path.join(basein, 'script')

for fannot in listFiles(baseann, ['.txt', '.TXT']):
    cpKeywords(fannot, baseout)
