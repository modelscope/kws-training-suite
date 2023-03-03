'''
Rename files randomly to replace strange file name.
'''

import os
import sys
import tempfile
import re
import traceback
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


if len(sys.argv) != 2:
    sys.stderr.write('Rename files randomly to replace strange file name.\n');
    sys.stderr.write('Usage:\n');
    sys.stderr.write('RandRename <basein>\n')
    exit(-1)

basein = sys.argv[1]

for wavin in listFiles(basein, ['.wav']):
    tmpfd, tmppath = tempfile.mkstemp(prefix = os.path.split(basein)[1] + '_', suffix = '.wav', dir = basein)
    os.close(tmpfd)
    os.remove(tmppath)
    
    shutil.move(wavin, tmppath)
    
    namein = os.path.split(wavin)[1]
    nameout = os.path.split(tmppath)[1]
    print('DONE: ' + namein + ' -> ' + nameout)
