'''
Move files to the same directory, add random name suffix to prevent duplicate name.

Copyright: 2022-10-27 yueyue.nyy
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


if len(sys.argv) != 3:
    sys.stderr.write('Move files to the same directory, add random name suffix to prevent duplicate name.\n');
    sys.stderr.write('Usage:\n');
    sys.stderr.write('MVBatch <basein> <baseout>\n')
    exit(-1)

basein = sys.argv[1]
baseout = sys.argv[2]

if os.path.isfile(basein):
    with open(basein, 'r', encoding = 'UTF-8') as fd:
        files = fd.readlines()
else:
    files = listFiles(basein, ['.wav'])

for wavin in files:
    wavin = wavin.strip()
    name = os.path.split(wavin)[1]
    name = os.path.splitext(name)[0]
    name = name.replace('.wav', '')
    name = name.replace('.WAV', '')
    name = name.replace(' ', '_')
    
    fout = os.path.join(baseout, name + '.wav')
    if os.path.exists(fout):
        tmpfd, fout = tempfile.mkstemp(prefix = name + '_', suffix = '.wav', dir = baseout)
        os.close(tmpfd)
    
    shutil.move(wavin, fout)
    
    print('DONE: ' + wavin + ' -> ' + fout)
