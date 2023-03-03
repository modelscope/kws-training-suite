'''
Extract wave files from ark.

Copyright: 2023-01-16 yueyue.nyy
'''

import sys
import os


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
extract from scp file
scppath:                scp file path
'''
def extractSCP(scppath):
    scpbase = os.path.split(scppath)[0]
    
    with open(scppath, 'r', encoding = 'UTF-8') as fd:
        flist = fd.readlines()
    
    for line in flist:
        name, path = line.strip().split()
        farkname = os.path.split(path)[1]
        # print(name)
        # print(farkname)
        
        pathin = os.path.join(scpbase, farkname)
        # if not os.path.exists(pathin):
        #     pathin = os.path.join(basein, 'ark/' + farkname)
        
        dirout = os.path.join(basein, os.path.splitext(farkname)[0])
        if not os.path.exists(dirout):
            os.mkdir(dirout)
        pathout = os.path.join(dirout, name + '.wav')
        
        cmd = './wav-copy ' + os.path.abspath(pathin) + ' ' + os.path.abspath(pathout)
        print(cmd)
        os.system(cmd)


if len(sys.argv) != 2:
    sys.stderr.write('Usage:\n')
    sys.stderr.write('ExtractArk <basein>\n')
    exit(-1)

basein = sys.argv[1]


fin = os.path.join(basein, 'wavelist.txt')
if os.path.exists(fin):
    extractSCP(fin)
else:
    for fscp in listFiles(basein, ['.scp', '.txt']):
        extractSCP(fscp)
