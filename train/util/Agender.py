'''
Perform gender classification: 1-child, 2-male, 3-female.

Copyright: 2022-07-26 yueyue.nyy
'''

import os
import sys
import math
import tempfile
import numpy as np
import re
import subprocess
import traceback
import threading
import queue
from scipy.io import wavfile


NUM_THS = 50
BASE_AGENDER = '/home/yueyue.nyy/data2/agender'


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
perform gender classification
fwav:                   input wave file path
return:                 1-child, 2-male, 3-female
'''
def agender1(fwav):
    try:
        # extract the first channel
        fs, data = wavfile.read(fwav)
        tmpfd, wavpath = tempfile.mkstemp(prefix = 'input_', suffix = '.wav', dir = '.')
        os.close(tmpfd)
        wavfile.write(wavpath, fs, data[:, 0])
        
        # generate scp file
        tmpfd, scppath = tempfile.mkstemp(prefix = 'scp_', suffix = '.txt', dir = '.')
        os.close(tmpfd)
        
        fd = open(scppath, 'w', encoding = 'UTF-8')
        fd.write(wavpath + '\n')
        fd.flush()
        fd.close()
        
        # export LD_LIBRARY_PATH=/home/yueyue.nyy/data2/agender/lib
        # call agender
        # cmd = os.path.join(BASE_AGENDER, 'run') + ' ' + scppath + ' ' + os.path.join(BASE_AGENDER, 'models')
        # print(cmd)
        # retval = os.system(cmd)
        retstr = subprocess.check_output(
            [os.path.join(BASE_AGENDER, 'run'), 
             scppath, 
             os.path.join(BASE_AGENDER, 'models')]).decode('UTF-8').strip()
        # print(retstr)
        
        print('DONE: ' + fwav)
        return int(retstr[-1])
    except:
        print('FAILED: ' + fwav)
    finally:
        os.remove(wavpath)
        os.remove(scppath)


if len(sys.argv) != 5:
    sys.stderr.write('Perform gender classification: 1-child, 2-male, 3-female.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('Agender <base or list in> <child list> <male list> <female list>\n')
    exit(-1)

basein = sys.argv[1]
fchild = sys.argv[2]
fmale = sys.argv[3]
ffemale = sys.argv[4]

fdchild = open(fchild, 'w', encoding = 'UTF-8')
fdmale = open(fmale, 'w', encoding = 'UTF-8')
fdfemale = open(ffemale, 'w', encoding = 'UTF-8')

if os.path.isdir(basein):
    flist = listFiles(basein, ['.wav'])
else:
    with open(basein, 'r', encoding = 'UTF-8') as fd:
        flist = fd.readlines()


class AgenderThread(threading.Thread):
    
    '''
    taskqueue:          the task queue
    '''
    def __init__(self, taskqueue):
        threading.Thread.__init__(self)
        
        self.taskqueue = taskqueue
    
    
    def run(self):
        while not self.taskqueue.empty():
            fin = self.taskqueue.get().strip()
            self.taskqueue.task_done()
            
            retval = agender1(fin)
            
            if retval == 1:
                fdchild.write(fin + '\n')
            elif retval == 2:
                fdmale.write(fin + '\n')
            elif retval == 3:
                fdfemale.write(fin + '\n')


# convert list to queue
fqueue = queue.Queue(len(flist))
for fin in flist:
    fqueue.put(fin)

thl = []
for i in range(NUM_THS):
    th = AgenderThread(fqueue)
    th.start()
    thl.append(th)

for th in thl:
    th.join()

fdchild.flush()
fdchild.close()

fdmale.flush()
fdmale.close()

fdfemale.flush()
fdfemale.close()
