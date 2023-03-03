'''
Data augmentation by sox speed.

Copyright: 2022-03-18 yueyue.nyy
'''

import os
import sys
import traceback
import threading
import queue


# no. of threads
NUM_THS = 50


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


if len(sys.argv) < 3:
    sys.stderr.write('Data augmentation by sox speed.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('AugSpeed <src list or directory> <baseout> [speed ratio list]\n')
    exit(-1)

srclist = sys.argv[1]
baseout = sys.argv[2]
if len(sys.argv) > 3:
    speedl = sys.argv[3:]
else:
    speedl = ['0.9', '1.1']


class AugThread(threading.Thread):
    
    '''
    taskqueue:          the task queue
    '''
    def __init__(self, taskqueue):
        threading.Thread.__init__(self)
        
        self.taskqueue = taskqueue
    
    
    def run(self):
        while not self.taskqueue.empty():
            ts = self.taskqueue.get()
            self.taskqueue.task_done()
            
            ts = ts.strip()
            # get file name
            name = os.path.split(ts)[1]
            name = os.path.splitext(name)[0]
            
            for speed in speedl:
                fout = os.path.join(baseout, name + '_x' + speed +'.wav')
                
                try:
                    cmd = 'sox ' + ts + ' ' + fout + ' speed ' + speed
                    # print(cmd)
                    retval = os.system(cmd)
                    print('DONE: ' + fout)
                except:
                    traceback.print_exc()
                    print('FAILED: ' + fout)


if os.path.isfile(srclist):
    with open(srclist, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
else:
    lines = listFiles(srclist, ['.wav'])

# convert list to queue
fqueue = queue.Queue(len(lines))
for fin in lines:
    fqueue.put(fin)

for i in range(NUM_THS):
    th = AugThread(fqueue)
    th.start()
