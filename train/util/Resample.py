'''
Created on 2020-11-04

@author: yueyue.nyy

Resample wave files.

'''

import sys
import os
import traceback
import threading
import queue


# no. of threads
NUM_THS = 50


def listFiles(base, suffix):
    suffix2 = []
    for ext in suffix:
        suffix2.append(ext.replace('.', ''))
    
    filelist = []
    filelist = listFilesRec(base, suffix2, filelist)
    return filelist
    
    
def listFilesRec(path, suffix, filelist):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].replace('.', '')
        if ext in suffix:
            filelist.append(path)            
    else:
        for f in os.listdir(path):
            listFilesRec(os.path.join(path, f).replace('\\', '/'), suffix, filelist)
                 
    return filelist


if len(sys.argv) != 4:
    sys.stderr.write('Resample wave files\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('Resample <fs> <basein> <baseout>\n')
    exit(-1)

fs = sys.argv[1]
basein = sys.argv[2]
baseout = sys.argv[3]


if os.path.isfile(basein):
    with open(basein, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
else:
    lines = listFiles(basein, ['.wav'])

# convert list to queue
fqueue = queue.Queue(len(lines))
for fin in lines:
    fqueue.put(fin)


class Worker(threading.Thread):
    
    '''
    taskqueue:          the task queue
    '''
    def __init__(self, taskqueue):
        threading.Thread.__init__(self)
        
        self.taskqueue = taskqueue
    
    
    def run(self):
        while not self.taskqueue.empty():
            fin = self.taskqueue.get()
            self.taskqueue.task_done()
            
            fin = fin.strip()
            fout = os.path.join(baseout, os.path.split(fin)[1])
            
            try:
                cmd = 'sox ' + fin + ' -r ' + fs + ' ' + fout
                # print(cmd)
                retval = os.system(cmd)
                print('DONE: ' + fout)
            except:
                traceback.print_exc()
                print('FAILED: ' + fout)


for i in range(NUM_THS):
    th = Worker(fqueue)
    th.start()
