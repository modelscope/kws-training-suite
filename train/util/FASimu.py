'''
Generate false alarm utterances by data simulation.

Copyright: 2022-12-28 yueyue.nyy
'''

import os
import sys
import math
import tempfile
import numpy as np
import re
import traceback
import threading

# no. of threads
NUM_THS = 40
BASE_OUT = './fa_utt'

FE_EXE_PATH = './SoundConnect'
if not os.path.exists(FE_EXE_PATH):
    FE_EXE_PATH = './SoundConnect.exe'
if not os.path.exists(FE_EXE_PATH):
    FE_EXE_PATH = 'D:/data/programming/eclipse-workspace/SoundConnect/Debug/SoundConnect.exe'


if len(sys.argv) != 2:
    sys.stderr.write('Generate false alarm utterances by data simulation.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('FASimu <conf>\n')
    exit(-1)

fconf = sys.argv[1]


class Worker(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
    
    
    def run(self):
        cmd = FE_EXE_PATH + ' ' + fconf + ' 0 0 >/dev/null'
        retval = os.system(cmd)

cmd = 'rm -rf ' + BASE_OUT
os.system(cmd)
cmd = 'mkdir ' + BASE_OUT
os.system(cmd)

for i in range(NUM_THS):
    th = Worker()
    th.start()
