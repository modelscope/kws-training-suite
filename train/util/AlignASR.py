'''
Extract keyword and generate label by external align tool, from asr result.

Copyright: 2022-10-27 yueyue.nyy
'''

import os
import sys
import math
import tempfile
import numpy as np
import re
import traceback
import threading
import queue
from scipy.io import wavfile


# no. of threads
NUM_THS = 1
KEYWORD = '声音大一点'
# label gain
LABEL_GAIN = 100.0
LD_LIBRARY_PATH = '/home/yueyue.nyy/data2/ForceAlign/FA_0413/srbp/lib'
ALIGN_TOOL_PATH = '/home/yueyue.nyy/data2/ForceAlign/FA_0413/srbp/lt-force_align'
ALIGN_MODEL_PATH = '/home/yueyue.nyy/data2/ForceAlign/deploy_model_0628_16k'
ALIGN_CONF_PATH = '/home/yueyue.nyy/data2/ForceAlign/deploy_model_0628_16k/api_ngram.cfg'
SPACE_SEC = 0.5


'''
pad wave file with space at the beginning and the end.
wavin:                  input wave path
wavout:                 output wave path
'''
def padWave(wavin, wavout):
    if wavin.endswith('.wav'):
        try:
            fs, data = wavfile.read(wavin)
        except ValueError:
            raise IOError
        except WavFileWarning:
            raise IOError
    else:
        fs = 16000
        data = np.fromfile(wavin, dtype = 'int16')
    
    padlen = math.floor(fs * SPACE_SEC)
    
    data2 = np.zeros(data.shape[0] + padlen * 2, dtype = 'int16')
    data2[padlen:padlen + data.shape[0]] = data[:]
    
    wavfile.write(wavout, fs, data2)


'''
perform align
wavin:                  input wave path
asrres:                 asrresult
baseout:                output directory
return:                 align_result.txt
'''
def forceAlign(wavin, asrres, baseout):
    try:
        tmpfd, fres = tempfile.mkstemp(prefix = 'align_result_', suffix = '.txt', dir = baseout)
        os.close(tmpfd)
        
        tmpfd, flog = tempfile.mkstemp(prefix = 'align_result_', suffix = '.log', dir = baseout)
        os.close(tmpfd)
        
        # text file for split
        tmpfd, ftextchar = tempfile.mkstemp(prefix = 'text_char_', suffix = '.txt', dir = baseout)
        os.close(tmpfd)
        fd = open(ftextchar, 'w', encoding = 'UTF-8')
        
        fd.write(os.path.abspath(wavin) + '\t')
        for i in range(len(asrres)):
            fd.write(asrres[i])
            if i < len(asrres) - 1:
                fd.write(' ')
        fd.write('\n')
        
        fd.close()
        
        # call align
        cmd = 'export LD_LIBRARY_PATH=' + LD_LIBRARY_PATH
        retval = os.system(cmd)
        
        cmd = ALIGN_TOOL_PATH + ' ' + ALIGN_MODEL_PATH + ' ' + ALIGN_CONF_PATH + ' ' + os.path.abspath(ftextchar) \
              + ' >>' + os.path.abspath(fres) + ' 2>>' +os.path.abspath(flog)
        retval = 0
        
        try:
            retval = os.system(cmd)
        except BaseException:
            raise IOError
        
        if retval == 0:
            return fres
        else:
            raise IOError
    finally:
        try:
            os.remove(flog)
        except:
            pass
        
        try:
            os.remove(ftextchar)
        except:
            pass


'''
apply align result
asrres:                 asrresult
alignres:               align result
wavin:                  input wave
wavout:                 output wave
'''
def applyAlign(asrres, alignres, wavin, wavout):
    with open(alignres, 'r', encoding = 'UTF-8') as file:
        lines = file.readlines()
    
    for salign in lines:
        salign = salign.strip()
        if 'TIMESTAMP' not in salign:
            continue
        
        align_list = salign.split()[2:]
        
        try:
            fs, data = wavfile.read(wavin)

            idx = asrres.find(KEYWORD)            
            startidx = int(align_list[idx].split('-')[0]) * fs // 1000
            endidx = int(align_list[idx + len(KEYWORD) - 1].split('-')[1]) * fs // 1000
            
            data2 = np.zeros((endidx - startidx, 2), dtype = 'int16')
            data2[:, 0] = data[startidx:endidx]
            
            len_prefix = 0
            maxseglen = 0
            for i in range(len(KEYWORD)):
                idx0, idx1 = align_list[i].split('-')
                idx0 = int(idx0) * fs // 1000 - startidx
                idx1 = int(idx1) * fs // 1000 - startidx
                
                ilabel = int(float(i + 1) * 32768.0 / LABEL_GAIN)
                if ilabel >= 32768.0:
                    ilabel = 32767.0
                
                data2[idx0:idx1, 1] = ilabel
                
                # control last character's length
                seglen = idx1 - idx0
                
                if i == len(asrres) - 1:
                    len_prefix += min(seglen, maxseglen)
                else:
                    len_prefix += seglen
                    
                    if seglen > maxseglen:
                        maxseglen = seglen
                
                # if i == 0 or len(asrres) <= 2:
                #     len_prefix += idx1 - idx0
                # elif i < len(asrres) - 1:
                #     len_prefix += (idx1 - idx0) * 2
            
            wavfile.write(wavout, fs, data2[:min(data2.shape[0], len_prefix)])
        except:
            raise IOError


'''
align one file
scpentry:               scp entry, path (id) asr result
baseout:                output base
'''
def align1(scpentry, baseout):
    try:
        sts = scpentry.split()
        if len(sts) <= 1:
            return
        
        name = sts[0]
        asrres = ''.join(sts[1:])
        
        if KEYWORD not in asrres:
            return
        
        wavin = './wav/' + name + '.wav'
        if not os.path.exists(wavin):
            return
        
        # pad audio with space
        name = os.path.splitext(os.path.split(wavin)[1])[0]
        # name = name.replace('.wav', '')
        # name = name.replace('.WAV', '')
        
        tmpfd, tmppath = tempfile.mkstemp(
            prefix = name + '_', suffix = '.wav', dir = baseout)
        os.close(tmpfd)
        
        padWave(wavin, tmppath)
        
        # force align
        alignres = forceAlign(tmppath, asrres, baseout)
        
        # apply align result
        wavout = os.path.join(baseout, name + '.wav')
        if os.path.exists(wavout):
            tmpfd, wavout = tempfile.mkstemp(
                prefix = name + '_', suffix = '.wav', dir = baseout)
            os.close(tmpfd)
        
        applyAlign(asrres, alignres, tmppath, wavout)
        
        print('DONE: ' + wavin + ' ' + asrres)
    except IOError:
        traceback.print_exc()
        print('FAILED: ' + scpentry)
    finally:
        try:
            os.remove(tmppath)
        except:
            pass
        
        try:
            os.remove(alignres)
        except:
            pass


class AlignThread(threading.Thread):
    
    '''
    taskqueue:          the task queue
    baseout:            output directory
    '''
    def __init__(self, taskqueue, baseout):
        threading.Thread.__init__(self)
        
        self.taskqueue = taskqueue
        self.baseout = baseout
    
    
    def run(self):
        while not self.taskqueue.empty():
            fin = self.taskqueue.get()
            self.taskqueue.task_done()
            
            align1(fin.strip(), self.baseout)


if len(sys.argv) != 3:
    sys.stderr.write("Extract keyword and generate label by external align tool, from asr result.\n")
    sys.stderr.write("Usage:\n")
    sys.stderr.write("AlignASR <scp> <output dir>\n")
    exit(-1)

pathscp = sys.argv[1]
baseout = sys.argv[2]

with open(pathscp, 'r', encoding = 'UTF-8') as fd:
    flist = fd.readlines()

# convert list to queue
fqueue = queue.Queue(len(flist))
for fin in flist:
    fqueue.put(fin)

for i in range(NUM_THS):
    th = AlignThread(fqueue, baseout)
    th.start()
