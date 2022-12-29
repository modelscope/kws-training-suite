# Copyright (c) Alibaba, Inc. and its affiliates.
# Align command utterances.
# 2022-06-27 yueyue.nyy
# 2022-11 update by bin.xue

import argparse
import os
import math
import sys
import tempfile
import numpy as np
import traceback
import threading
import queue
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

# no. of threads
NUM_THS = 1
# label gain
LABEL_GAIN = 100.0
LD_LIBRARY_PATH = 'lib'
ALIGN_TOOL_PATH = 'bin/lt-force_align'
ALIGN_MODEL_PATH = 'fa_model'
ALIGN_CONF_PATH = 'fa_model/api_ngram.cfg'
SPACE_SEC = 0.5


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
        data = np.fromfile(wavin, dtype='int16')

    padlen = math.floor(fs * SPACE_SEC)

    data2 = np.zeros(data.shape[0] + padlen * 2, dtype='int16')
    data2[padlen:padlen + data.shape[0]] = data[:]

    wavfile.write(wavout, fs, data2)


def applyAlign(asrres, alignres, wavin, wavout):
    """
    apply align result
    asrres:                 asrresult
    alignres:               align result
    wavin:                  input wave
    wavout:                 output wave
    """
    with open(alignres, 'r', encoding='UTF-8') as file:
        lines = file.readlines()

    for salign in lines:
        salign = salign.strip()
        if 'TIMESTAMP' not in salign:
            continue

        salign = salign.split('.wav')[1]
        align_list = salign.split()[1:]
        try:
            fs, data = wavfile.read(wavin)
            startidx = int(align_list[0].split('-')[0]) * fs // 1000
            endidx = int(align_list[len(asrres) - 1].split('-')[1]) * fs // 1000

            data2 = np.zeros((endidx - startidx, 2), dtype='int16')
            data2[:, 0] = data[startidx:endidx]

            len_prefix = 0
            maxseglen = 0
            for i in range(len(asrres)):
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
            wavfile.write(wavout, fs, data2[:min(data2.shape[0], len_prefix)])
        except:
            raise IOError


class AlignThread(threading.Thread):
    """
    taskqueue:          the task queue
    baseout:            output directory
    """

    def __init__(self, base_path, keyword, taskqueue, baseout):
        threading.Thread.__init__(self)
        self.base_cmd = f'LD_LIBRARY_PATH={os.path.join(base_path, LD_LIBRARY_PATH)}:$LD_LIBRARY_PATH ' \
                        f'{os.path.join(base_path, ALIGN_TOOL_PATH)} ' \
                        f'{os.path.join(base_path, ALIGN_MODEL_PATH)} ' \
                        f'{os.path.join(base_path, ALIGN_CONF_PATH)} '
        self.keyword = keyword
        self.split_keyword = ' '.join(tuple(keyword))
        self.taskqueue = taskqueue
        self.baseout = baseout

    def run(self):
        while not self.taskqueue.empty():
            fin = self.taskqueue.get()
            self.taskqueue.task_done()
            self.align(fin.strip())

    def align(self, wavin):
        """
        align one file
        wavin:                  input file
        baseout:                output base
        """
        try:
            # create a temp audio file in baseout, pad audio with space
            name = os.path.splitext(os.path.split(wavin)[1])[0]
            tmpfd, tmppath = tempfile.mkstemp(prefix=name + '_', suffix='.wav', dir=self.baseout)
            os.close(tmpfd)
            padWave(wavin, tmppath)

            # get force align result file
            alignres = self.forceAlign(tmppath, self.baseout)
            # apply align result
            wavout = os.path.join(self.baseout, name + '.wav')
            if os.path.exists(wavout):
                tmpfd, wavout = tempfile.mkstemp(prefix=name + '_', suffix='.wav', dir=self.baseout)
                os.close(tmpfd)
            applyAlign(self.keyword, alignres, tmppath, wavout)
            print('DONE: ' + wavin)
        except IOError:
            traceback.print_exc()
            print('FAILED: ' + wavin)
        finally:
            os.remove(tmppath)
            os.remove(alignres)

    def forceAlign(self, wavin, baseout):
        """
        perform align
        wavin:                  input wave path
        asrres:                 asrresult
        baseout:                output directory
        return:                 align_result.txt
        """
        try:
            tmpfd, fres = tempfile.mkstemp(prefix='align_result_', suffix='.txt', dir=baseout)
            os.close(tmpfd)

            tmpfd, flog = tempfile.mkstemp(prefix='align_result_', suffix='.log', dir=baseout)
            os.close(tmpfd)

            # text file for split
            tmpfd, ftextchar = tempfile.mkstemp(prefix='text_char_', suffix='.txt', dir=baseout)
            os.close(tmpfd)
            fd = open(ftextchar, 'w', encoding='UTF-8')
            fd.write(f'{os.path.abspath(wavin)}\t{self.split_keyword}\n')
            fd.close()

            # call align
            cmd = f'{self.base_cmd} {ftextchar} >>{fres} 2>>{flog}'
            print(cmd)
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
            os.remove(flog)
            os.remove(ftextchar)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KWS force align script')
    parser.add_argument('input', help='path of the audio list file or the directory storing audio files')
    parser.add_argument('keyword', help='the keyword in Chinese')
    parser.add_argument('-o', '--out_dir', help='output directory, default: [input]-align')
    parser.add_argument('-t', '--threads', help='parallel thread number, default: 1', type=int)
    args = parser.parse_args()

    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    basein = args.input
    if args.out_dir:
        baseout = args.out_dir
    else:
        if basein[-1] == '/':
            basein = basein[:-1]
        baseout = basein + '-align'
    os.makedirs(baseout)
    kw = args.keyword
    threads = args.threads if args.threads else NUM_THS

    if os.path.isdir(basein):
        flist = listFiles(basein, ['.wav'])
    else:
        with open(basein, 'r', encoding='UTF-8') as fd:
            flist = fd.readlines()

    # convert list to queue
    fqueue = queue.Queue(len(flist))
    for fin in flist:
        fqueue.put(fin)

    for i in range(threads):
        th = AlignThread(script_path, kw, fqueue, baseout)
        th.start()
