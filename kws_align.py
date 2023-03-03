# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Align by keyword spotting results.
# 2022-03-09 yueyue.nyy
# 2022-12-09 bin.xue updated

import argparse
import os
import sys
import tempfile
from concurrent import futures

import numpy as np
import re
import traceback

from modelscope.utils.audio.audio_utils import update_conf
from scipy.io import wavfile
from tqdm import tqdm

# no. of threads
NUM_THS = 1
# audio repeats
TRAIN_REPEAT = 2
# data block size (second)
BLOCK_SIZE = 0.02
# left offset (second)
L_OFFSET = -0.1
# label gain
LABEL_GAIN = 100.0
# default sample rate
FS = 16000
# fbank size
FBANK_SIZE = 40

FE_EXE_PATH = 'bin/SoundConnect'


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


def loadAudio(fin):
    """ load audio from file
    fin:                    input file path
    return:                 datatype, loaded data
    """
    fmt = os.path.splitext(fin)[1]

    if fmt == '.wav':
        dtype = 'int16'
        fs, data = wavfile.read(fin)
    elif fmt == '.pcm':
        dtype = 'int16'
        data = np.fromfile(fin, dtype)
    elif fmt == '.f32':
        dtype = 'float32'
        data = np.fromfile(fin, dtype)
    else:
        raise IOError('Failed to load audio: ' + fin)

    return dtype, data


def saveAudio(fout, data):
    """ save audio into file
    fout:                   output file path
    data:                   audio data
    """
    fmt = os.path.splitext(fout)[1]

    if fmt == '.wav':
        wavfile.write(fout, FS, data)
    elif fmt == '.pcm':
        data.tofile(fout)
    elif fmt == '.f32':
        data.tofile(fout)
    else:
        raise IOError('Failed to save audio: ' + fout)


def loadKeywords(fconf):
    """ parse keywords and corresponding labels
    fconf:                  conf file path
    return:                 {keyword: [labels]}
    """
    with open(fconf, 'r', encoding='UTF-8') as fd:
        lines = fd.readlines()

    begindesc = False
    kwdict = {}

    for ts in lines:
        ts = ts.strip()
        if ts.startswith('#'):
            continue
        if len(ts) <= 0:
            continue

        if begindesc:
            m = re.match('.+\s*=\s*', ts)
            if m is not None:
                break

            # parse keywords and labels
            sts = ts.split(',')
            kw = sts[0]
            labels = list(map(int, sts[1:]))
            kwdict.update({kw: labels})

        if ts.startswith('kws_decode_desc ='):
            begindesc = True

    return kwdict


def createFeIn(fin):
    """ generate temp file as front-end input
    fin:                    original audio file
    return:                 tmp file path, padded data
    """
    # load audio file
    dtype, data = loadAudio(fin)

    # pad audio
    dataout = np.zeros((datapad.shape[0] + data.shape[0]) * TRAIN_REPEAT, dtype)
    for i in range(TRAIN_REPEAT):
        offset = (datapad.shape[0] + data.shape[0]) * i
        dataout[offset:offset + datapad.shape[0]] = datapad[:]

        offset += datapad.shape[0]
        dataout[offset:offset + data.shape[0]] = data[:]

    # output audio
    name = os.path.splitext(os.path.split(fin)[1])[0]
    tmpfd, tmppath = tempfile.mkstemp(
        prefix='fein_' + name + '_', suffix=os.path.splitext(fin)[1], dir=baseout)
    os.close(tmpfd)
    saveAudio(tmppath, dataout)

    return tmppath, dataout


def applyFE(fconf, fin):
    """ apply front-end
    fconf:                  fe conf path
    fin:                    input file path
    return:                 feout, stdout, stderr file path
    """
    name = os.path.splitext(os.path.split(fin)[1])[0]

    feoutpath = os.path.join(baseout, name + '_feout.wav')
    stdoutpath = os.path.join(baseout, name + '_stdout.txt')
    stderrpath = os.path.join(baseout, name + '_stderr.txt')

    # call fe
    cmd = FE_EXE_PATH + ' ' + fconf
    cmd += ' ' + fin + ' ' + feoutpath + ' 1>' + stdoutpath + ' 2>' + stderrpath

    retval = 0
    try:
        retval = os.system(cmd)
    except BaseException:
        raise IOError('Failed to apply fe: ' + str(retval))

    if retval == 0:
        return feoutpath, stdoutpath, stderrpath
    else:
        raise IOError('Failed to apply fe: ' + str(retval))


def updateToken(stseq, stin, token):
    """ update token
    stseq:                  state sequence to be detected
    stin:                   input state
    token:                  current token index, -1 means final state
    return:                 output token index, -1 means final state
    """
    if token < -1:
        token = -1
    elif token > len(stseq) - 1:
        token = len(stseq) - 1

    if 0 <= token < len(stseq) - 1:
        # current is middle state
        if stin == stseq[token + 1]:
            token += 1
        elif stin != stseq[token]:
            token = -1
    elif token == len(stseq) - 1:
        # current is the last state
        if stin != stseq[token]:
            token = -1

    # from final move to the first state
    if token == -1 and stin == stseq[0]:
        token = 0

    return token


def detectStrictBoundary(bestpath, offset, length, stseq):
    """ detect strict keyword boundary, keyword label order considered
    bestpath:               decode path
    offset:                 lookup offset
    length:                 lookup length
    stseq:                  keyword label sequence
    return:                 kwoffset: keyword offset, -1 means failed
                            kwlen: keyword length
    """
    # detect strict boundary
    token = -1
    kwoffset = -1
    kwlen = 0
    kwexists = False

    for tau in range(offset, offset + length):
        token2 = updateToken(stseq, bestpath[tau], token)

        if token != 0 and token2 == 0:
            kwoffset = tau
        elif token2 == len(stseq) - 1:
            kwlen = tau - kwoffset + 1
            kwexists = True

        token = token2

    if not kwexists:
        return -1, 0
    else:
        return kwoffset, kwlen


def detectBoundary(bestpath, stseq):
    """ detect keyword boundary
    bestpath:               decode path
    stseq:                  keyword label list
    return:                 augpath: boundary augmented path, None means failed
                            offset: keyword offset
                            len: keyword length
                            relax: no. of relaxed labels at the beginning
    """
    # find keyword label boundary returned by the event log
    taustart = 0
    tauend = 0

    for tau in range(len(bestpath) - 2, -1, -1):
        if bestpath[tau + 1] == 0 and bestpath[tau] != 0:
            tauend = tau
        elif bestpath[tau + 1] != 0 and bestpath[tau] == 0:
            taustart = tau + 1
            break

    if bestpath[-1] != 0:
        tauend = len(bestpath) - 1

    taulen = tauend - taustart + 1
    if taulen <= 0:
        return None, -1, 0, 0

    # detect strict boundary
    kwoffset, kwlen = detectStrictBoundary(bestpath, taustart, taulen, stseq)

    if kwoffset <= 0:
        kwoffset, kwlen = detectStrictBoundary(
            bestpath, taustart, taulen, stseq[:len(stseq) - 1])

    if kwoffset <= 0:
        kwoffset, kwlen = detectStrictBoundary(
            bestpath, taustart, taulen, stseq[1:])

    if kwoffset <= 0:
        return None, -1, 0, 0

    # boundary relax
    count = [0] * len(stseq)
    for tau in range(kwoffset, kwoffset + kwlen):
        for i in range(len(stseq)):
            if bestpath[tau] == stseq[i]:
                count[i] += 1

    augpath = bestpath.copy()
    # duration[0] mismatch considered
    relax = taustart - kwoffset

    if count[0] < count[1]:
        for i in range(count[1] - count[0]):
            kwoffset -= 1
            kwlen += 1
            relax += 1

            if kwoffset < 0:
                augpath.insert(0, stseq[0])
                kwoffset = 0
            else:
                augpath[kwoffset] = stseq[0]

    if count[-1] < count[-2]:
        for i in range(count[-2] - count[-1]):
            if kwoffset + kwlen >= len(augpath):
                augpath.append(stseq[-1])
            else:
                augpath[kwoffset + kwlen] = stseq[-1]

            kwlen += 1
    return augpath, kwoffset, kwlen, relax


def alignByKWS(forigin, datarpt, flog):
    """ align one file by kws log
    forigin:                original audio file
    datarpt:                padded data
    flog:                   kws log file path
    return:                 aligned file path, or None if not waked
    """
    # load kws log
    with open(flog, 'r', encoding='UTF-8') as fd:
        lines = fd.readlines()

    kw = None
    duration = None
    confidence = 0.0
    bestpath = None
    pathlidx = -1
    usethiskw = False

    for lidx, ts in enumerate(lines):
        ts = ts.strip()

        m = re.match(
            '\[detected\s+(\d+)\], kw: (.+), spot: (.+), bestend: (.+), duration: \[(.+)-(.+)\], confidence: (.+), bestch: (\d+)',
            ts)
        if m is not None:
            tmpc = float(m.group(7))

            if tmpc > confidence:
                kw = m.group(2)
                duration = [float(m.group(5)), float(m.group(6))]
                confidence = tmpc
                usethiskw = True
            else:
                usethiskw = False

        if ts.startswith('best path:') and usethiskw:
            pathlidx = lidx + 1

        if lidx == pathlidx:
            bestpath = list(map(int, ts.split()))

    # not waked
    if duration is None:
        return None

    # find decode path boundary
    augpath, labeloffset, labellen, labelrelax = detectBoundary(bestpath, kwdict[kw])

    if augpath is None:
        return None

    # determine audio boundary
    if os.path.splitext(forigin)[1] == '.f32':
        tstart = max(
            int((duration[0] - labelrelax * BLOCK_SIZE * 2 + L_OFFSET) / BLOCK_SIZE),
            0) * FBANK_SIZE
        lsize = FBANK_SIZE * 2
    else:
        tstart = max(
            int(FS * (duration[0] - labelrelax * BLOCK_SIZE * 2 + L_OFFSET)),
            0)
        lsize = int(FS * BLOCK_SIZE * 2)

    uttlen = min(lsize * labellen, datarpt.shape[0] - tstart)

    # copy wave data
    data2 = np.zeros((uttlen, 2), dtype=dtypepad)
    data2[:, 0] = datarpt[tstart:tstart + uttlen]

    # copy label
    label = augpath[labeloffset:labeloffset + labellen]
    for li in range(len(label)):
        if dtypepad == 'float32':
            val = label[li] / LABEL_GAIN
        else:
            val = int(32768.0 * label[li] / LABEL_GAIN)

        data2[lsize * li:lsize * (li + 1), 1] = val

    # output file
    nameout, extout = os.path.splitext(os.path.split(forigin)[1])
    if extout == '.pcm':
        extout = '.wav'

    dirout = os.path.join(baseout, kw)
    if not os.path.exists(dirout):
        os.mkdir(dirout)

    fout = os.path.join(dirout, nameout + ('_confidence_{:0.2f}' + extout).format(confidence))

    if os.path.exists(fout):
        tmpfd, fout = tempfile.mkstemp(
            prefix=nameout + '_', suffix=('_confidence_{:0.2f}' + extout).format(confidence), dir=dirout)
        os.close(tmpfd)

    saveAudio(fout, data2)

    return fout


def align(conf_path, fin):
    """
    align 1 file
    fin:                    original audio file
    """
    feinpath = None
    feoutpath = None
    stdoutpath = None
    stderrpath = None
    try:
        # generate fe input
        feinpath, feindata = createFeIn(fin)

        # apply fe
        feoutpath, stdoutpath, stderrpath = applyFE(conf_path, feinpath)

        # align
        fout = alignByKWS(fin, feindata, stdoutpath)
        return fin, fout
    except IOError as e:
        traceback.print_exc()
        return fin, f'Error: {e}'
    finally:
        if feinpath and os.path.isfile(feinpath):
            os.remove(feinpath)
        if feoutpath and os.path.isfile(feoutpath):
            os.remove(feoutpath)
        if stdoutpath and os.path.isfile(stdoutpath):
            os.remove(stdoutpath)
        if stderrpath and os.path.isfile(stderrpath):
            os.remove(stderrpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KWS align script')
    parser.add_argument('input', help='path of the audio list file or the directory storing audio files')
    parser.add_argument('keyword_desc', help='the keyword description')
    parser.add_argument('-m', '--model_txt', required=True, help='the path of model .txt file')
    parser.add_argument('-o', '--out_dir', help='output directory, default: [input]-align')
    parser.add_argument('-t', '--threads', help='parallel thread number, default: 1', type=int)
    args = parser.parse_args()

    threads = args.threads if args.threads else NUM_THS
    basein = args.input
    if args.out_dir:
        baseout = args.out_dir
    else:
        if basein[-1] == '/':
            basein = basein[:-1]
        baseout = basein + '-align'
    os.makedirs(baseout)

    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    fpad = os.path.join(script_path, 'data', 'padding.wav')
    # load padding audio
    dtypepad, datapad = loadAudio(fpad)

    my_conf = {'nummics': 1,
               'numrefs': 0,
               'numins': 1,
               'validate_numouts': 1,
               'kws_log_level': 3,
               'kws_level': '0.0',
               'kws_decode_desc': args.keyword_desc,
               'kws_model': args.model_txt}
    fe_conf_path = os.path.join(os.path.dirname(__file__), 'evaluate', 'conf', 'sc.conf')
    tmpconfpath = os.path.join(baseout, 'tmp.conf')
    update_conf(fe_conf_path, tmpconfpath, my_conf)

    # load keywords and labels
    kwdict = loadKeywords(tmpconfpath)

    if os.path.isdir(basein):
        fmt = os.path.splitext(fpad)[1]
        if fmt == '.wav':
            flist = listFiles(basein, ['.wav', '.pcm'])
        elif fmt == '.f32':
            flist = listFiles(basein, ['.f32'])
        else:
            raise ValueError(f'Unsupported file type!')
    else:
        with open(basein, 'r', encoding='UTF-8') as fd:
            flist = fd.readlines()

    tasks = []
    with open(os.path.join(baseout, 'result.txt'), 'w') as result_f:
        with futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for f in flist:
                tasks.append(executor.submit(align, tmpconfpath, f))
            for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
                result = task.result()
                result_f.write(f'{result[0]}\t{result[1]}\n')
