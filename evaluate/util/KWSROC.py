# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Collect data for kws roc curve, supports multiple keywords.
# 2022-03-17 yueyue.nyy
# 2022-11-09 updated by bin.xue

import os
import sys
import numpy as np
import re
from librosa import get_duration


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


def loadPositive(base):
    """ load positive examples
    base:                   positive result base
    return:                 {kw: count}, {kw: [confidence list]}
    """
    countdict = {}
    confdict = {}
    
    for f in listFiles(base, ['.txt']):
        fname = os.path.split(f)[1]
        if fname != 'wake_summary.txt':
            continue
        
        with open(f, 'r', encoding = 'UTF-8') as fd:
            lines = fd.readlines()
        
        for ts in lines:
            ts = ts.strip()
            
            # parse realcount of each keyword
            if ts.startswith('Recall: '):
                ts = ts[len('Recall: '):]
                
                for ts2 in ts.split(', '):    
                    m = re.match('(.+): .+% \(\d+/(\d+)\)', ts2)
                    if m:
                        kw = countdict.get(m.group(1))
                        if kw is None:
                            countdict.update({m.group(1): 0})
                        
                        countdict[m.group(1)] += int(m.group(2))
            else:
                # collect confidence
                m = re.match('\[detected\s+(\d+)\], kw: (.+), spot: (.+), bestend: (.+), duration: \[(.+)-(.+)\], confidence: (.+), bestch: (\d+)', ts)
                if m:
                    kw = confdict.get(m.group(2))
                    if kw is None:
                        confdict.update({m.group(2): []})
                    
                    confdict[m.group(2)].append(float(m.group(7)))
    
    return countdict, confdict


def loadFA(base):
    """ load fa examples
    base:                   negative result base
    return:                 fa confidence list {kw: [confidence list]}
    """
    confdict = {}
    
    for f in listFiles(base, ['.txt']):
        fname = os.path.split(f)[1]
        if fname != 'wake_summary.txt':
            continue
        
        with open(f, 'r', encoding='UTF-8') as fd:
            lines = fd.readlines()
        
        for ts in lines:
            ts = ts.strip()
            m = re.match(
                '\[detected\s+(\d+)\], kw: (.+), spot: (.+), bestend: (.+), duration: \[(.+)-(.+)\], confidence: (.+), bestch: (\d+)',
                ts)
            if m:
                kw = confdict.get(m.group(2))
                if kw is None:
                    confdict.update({m.group(2): []})

                confdict[m.group(2)].append(float(m.group(7)))
    
    return confdict


def totalWavLength(base):
    """ get total wave files length (hour)
    base:                   wav file base
    """
    lensec = 0.0
    
    for f in listFiles(base, ['.wav']):
        # fs, data = wavfile.read(f)
        # lensec += float(data.shape[0]) / fs
        lensec += get_duration(filename = f)
    
    return lensec / 3600.0


def printROC(kw, myprint):
    """ print roc for a single keyword
    kw:                     the keyword
    return:                 count list [thres, wakecount, facount]
    """
    myprint(kw)
    retl = []
    
    for thres in np.linspace(1.0, 0.0, 101):
        # cut positive data according to thres
        wakecount = 0
        if kw in posconfldict:
            for val in posconfldict[kw]:
                if val >= thres:
                    wakecount += 1
        
        # cut negative data according to thres
        facount = 0
        if kw in faconfldict.keys():
            for val in faconfldict[kw]:
                if val >= thres:
                    facount += 1
        
        # avoid zero denominator
        if falen == 0.0:
            myprint(0.0, 1.0 - wakecount / poscountdict[kw], '{:.2f}'.format(thres))
        elif poscountdict[kw] == 0.0:
            myprint(facount / falen, 1.0, '{:.2f}'.format(thres))
        else:
            myprint(facount / falen, 1.0 - wakecount / poscountdict[kw], '{:.2f}'.format(thres))
        
        retl.append([thres, wakecount, facount])
    
    return retl


def printTotalROC(myprint=None):
    """
    print unified roc for all keywords
    """
    myprint('Total')
    
    # total ground truth positive examples    
    poscount = 0
    for kw in poscountdict:
        poscount += poscountdict[kw]
    
    # convert dict to list
    wakefacountl = []
    for kw in sorted(wakefacountdict.keys()):
        wakefacountl.append(wakefacountdict[kw])
    
    # indices of each keyword
    idx = [0] * len(wakefacountdict)
    
    for i in range(len(wakefacountl) * len(wakefacountl[0]) - (len(wakefacountl) - 1)):
        wakecount = 0
        facount = 0
        thres = []
        
        for j in range(len(wakefacountl)):
            thres.append(wakefacountl[j][idx[j]][0])
            wakecount += wakefacountl[j][idx[j]][1]
            facount += wakefacountl[j][idx[j]][2]
        
        ts = str(facount / falen) + ' ' + str(1.0 - wakecount / poscount) + ' '
        for j in range(len(thres)):
            ts += '{:.2f}'.format(thres[j])
            if j < len(thres) - 1:
                ts += ' '
        
        myprint(ts)
        
        idx[i % len(idx)] += 1


def kws_roc(basepos, baseneg, basenegwav, lenratio, myprint=None):
    global poscountdict, posconfldict, faconfldict, falen, wakefacountdict
    # load positive examples
    poscountdict, posconfldict = loadPositive(basepos)
    # load fa
    faconfldict = loadFA(baseneg)
    # fa data length (hour)
    falen = totalWavLength(basenegwav) * lenratio
    # print roc for each keyword
    wakefacountdict = {}
    for kw in sorted(poscountdict.keys()):
        retl = printROC(kw, myprint)
        wakefacountdict.update({kw: retl})
        myprint()


def get_myprint(f_name):
    def file_print(*args):
        with open(f_name, 'a') as f:
            first = True
            for value in args:
                if first:
                    first = False
                else:
                    f.write(' ')
                if not isinstance(value, str):
                    value = str(value)
                f.write(value)
            f.write('\n')
    return file_print


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.stderr.write('Collect data for kws roc curve, supports multiple keywords.\n')
        sys.stderr.write('Usage:\n')
        sys.stderr.write('KWSROC <base positive> <base negative> <wav negative> [negative dataset length ratio]\n')
        exit(-1)

    basepos = sys.argv[1]
    baseneg = sys.argv[2]
    basenegwav = sys.argv[3]
    # negative dataset length ratio
    lenratio = 1.0
    if len(sys.argv) > 4:
        lenratio = float(sys.argv[4])

    kws_roc(basepos, baseneg, basenegwav, lenratio, myprint=print)
    # print total roc
    printTotalROC(myprint=print)
