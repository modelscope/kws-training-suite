# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Test positive kws dataset.

import os
import sys
import re

from modelscope.utils.logger import get_logger

COMMENT_PREFIX = '#'
UTF8_COMMENT_PREFIX = '\ufeff#'
WAKE_SUMMARY_NAME = 'wake_summary.txt'
TOTAL_WAKE_SUMMARY_NAME = 'total_wake_summary.txt'

logger = get_logger()


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
load annotation data
baseannot:              annotation base
return:                 {scene: {file: {keyword: count}}}
'''
def loadAnnot(baseannot):
    retdict = {}
    
    # add scene
    for scene in os.listdir(baseannot):
        pscene = os.path.join(baseannot, scene)
        if os.path.isdir(pscene):
            retdict.update({scene: {}})
    if len(retdict) == 0:
        raise RuntimeError(f'The config "test_pos_anno_dir" is invalid! '
                           f'Failed to find sub directory under {baseannot}')

    txt_found = False
    for scene in retdict:
        scenedict = retdict[scene]

        # read annotation files in each scene
        for f in listFiles(os.path.join(baseannot, scene), ['.txt']):
            txt_found = True
            with open(f, 'r', encoding='UTF-8') as fd:
                lines = fd.readlines()
            for ts in lines:
                if ts.startswith(COMMENT_PREFIX) or ts.startswith(UTF8_COMMENT_PREFIX):
                    continue
                sts = ts.strip().split()
                if len(sts) <= 0:
                    continue
                if len(sts) < 3:
                    raise RuntimeError(f'Invalid annotation format in {f}! The line is "{ts}"')
                # get file name without extension
                filename = os.path.splitext(os.path.split(sts[0])[1])
                if filename[1] != '.wav':
                    raise RuntimeError(
                        f'Invalid file name: {sts[0]} in {f}! Only support *.wav.')
                name = filename[0]

                # get keyword and corresponding count
                kw = {}
                for i in range(len(sts) // 2):
                    kw.update({sts[1 + 2 * i]: int(sts[1 + 2 * i + 1])})

                # add audio file and corresponding keywords and count into scene
                scenedict.update({name: kw})
    if not txt_found:
        raise RuntimeError(f'The config "test_pos_anno_dir" is invalid! '
                           f'Failed to find .txt file in sub directories of {baseannot}.')
    return retdict


'''
parse one row kws log
str:                    log string
return:                 {id, time, kw, duration, confidence, bestch, str}
                        return None for wrong format
'''
def parseKWSLog(str):
    m = re.match('\[detected\s+(\d+)\], kw: (.+), spot: (.+), bestend: (.+), duration: \[(.+)-(.+)\], confidence: (.+), bestch: (\d+)', str)
    if m is None:
        return None
    
    id = int(m.group(1))
    kw = m.group(2)
    spot = float(m.group(3))
    bestend = float(m.group(4))
    duration = [float(m.group(5)), float(m.group(6))]
    confidence = float(m.group(7))
    bestch = int(m.group(8))
    
    return {'id':id, 'kw': kw, 'spot': spot, 'bestend': bestend, 'duration': duration, 'confidence': confidence, 'bestch': bestch, 'str': str}


'''
load kws log from a file
path:                   file path
return:                 log list
'''
def loadKWSLog(path):
    loglist = []
    
    with open(path, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
    
    for str in lines:
        log = parseKWSLog(str.strip())
        if log is not None:
            loglist.append(log)
    
    return loglist


'''
count kws log for wake count
kwslog:                 kws log list
return:                 wake count of each keyword {kw: count}
'''
def countKWSLog(kwslog):
    countdict = {}
    
    for log in kwslog:
        kw = log['kw']
        count = countdict.get(kw)
        if count is None:
            countdict.update({kw: 0})
        
        countdict[kw] += 1
    
    return countdict


'''
find annotation by log file path
path:                   log file path
return:                 annotation {kw: count}, or None if not found
'''
def findAnnot(path):
    scene = os.path.split(os.path.dirname(path))[1]
    scenedict = annotdict.get(scene)
    if scenedict is None:
        return None
    
    name = os.path.splitext(os.path.split(path)[1])[0]
    name = name[:-len('_stdout')]
    if name in scenedict:
        return scenedict.get(name)
    else:
        return {}


'''
kws evaluation for one scene
scene:                  scene name
return:                 eval result {kw: [realcount, annotcount]}
'''
def evalScene(scene, basein, baseout):
    # result of this scene
    resscene = {}
    
    # output result file for the scene
    outfpath = os.path.join(baseout, scene)
    if not os.path.exists(outfpath):
        os.makedirs(outfpath)
    
    outfpath = os.path.join(outfpath, WAKE_SUMMARY_NAME)
    fdout = open(outfpath, mode='w', encoding = 'UTF-8')
    
    # get all files in this scene
    flist = listFiles(os.path.join(basein, scene), ['.txt'])
    flist.sort()
    
    for f in flist:
        # find annotation
        if not f.endswith('_stdout.txt'):
            continue
        # load log
        kwslog = loadKWSLog(f)
        # count wake times
        countdict = countKWSLog(kwslog)
        # result of one file
        resfile = {}
        annot = findAnnot(f)
        if annot is None:
            continue
        elif len(annot) == 0:
            for kw, realcount in countdict.items():
                # update real/annot count
                if kw not in resfile:
                    resfile.update({kw: [0, 0]})
                resfile[kw][0] += realcount
                resfile[kw][1] += 0
                if kw not in resscene:
                    resscene.update({kw: [0, 0]})
                resscene[kw][0] += realcount
                resscene[kw][1] += 0
        else:
            # accumulate count
            for kw in annot:
                # get annotation count of a keyword
                annotcount = annot[kw]
                # get real kws count
                realcount = countdict.get(kw)
                if realcount is None:
                    realcount = 0
                # update real/annot count
                countpair = resfile.get(kw)
                if countpair is None:
                    resfile.update({kw: [0, 0]})
                resfile[kw][0] += realcount
                resfile[kw][1] += annotcount
            # accumulate file count into total count
            for kw in resfile:
                countpair = resscene.get(kw)
                if countpair is None:
                    resscene.update({kw: [0, 0]})
                resscene[kw][0] += resfile[kw][0]
                resscene[kw][1] += resfile[kw][1]

        #
        # output file eval result
        #

        # file name
        tmpstr = os.path.split(f)[1]
        print(tmpstr)
        fdout.write(tmpstr + '\n')
        
        # recall
        tmpstr = 'Recall: '
        idx = 0
        for kw in sorted(resfile.keys()):
            tmpstr += kw + ': '
            countpair = resfile[kw]
            expected_count = countpair[1] if countpair[1] > 0 else 1
            tmpstr += '{:.1%}'.format(countpair[0] / expected_count) + ' (' + str(countpair[0]) + '/' + str(countpair[1]) + ')'
            
            if idx < len(resfile) - 1:
                tmpstr += ', '
            idx += 1
        
        print(tmpstr)
        fdout.write(tmpstr + '\n')
        
        # log items
        for log in kwslog:
            print(log['str'])
            fdout.write(log['str'] + '\n')
        
        print()
        fdout.write('\n')
    
    #
    # print total recall
    #
    tmpstr = 'Total Recall:\n'
    for kw in sorted(resscene.keys()):
        tmpstr += kw + ': '
        countpair = resscene[kw]
        expected_count = countpair[1] if countpair[1] > 0 else 1
        tmpstr += '{:.1%}'.format(countpair[0] / expected_count) + ' (' + str(countpair[0]) + '/' + str(countpair[1]) + ')\n'
    
    print(tmpstr)
    fdout.write(tmpstr)
    
    fdout.flush()
    fdout.close()
    return resscene


def kws_eval(baseannot, basein, baseout):
    global annotdict
    # load annotations
    annotdict = loadAnnot(baseannot)
    # total result, {kw, [realcount, annotcount]}
    restotal = {}
    # result in each scene, {scene: {kw: [realcount, annotcount]}}
    resscene = {}
    # evaulate each scene
    for scene in os.listdir(basein):
        pscene = os.path.join(basein, scene)
        if not os.path.isdir(pscene):
            continue

        # evaluate one scene
        ret = evalScene(scene, basein, baseout)
        resscene.update({scene: ret})

        # accumulate keywords count into total count
        for kw in ret:
            countpair = restotal.get(kw)
            if countpair is None:
                restotal.update({kw: [0, 0]})

            restotal[kw][0] += ret[kw][0]
            restotal[kw][1] += ret[kw][1]

        #
    # output file for the total result
    #
    fdtotal = open(os.path.join(baseout, TOTAL_WAKE_SUMMARY_NAME), mode='w', encoding='UTF-8')
    print()
    tmpstr = 'Scene\t\t'
    idx = 0
    for kw in sorted(restotal.keys()):
        tmpstr += kw
        if idx < len(restotal) - 1:
            tmpstr += '\t\t'
        idx += 1
    print(tmpstr)
    fdtotal.write(tmpstr + '\n')
    for scene in sorted(resscene.keys()):
        tmpstr = scene + '\t\t'
        res = resscene[scene]
        idx = 0
        for kw in sorted(restotal.keys()):
            countpair = res.get(kw)
            if countpair is None:
                tmpstr += '----'
            else:
                expected_count = countpair[1] if countpair[1] > 0 else 1
                tmpstr += '{:.1%}'.format(countpair[0] / expected_count) + ' (' + str(countpair[0]) + '/' + str(
                    countpair[1]) + ')'
            if idx < len(restotal) - 1:
                tmpstr += '\t\t'
            idx += 1
        print(tmpstr)
        fdtotal.write(tmpstr + '\n')

    #
    # print total recall
    #
    tmpstr = 'Total Recall:\n'
    for kw in sorted(restotal.keys()):
        tmpstr += kw + ': '
        countpair = restotal[kw]
        expected_count = countpair[1] if countpair[1] > 0 else 1
        tmpstr += '{:.1%}'.format(countpair[0] / expected_count) + ' (' + str(countpair[0]) + '/' + str(
            countpair[1]) + ')\n'
    print()
    print(tmpstr)
    fdtotal.write('\n')
    fdtotal.write(tmpstr)
    fdtotal.flush()
    fdtotal.close()


# baseannot = 'D:/origin_pos_list'
# basein = 'D:/feout'
# baseout = 'D:/2_fewake_eval'
if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write("Keyword spotting evaluation.\n")
        sys.stderr.write("Usage:\n")
        sys.stderr.write("KWSEval <annotation directory> <input directory> <output directory>\n")
        exit(-1)
    baseannot = sys.argv[1]
    basein = sys.argv[2]
    baseout = sys.argv[3]

    kws_eval(baseannot, basein, baseout)
