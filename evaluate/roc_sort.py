# Copyright (c) Alibaba, Inc. and its affiliates.
# Sort roc curves.

import os
import sys
import numpy as np
import shutil


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


def loadROC(path):
    """ load roc curves
    path:                   input file path
    return:                 {kw: [far, frr, thres]}
    """
    retdict = {}
    
    with open(path, 'r', encoding = 'UTF-8') as fd:
        lines = fd.readlines()
    
    for ts in lines:
        ts = ts.strip()
        if len(ts) <= 0:
            continue
        
        sts = ts.split()
        if len(sts) == 1:
            rocl = retdict.get(ts)
            if rocl is None:
                retdict.update({ts: []})
                rocl = retdict[ts]
        else:
            fl = []
            for snumber in sts:
                fl.append(float(snumber))
            
            rocl.append(fl)
    
    return retdict


def rocArea(roc_dict, far_th, frr_th):
    """ calculate the area of the roc curve
    roc_dict: {kw: [far, frr, thres]}
    far_th: the threshold of FAR
    frr_th: the threshold of FRR
    """
    area = 0.0
    min_frr = 1.0
    thres = 0
    for i in range(1, len(roc_dict)):
        if roc_dict[i][0] >= far_th:
            break
        if roc_dict[i][1] < min_frr:
            min_frr = roc_dict[i][1]
            thres = roc_dict[i][2]
        area += (roc_dict[i - 1][1] + roc_dict[i][1]) * (roc_dict[i][0] - roc_dict[i - 1][0]) / 2.0
    
    # frr not good
    if roc_dict[-1][1] > frr_th or roc_dict[-1][1] < 0:
        area += 100.0
    
    return area, min_frr, thres


def roc_sort(basein, baseout, far_th, frr_th, kw=None):
    """ 计算roc曲线在坐标图上的面积，按面积从小到大排序

    Args:
        basein: 输入目录
        baseout: 输出目录
        far_th: 每小时误唤醒次数阈值，范围[0, MAX]，计算面积时只取误唤醒次数小于阈值的
        frr_th: 未唤醒率阈值，如果未唤醒率较高也直接放弃。未唤醒率=1-唤醒率，范围[0, 1]
        kw: 指定唤醒词

    Returns:
        list:[(model_name, {kw: (min_frr, thres)}),...]
    """
    # calculate roc total_area
    rocl = listFiles(basein, ['.txt'])
    areal = np.zeros((len(rocl),), dtype='float32')
    thresl = []
    for i in range(len(rocl)):
        rocdict = loadROC(rocl[i])
        kw_thres = {}
        if kw is None:
            total_area = 0.0
            for k in rocdict.keys():
                area, min_frr, thres = rocArea(rocdict[k], far_th, frr_th)
                total_area += area
                kw_thres[k] = (min_frr, thres)
            areal[i] = total_area / len(rocdict)
        else:
            area, min_frr, thres = rocArea(rocdict[kw], far_th, frr_th)
            areal[i] = area
            kw_thres = {kw: (min_frr, thres)}
        thresl.append(kw_thres)
    # sort roc total_area
    idxl = np.argsort(areal)
    # output results
    sorted_models = []
    for topi in range(len(rocl)):
        rocpath = rocl[idxl[topi]]
        name = os.path.split(rocpath)[1]
        outpath = os.path.join(baseout, 'top_{:03d}_auc_{}_'.format(topi + 1, areal[idxl[topi]]) + name)
        shutil.copyfile(rocpath, outpath)
        sorted_models.append((name, thresl[idxl[topi]]))
    return sorted_models


if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.stderr.write('Sort roc.\n');
        sys.stderr.write('Usage:\n');
        sys.stderr.write('ROCSort <basein> <baseout> <far threshold (times/hour)> <frr threshold> [keyword]\n')
        exit(-1)

    basein = sys.argv[1]
    baseout = sys.argv[2]
    far_th = float(sys.argv[3])
    frr_th = float(sys.argv[4])
    kw = None
    if len(sys.argv) >= 6:
        kw = sys.argv[5]

    roc_sort(basein, baseout, far_th, frr_th, kw)
