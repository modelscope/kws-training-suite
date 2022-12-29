# Copyright (c) Alibaba, Inc. and its affiliates.

import os, sys
import matplotlib.pyplot as plt


def load_one_roc_txt(fid):
    lines = fid.readlines()
    fa = list()
    wk = list()
    thr = list()
    for l in lines:
        tmps = l.strip().split(" ")
        if len(tmps) < 3:
            continue
        fa.append(float(tmps[0]))
        wk.append(float(tmps[1]))
        thr.append(float(tmps[2]))
    return fa, wk, thr


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} roc_file [other_roc_file_to_compare]')
        sys.exit(-1)
    plt.figure()
    roc_file = sys.argv[1]
    fid = open(roc_file)
    fa1, wk1, thr1 = load_one_roc_txt(fid)
    plt.plot(fa1, wk1, 'r^')

    plt.xlabel('false alarm times/per hour')
    plt.ylabel('wake rejection rate')

    if len(sys.argv) > 2:
        ref_fid = open(sys.argv[2])
        fa2, wk2, thr2 = load_one_roc_txt(ref_fid)
        plt.plot(fa2, wk2, 'bs')

    basename = os.path.basename(roc_file)
    dirname = os.path.dirname(roc_file)
    name, ext = os.path.splitext(basename)
    out_fullname = os.path.join(dirname, f'{name}.png')
    print(out_fullname)
    plt.savefig(out_fullname)
