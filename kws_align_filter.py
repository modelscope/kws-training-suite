import argparse
import math
import os
import shutil
import sys


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to further filter kws force aligned files.')
    parser.add_argument('condition', help='the condition to filter files, eg:\n'
                                          '0.80 means just select files that confidence >= 0.80\n'
                                          '80%  means order files by confidence and select the top 80%')
    parser.add_argument('input', help='path of the audio list file or the directory storing audio files')
    parser.add_argument('-o', '--out_dir', help='output directory, default: [input]-filter')
    args = parser.parse_args()

    basein = args.input
    if args.out_dir:
        baseout = args.out_dir
    else:
        if basein[-1] == '/':
            basein = basein[:-1]
        baseout = basein + '-filter'

    by_confidence = True
    if args.condition[-1] == '%':
        by_confidence = False
        condition = float(args.condition[:-1])/100
        if condition > 1 or condition < 0:
            print(f'Error: invalid argument condition "{args.condition}"')
            sys.exit(-1)
    else:
        condition = float(args.condition)

    all_files = listFiles(args.input, ['.wav'])
    file_count = len(all_files)
    file_data = [(f, float(f[-8:-4])) for f in all_files]
    file_data.sort(key=lambda x: x[1], reverse=True)
    selected_count = file_count
    if by_confidence:
        confidence = condition
        for i, data in enumerate(file_data):
            if data[1] < condition:
                selected_count = i
                break
    else:
        selected_count = math.ceil(file_count * condition)
        confidence = file_data[selected_count-1][1]

    r = input(f'Total file count {file_count}\n'
              f'Will copy {selected_count} files to {baseout}/\n'
              f'Selected kws confidence >= {confidence}. Proceed(Y/n)? ')
    if r.lower() in ('n', 'no'):
        sys.exit(-1)

    os.makedirs(baseout)
    for f, c in file_data[:selected_count]:
        dst = os.path.join(baseout, os.path.basename(f))
        shutil.copyfile(f, dst)
