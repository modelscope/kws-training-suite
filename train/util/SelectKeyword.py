'''
Select specified keyword list according to asr result.
'''

import os
import sys


if len(sys.argv) != 3:
    sys.stderr.write("Usage:\n");
    sys.stderr.write("SelectKeyword <asr result> <keyword>\n")
    exit(-1)


with open(sys.argv[1], 'r', encoding = 'UTF-8') as file:
    lines = file.readlines()

for l in lines:
    l = l.strip()
    ss = l.split(': ')
    if len(ss) >= 2 and sys.argv[2] in ss[1]:
        print(ss[0])
