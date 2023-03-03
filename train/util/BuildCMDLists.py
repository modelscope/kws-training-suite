'''
Build commands audio data lists.

Copyright: 2022-05-16 yueyue.nyy
'''

import os
import sys


if len(sys.argv) < 2:
    sys.stderr.write('Build commands audio data lists.\n')
    sys.stderr.write('Usage:\n')
    sys.stderr.write('BuildCMDList <basein>\n')
    exit(-1)

basein = sys.argv[1]


for dir in os.listdir(basein):
    name = os.path.split(dir)[1]
    scene = os.path.abspath(os.path.join(basein, name))
    if (not os.path.exists(scene)) or (not os.path.isdir(scene)):
        continue
    
    cmd = 'find ' + os.path.abspath(scene) + ' -name \"*.wav\" >' + os.path.join(basein, name + '.txt')
    retval = os.system(cmd)
    
    if retval == 0:
        print('DONE: ' + dir)
    else:
        print('FAILED: ' + dir)
