#!/bin/bash
if [ ! -n "$1" ] ; then
    nvidia-smi
    echo
    echo "You should give GPU No."
else
    chmod 755 bin/SoundConnect
    export CUDA_VISIBLE_DEVICES=$1
    export PYTHONPATH=.:$PYTHONPATH
    shift
    cfg=config.yml
    if [ -n "$1" ] ; then
        cfg=$1
    fi
    echo "Use config file: $cfg"
    shift
    python pipeline.py $cfg $*
fi
