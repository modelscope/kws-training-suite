# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re

from modelscope.utils.audio.audio_utils import update_conf
from modelscope.utils.logger import get_logger

from evaluate.util.KWSEval import kws_eval
from evaluate.util.KWSROC import kws_roc, get_myprint

SC_PATH = './bin/SoundConnect'
LIB_AFEJAVA_PATH = './lib/AFEJava.jar'
LIB_LIBSIGNAL_PATH = './lib/libsignal.jar'
BASE_POS_EXPERIMENT = 'TianGongExperiment_pos'
BASE_NEG_EXPERIMENT = 'TianGongExperiment_neg'

# neg dataset length ratio, compressed by feature extraction
NEG_LEN_RATIO = 1.0

logger = get_logger()


def batch_roc(work_dir, model_path, fe_conf, roc_dir):
    pos_data_dir = fe_conf['test_pos_data_dir']
    pos_anno_dir = fe_conf['test_pos_anno_dir']
    neg_data_dir = fe_conf['test_neg_data_dir']
    neg_anno_dir = os.path.join(work_dir, os.path.split(neg_data_dir)[1] + '_anno')
    eval_dir = roc_dir + '_eval'

    # generate negative annotation
    for wav in list_files(neg_data_dir, '.wav', abs_path=False):
        txt = wav.replace('.wav', '.txt')
        anno_path = os.path.join(neg_anno_dir, txt)
        anno_sub_dir = os.path.dirname(anno_path)
        os.makedirs(anno_sub_dir, exist_ok=True)
        try:
            open(anno_path, 'ab', 0).close()
        except OSError:
            pass

    threads = fe_conf['workers']
    base_pos_dir = os.path.join(work_dir, BASE_POS_EXPERIMENT)
    base_neg_dir = os.path.join(work_dir, BASE_NEG_EXPERIMENT)
    for d in ('0_input', '1_cut', '2_fewake_eval', '2_wake', '3_asr', '4_asr_eval', '5_chselection_eval', '6_voiceprint', '7_vad_eval'):
        os.makedirs(os.path.join(base_pos_dir, d), exist_ok=True)
        os.makedirs(os.path.join(base_neg_dir, d), exist_ok=True)

    fe_conf_path = os.path.join(os.path.dirname(__file__), 'conf', 'sc.conf')
    for mp in list_files(model_path, '.txt'):
        tmpconfpath = os.path.join(work_dir, 'tmp.conf')

        my_conf = {**fe_conf, 'kws_level': '0.0', 'kws_model': mp}
        update_conf(fe_conf_path, tmpconfpath, my_conf)

        name = os.path.split(mp)[1]
        pos_result_dir = os.path.join(eval_dir, name, 'pos')
        neg_result_dir = os.path.join(eval_dir, name, 'neg')
        os.makedirs(pos_result_dir, exist_ok=True)
        os.makedirs(neg_result_dir, exist_ok=True)
        eval_on_rough_anno(tmpconfpath, base_pos_dir, pos_anno_dir, pos_anno_dir, threads, pos_result_dir)
        eval_on_rough_anno(tmpconfpath, base_neg_dir, neg_anno_dir, neg_data_dir, threads, neg_result_dir)
        # eval_on_manual_anno(tmpconfpath, base_neg_dir, neg_data_dir, threads)

        model_roc_file = os.path.join(roc_dir, os.path.split(mp)[1])

        kws_roc(pos_result_dir,
                neg_result_dir,
                os.path.join(base_neg_dir, '0_input'),
                NEG_LEN_RATIO,
                get_myprint(model_roc_file))
        print('DONE: ' + mp)


def list_files(path, ext, abs_path=True):
    path_length = len(path)
    if not path.endswith('/'):
        path_length += 1
    for root, dirs, files in os.walk(path, followlinks=True):
        for file in files:
            if file.endswith(ext):
                if abs_path:
                    yield os.path.join(root, file)
                else:
                    yield os.path.join(root, file)[path_length:]


def generate_fe_conf(confin, confout, modelp):
    """ generate front-end configure file

    Args:
        confin:                 input conf path
        confout:                output conf path
        modelp:                 model path
    """
    with open(confin, 'r', encoding='UTF-8') as fd:
        lines = fd.readlines()
    fd = open(confout, 'w', encoding='UTF-8')
    for line in lines:
        m = re.match('kws_model_base = (.+)', line)
        if m:
            fd.write('kws_model_base = ' + os.path.abspath(modelp) + '\n')
        elif line.startswith('kws_level ='):
            fd.write('kws_level = 0.0\n')
        else:
            fd.write(line)
    fd.flush()
    fd.close()


def eval_on_manual_anno(conf_path, base_experiment, base_data, threads):
    cmd = 'rm -rf ' + base_experiment + '/0_input/*'
    os.system(cmd)
    cmd1 = 'java -cp ' + LIB_AFEJAVA_PATH + ' cc.soundconnect.toolkit.AFEBatch ' + SC_PATH
    cmd1 += ' ' + conf_path
    cmd1 += ' ' + base_data
    cmd1 += ' ' + base_experiment + '/0_input/ --numths ' + str(threads)
    logger.info('cmdline: %s', cmd1)
    os.system(cmd1)
    cmd1 = 'java -cp ' + LIB_AFEJAVA_PATH + ':' + LIB_LIBSIGNAL_PATH
    cmd1 += ' project.tiangong.TianGongExperiment ' + base_experiment + '/wake.conf'
    logger.info('cmdline: %s', cmd1)
    os.system(cmd1)


def eval_on_rough_anno(conf_path, base_experiment, pos_anno, base_data, threads, eval_result_dir):
    """
    perform positive experiment
    confpath:               fe configure file path
    """
    cmd = 'rm -rf ' + base_experiment + '/0_input/*'
    os.system(cmd)
    cmd1 = 'java -cp ' + LIB_AFEJAVA_PATH + ' cc.soundconnect.toolkit.AFEBatch ' + SC_PATH
    cmd1 += ' ' + conf_path
    cmd1 += ' ' + base_data
    cmd1 += ' ' + base_experiment + '/0_input/ --numths ' + str(threads)
    logger.info('cmdline: %s', cmd1)
    os.system(cmd1)
    kws_eval(pos_anno, os.path.join(base_experiment, '0_input'), eval_result_dir)

