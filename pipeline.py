# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import math
import os
import sys

import yaml
from modelscope.trainers import build_trainer
from modelscope.utils.logger import get_logger

from download import Musan, AIShell2, DNSChallenge
from evaluate.batch_roc import batch_roc, check_conf
from evaluate.roc_sort import roc_sort

MODEL_ID = 'damo/speech_dfsmn_kws_char_farfield_16k_nihaomiya'
REVISION = 'v1.1.0'

BASE_POS_DATA = 'data_pos'
BASE_ANNO = 'data_annotation'
BASE_NEG_DATA = 'data_neg'

MAX_EPOCHS = 500
BASETRAIN_RATIO_FIRST = 0.5
BASETRAIN_RATIO_SECOND = 0.05

# max false alarm (times/hour)
FAR_TH = 0.2
# max false rejection rate threshold
FRR_TH = 0.1

logger = get_logger(log_file='train.txt', log_level=logging.DEBUG)


def main(cfg, download_dir, base_only):
    """
    后续还需要增加的配置项：
    唤醒词：kws_decode_desc, kws_level
    输入通道： numins, chorder, 可能还需要 nummics, numrefs, bf_algorithm
    """
    check_conf(cfg, download_dir)
    work_dir = cfg['work_dir']
    if download_dir:
        os.makedirs(download_dir, exist_ok=True)
        prepare_data(download_dir, cfg)
        with open(os.path.join(work_dir, 'config_updated.yml'), 'w') as f:
            yaml.dump(cfg, f)

    first_train_dir = os.path.join(work_dir, 'first')
    os.makedirs(first_train_dir, exist_ok=True)
    first_epoch_num = MAX_EPOCHS
    if 'max_epochs' in cfg:
        first_epoch_num = cfg['max_epochs']
    train(cfg, first_train_dir, max_epochs=first_epoch_num)
    model_pth_path = validate(cfg, work_dir, first_train_dir)
    logger.info(f'model path: {model_pth_path}')

    if base_only:
        return
    second_train_dir = os.path.join(work_dir, 'second')
    os.makedirs(second_train_dir, exist_ok=True)
    # 通过动态计算关联前后两次训练的轮数，目前有两种习惯配置:
    # 1) base_rate=0.05, second_epoch_num=100
    # 2) base_rate=0.1, second_epoch_num=200
    second_epoch_num = int(first_epoch_num * BASETRAIN_RATIO_SECOND * 4)
    train(cfg,
          second_train_dir,
          single_rate=BASETRAIN_RATIO_SECOND,
          max_epochs=second_epoch_num,
          model_bin=model_pth_path)
    model_pth_path = validate(cfg, work_dir, second_train_dir)
    logger.info(f'model path: {model_pth_path}')


def prepare_data(download_dir, cfg):
    """ 下载开源数据，生成音频列表和配置
    目标是每个列表中不同开源数据的取用时长相同
    由于训练程序是按配置比例选取音频文件数，而每个数据集的音频文件长度不同，所以配置中的比例并不相同
    """
    musan = Musan(download_dir)
    musan.fetch()
    dns = DNSChallenge(download_dir)
    dns.fetch()
    aishell = AIShell2(download_dir)
    aishell.fetch()
    neg_list = (aishell.list_files['all'], dns.list_files['clean'])
    merge_conf(cfg, 'train_neg_list', neg_list)
    ref_list = (aishell.list_files['all'] + ',1.8',
                musan.list_files['music'] + ',0.1',
                musan.list_files['speech'] + ',0.1')
    merge_conf(cfg, 'train_ref_list', ref_list)
    merge_conf(cfg, 'train_interf_list', ref_list)
    noise_list = (aishell.list_files['all'] + ',0.6',
                                 dns.list_files['noise'] + ',0.2',
                                 musan.list_files['all'] + ',0.2')
    merge_conf(cfg, 'single_noise1_list', noise_list)
    merge_conf(cfg, 'multi_noise1_list', noise_list)


def merge_conf(cfg, name, data):
    if name in cfg:
        cfg[name].extend(data)
    else:
        cfg[name] = data


def validate(cfg, work_dir, model_dir):
    # 把top模型转换为txt格式，写入dump_dir
    dump_dir = model_dir + '_txt'
    model2txt(model_dir, dump_dir)
    # 对排序top 的模型测试roc
    logger.info('=' * 80)
    logger.info('Start batch computing ROC...')
    roc_dir = model_dir + '_roc'
    os.makedirs(roc_dir, exist_ok=True)
    batch_roc(work_dir, dump_dir, cfg, roc_dir)
    top_model = pick_top_model(cfg, roc_dir)

    model_txt_name = top_model[0]
    model_txt_path = os.path.join(dump_dir, model_txt_name)
    model_pth_name = model_txt_name[7:].replace('.txt', '.pth')
    model_pth_path = os.path.join(model_dir, model_pth_name)
    logger.info(f'model txt path: {model_txt_path}')
    logger.info(f'model kw frr and level: {top_model[1]}')
    logger.info(f'model path: {model_pth_path}')
    return model_pth_path


def compute_num_syn(cfg):
    num_syn = 0
    for kw_conf in cfg['keywords']:
        class_list = kw_conf.split(',')[1:]
        for c in class_list:
            c_number = int(c.strip())
            if c_number > num_syn:
                num_syn = c_number
    num_syn += 1
    return num_syn


def train(cfg, train_dir, single_rate=BASETRAIN_RATIO_FIRST, max_epochs=None, model_bin=None):
    train_pos_list = '\n'.join(cfg['train_pos_list'])
    train_neg_list = '\n'.join(cfg['train_neg_list'])
    single_noise1_list = '\n'.join(cfg['single_noise1_list'])
    multi_noise1_list = '\n'.join(cfg['multi_noise1_list'])
    train_interf_list = '\n'.join(cfg['train_interf_list'])
    train_ref_list = '\n'.join(cfg['train_ref_list'])
    if 'train_noise2_list' in cfg:
        train_noise2_type = '1'
        train_noise1_ratio = '0.2'
        train_noise2_list = '\n'.join(cfg['train_noise2_list'])
    else:
        train_noise2_type = '0'
        train_noise1_ratio = '1.0'
        train_noise2_list = ''
    base_dict = dict(
        train_pos_list=train_pos_list,
        train_neg_list=train_neg_list,
        train_noise1_list=single_noise1_list)
    fintune_dict = dict(
        train_pos_list=train_pos_list,
        train_neg_list=train_neg_list,
        train_noise1_list=multi_noise1_list,
        train_noise1_ratio=train_noise1_ratio,
        train_noise2_type=train_noise2_type,
        train_noise2_list=train_noise2_list,
        train_interf_list=train_interf_list,
        train_ref_list=train_ref_list)
    custom_conf = dict(
        basetrain_easy=base_dict,
        basetrain_normal=base_dict,
        basetrain_hard=base_dict,
        finetune_easy=fintune_dict,
        finetune_normal=fintune_dict,
        finetune_hard=fintune_dict)
    workers = cfg['workers']
    # 组装训练需要的配置项
    kwargs = dict(
        model=MODEL_ID,
        work_dir=train_dir,
        model_revision=REVISION,
        workers=workers,
        single_rate=single_rate,
        custom_conf=custom_conf)
    num_syn = compute_num_syn(cfg)
    # 默认训练一个4字唤醒词时，模型输出维度为5，即模型4个字 + 其他
    if num_syn != 5:
        kwargs['num_syn'] = num_syn
    if max_epochs:
        kwargs['max_epochs'] = max_epochs
    if 'val_iters_per_epoch' in cfg:
        kwargs['val_iters_per_epoch'] = cfg['val_iters_per_epoch']
    if 'train_iters_per_epoch' in cfg:
        kwargs['train_iters_per_epoch'] = cfg['train_iters_per_epoch']
    if model_bin:
        kwargs['model_bin'] = model_bin
    logger.info('=' * 80)
    logger.info('Start training...')
    trainer = build_trainer('speech_dfsmn_kws_char_farfield', default_args=kwargs)
    trainer.train()
    logger.info('Training finished.')


def model2txt(model_dir, txt_dir):
    # 用扩展名过滤出模型文件，按loss排序
    model_files = [i for i in os.listdir(model_dir) if i.endswith('.pth')]
    top_n = math.ceil(len(model_files) / 10.0)
    # the length of the file name is fixed, so use absolute offset to get the loss of validation
    # checkpoint_0011_loss_train_0.5757_loss_val_0.5313.pth
    # model_files = sorted(model_files, key=lambda i: float(i[43:49]))
    f = 'loss_val_'
    model_files = sorted(model_files, 
        key=lambda a: float(a[a.find(f) + len(f):a.find(f) + len(f)+6]))
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    for i in range(min(len(model_files), top_n)):
        full_path = os.path.join(model_dir, model_files[i])
        logger.info(full_path)
        # 因为排序后不会取很多，两位数就够表示了
        dump_path = os.path.join(txt_dir, f'top_{i + 1:02}_{model_files[i][:-4]}.txt')
        cmd = f'python print_model.py {full_path} > {dump_path}'
        os.system(cmd)


def pick_top_model(cfg, roc_dir):
    roc_sort_dir = roc_dir + '_sort'
    os.makedirs(roc_sort_dir, exist_ok=True)
    kw_conf_list = cfg['keywords']
    main_kw = None
    if 'main_keyword' in cfg:
        main_kw = cfg['main_keyword']
    max_far = FAR_TH
    if 'max_far' in cfg:
        max_far = float(cfg['max_far'])
    sorted_models = roc_sort(roc_dir,
                             roc_sort_dir,
                             far_th=max_far,
                             frr_th=FRR_TH,
                             kw=main_kw)
    # sort by min_frr, sorted_models:[(model_name, {kw: (min_frr, thres)}),...]
    if main_kw:
        sorted_models = sorted(sorted_models, key=lambda t: float(t[1][main_kw][0]))
        top_model = sorted_models[0]
    else:
        top_model = sorted_models[0]
        min_sum = 1000
        for model_result in sorted_models:
            sum_min_frr = 0.0
            for min_frr, thres in model_result[1].values():
                sum_min_frr += float(min_frr)
            if sum_min_frr < min_sum:
                min_sum = sum_min_frr
                top_model = model_result
    return top_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KWS training script')
    parser.add_argument('config')
    parser.add_argument('--remote_dataset', help='download remote dataset for training')
    parser.add_argument('-1', '--base_only', help='only run base training',
                        action='store_true')
    parser.add_argument('-d', '--debug', help='print debug log',
                        action='store_true')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    conf_file = args.config
    if not os.path.exists(conf_file):
        logger.error('Config file "%s" is not exist!', conf_file)
        sys.exit(-1)
    logger.info('Loading config from %s', conf_file)
    with open(conf_file, encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    main(conf, args.remote_dataset, args.base_only)
