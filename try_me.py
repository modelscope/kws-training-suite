# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import os

from modelscope.utils.audio.audio_utils import update_conf

from download import RemoteDataSet

ANNO_LIST = 'anno_pos/01_easy/test.txt'
POS_WAV_PATH = 'origin_pos/01_easy/multi_easy_0001_2mic.wav'


class HiMia(RemoteDataSet):
    HTTP_PATH = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/dataset/himia.zip'
    NAME = 'himia'
    SUB_LISTS = ('clean', 'noise', 'himia-align')


def main(threads, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    cfg = {}
    # download, extract and create wav list
    himia = HiMia(work_dir)
    himia.fetch()
    pak_dir = himia.local_dir

    # 生成pos标注
    with open(os.path.join(pak_dir, ANNO_LIST), 'w') as f:
        f.write(f'{os.path.join(pak_dir, POS_WAV_PATH)} 0_ni_hao_mi_ya 57\n')

    # create conf
    cfg['work_dir'] = work_dir
    cfg['workers'] = str(threads)
    template_conf = os.path.join(pak_dir, 'config.tpl')
    conf_file = os.path.join(work_dir, 'config.yml')
    update_conf(template_conf, conf_file, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KWS training script')
    parser.add_argument('threads', type=int)
    parser.add_argument('work_dir')
    args = parser.parse_args()

    main(args.threads, args.work_dir)
