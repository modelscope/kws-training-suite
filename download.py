# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import shutil
import sys
import requests
import os


TIMEOUT = 10
MAX_RETRY = 10
# 屏蔽warning信息
requests.packages.urllib3.disable_warnings()


class RemoteDataSet:
    """ 从远端下载数据集并解压生成文件列表"""
    def __init__(self, local_root, clean=False):
        self.local_root = local_root
        self.clean = clean
        self.local_dir = os.path.join(self.local_root, self.NAME)
        self.local_zip = os.path.join(self.local_root, self.NAME+'.zip')
        self.list_files = {}

    def fetch(self):
        # 如果需要清除，先把本地文件全删除
        if self.clean:
            if os.path.exists(self.local_zip):
                os.remove(self.local_zip)
            if os.path.exists(self.local_dir):
                shutil.rmtree(self.local_dir)
        download(self.HTTP_PATH, self.local_zip)
        self.create_lists()

    def create_lists(self):
        all_name = 'all'
        all_list_file = os.path.join(self.local_dir, f'{all_name}.txt')
        self.list_files[all_name] = all_list_file
        # 如果文件列表已存在则跳过解压和生成列表步骤
        if os.path.exists(all_list_file):
            print(f'Find file list:{all_list_file}, skip extracting and scanning')
            if hasattr(self, 'SUB_LISTS'):
                for name in self.SUB_LISTS:
                    list_file = os.path.join(self.local_dir, f'{name}.txt')
                    self.list_files[name] = list_file
        else:
            # 使用系统命令解压比较快
            cmd = f'cd {self.local_root}; unzip {self.local_zip}'
            os.system(cmd)
            with open(all_list_file, 'w') as all_file:
                if hasattr(self, 'SUB_LISTS'):
                    for name in self.SUB_LISTS:
                        list_file = os.path.join(self.local_dir, f'{name}.txt')
                        self.list_files[name] = list_file
                        sub_dir = os.path.join(self.local_dir, name)
                        with open(list_file, 'w') as f:
                            for wav_file in list_files(sub_dir, '.wav'):
                                f.write(wav_file)
                                f.write('\n')
                                all_file.write(wav_file)
                                all_file.write('\n')
                else:
                    for wav_file in list_files(self.local_dir, '.wav'):
                        all_file.write(wav_file)
                        all_file.write('\n')


class DNSChallenge(RemoteDataSet):
    HTTP_PATH = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/dataset/dns_challenge.zip'
    NAME = 'dns_challenge'
    SUB_LISTS = ('clean', 'noise')


class Musan(RemoteDataSet):
    HTTP_PATH = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/dataset/musan.zip'
    NAME = 'musan'
    SUB_LISTS = ('music', 'speech', 'noise')


class AIShell2(RemoteDataSet):
    HTTP_PATH = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/dataset/AISHELL2.zip'
    NAME = 'AISHELL2'


def download(url, file_path):
    # 重试计数
    count = 0
    # 第一次请求是为了得到文件总大小
    r1 = requests.get(url, stream=True, verify=False)
    total_size = int(r1.headers['Content-Length'])

    # 判断本地文件是否存在，存在则读取文件数据大小
    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)  # 本地已经下载的文件大小
    else:
        temp_size = 0

    if temp_size >= total_size:
        print(f'Skip downloading {file_path}\n'
              f'remote file size: {total_size}, local file size: {temp_size}')
        return file_path

    # 开始下载
    while count < MAX_RETRY:
        if count != 0:
            temp_size = os.path.getsize(file_path)
        # 文件大小一致，跳出循环
        if temp_size >= total_size:
            print(f'Download is finished.')
            break
        count += 1
        print(
            "第[{}]次下载文件,已经下载数据大小:[{}],应下载数据大小:[{}]".format(
                count, temp_size, total_size))
        try:
            # 重新请求网址，加入新的请求头的
            # 核心部分，这个是请求下载时，从本地文件已经下载过的后面下载
            headers = {"Range": f"bytes={temp_size}-"}
            r = requests.get(url, timeout=TIMEOUT, stream=True, verify=False, headers=headers)
            with open(file_path, "ab") as f:
                if count != 1:
                    f.seek(temp_size)
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        temp_size += len(chunk)
                        f.write(chunk)
                        f.flush()
                        ###这是下载实现进度显示####
                        done = int(50 * temp_size / total_size)
                        sys.stdout.write("\r[%s%s] %d%%" % (
                            '█' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                        sys.stdout.flush()
            print("\n")
        except requests.exceptions.ConnectionError as e:
            print('\n')
            print(e)

    return file_path


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open source dataset downloader')
    parser.add_argument('download_dir')
    parser.add_argument('--clean', help='clean all local data before download',
                        action='store_true')
    args = parser.parse_args()

    if args.clean:
        r = input(f'Are you sure to DELETE local data of these datasets and download them again? (yes/NO)')
        if r.lower() not in ('y', 'yes'):
            sys.exit(-1)

    os.makedirs(args.download_dir, exist_ok=True)

    musan = Musan(args.download_dir, args.clean)
    musan.fetch()
    aishell = AIShell2(args.download_dir, args.clean)
    aishell.fetch()
    dns = DNSChallenge(args.download_dir, args.clean)
    dns.fetch()
