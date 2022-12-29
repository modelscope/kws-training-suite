## 概览
### 软件架构
![](https://intranetproxy.alipay.com/skylark/lark/0/2022/jpeg/2639/1670305061456-81c162b4-c564-4744-b45c-160c4087230c.jpeg)
### 输入输出
![](https://intranetproxy.alipay.com/skylark/lark/0/2022/jpeg/2639/1670304405390-09fd554c-d272-4611-9174-db8244bf15a4.jpeg)

### 训练流程
![](https://intranetproxy.alipay.com/skylark/lark/0/2022/jpeg/2639/1669865969613-8e2651a3-2286-40fa-bd18-4304b386ab21.jpeg)

## 数据准备
### 数据分类
训练需要的数据大致分如下几类，格式除特殊说明外要求为采样率16000Hz的单声道PCM编码.wav文件。

- 带标注的唤醒词音频
- 负样本音频
- 噪声音频（单通道/多通道）

其中负样本音频和单通道噪声音频，可以使用套件内置的自动下载开源数据的功能，提供的数据集有三个：AISHELL2，DNS-Challenge，musan，数据总大小170G，打包成zip包约130G。
建议用户针对实际使用场景录制一些音频，如果完全依靠开源数据，训练出的模型在用户实际场景下很难达到理想的性能。

#### 唤醒词音频
唤醒词音频文件，通常是众包采集的背景安静，发音清晰的唤醒词语音。
数据量：至少需要 100 人 * 100 句 = 10000条数据，每一条单独保存一个文件。
数据量越多越好，总数据量相同的情况下，人数越多越好。
##### 数据打标
把音频中的唤醒词信息通过工具自动标注出来，供模型学习。
如下图，上半部分是唤醒词音频，下半部分是对应标注信息。
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/2639/1670412114735-aa18ea8f-72b9-43d0-bb17-5d8363ca3f16.png#clientId=ua84797a3-1e78-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=244&id=uf6abdb13&margin=%5Bobject%20Object%5D&name=image.png&originHeight=488&originWidth=621&originalType=binary&ratio=1&rotation=0&showTitle=false&size=37706&status=done&style=none&taskId=ua14540ae-9e73-4dd3-8081-aad584bbe85&title=&width=310.5)
进入kws-training-scripts目录，使用force_align.py命令，可以加-h参数看帮助，以下为示例：

- -t 表示并发工作线程数
- /data/wav 是存放唤醒词音频的目录，不同的唤醒词应该分在不同目录存放，每次只能处理一个唤醒词
- 天猫精灵 是唤醒词，每次只能指定一个
```python
python force_align.py -t 10 /data/wav 天猫精灵
```

##### 真实线上数据的利用
如果用户的唤醒词已经有在线服务，积累了一批线上真实唤醒音频，把这些数据加入训练很很有帮助。但是线上唤醒场景不可控，可能有很多杂音干扰，需要经过筛选之后才能使用。
推荐方法是先用少量唤醒词音频训练一个初级唤醒模型，利用此模型对线上音频做筛选和打标。

#### 负样本音频
不含唤醒词的清晰人声音频，可以从[开源数据库](http://www.openslr.org/)中获取。

#### 噪声音频
单通道噪声音频，可包括音乐，扫地机，吸尘器等各种噪声；也可以是设备中播放出的音乐、故事、电视节目等。
多通道噪声音频，应当是真实设备录制的多通道音频。
建议用户准备噪音音频时覆盖尽可能多的场景，每种场景的音频时长应当在8小时以上，并切分成时长1分钟的片段。
想要达到比较好的效果，建议准备常规噪声数据如电视节目，音乐等100小时以上；另外特殊场景，如扫地机的噪音，风噪声 要20小时以上。

### 准备音频文件列表
训练程序是通过音频文件列表读取数据的，以上各类音频文件在本地准备好以后，要分别生成音频文件列表。
列表是一个文本文件(.txt)，其中每一行是一条wav音频的本地绝对路径。

## 搭建环境
### 环境要求
#### 硬件配置：

- 64 CPU 48G内存 ——此为推荐值，配置越高训练越快
- 1 GPU(Tesla P4或以上)  6G显存
- 400G存储空间
#### 软件环境：

- CUDA >= 11.0
- Java SDK >= 8
- Python >= 3.7
- Pytorch >= 1.11
- ModelScope >= 1.1

以上配置支持60并发，整个训练流程约耗时3天。

参考数据：

- 60并发跑第一次500轮耗时35小时；第二次训练200轮预计10小时
- 测试速度取决于测试数据量，测试 50 个模型约 10小时；
- 总体预计60+小时

#### 网络环境
程序运行过程中需要从网络存储下载开源数据，连接ModelScope网站更新模型等数据，所以需要连接公开网络。
需要连接的域名如下：

```
*.aliyuncs.com
*.modelscope.cn
```

### 选项A. 推荐使用Docker镜像
使用ModelScope提供的docker镜像，上面已经预装好了模型训练所需的Python环境和ModelScope框架。

```
# CPU版本：
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.1.0
# GPU版本: 
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.1.0
```

#### 安装其他依赖
安装官方docker镜像中缺少的依赖：
```shell
apt-get update
apt-get install unzip
apt-get install openjdk-11-jdk
```
然后可直接开始“验证内置唤醒模型推理”

### 选项B. 手动安装ModelScope和相关依赖
#### Python环境配置
建议使用Anaconda配置Python环境，具体安装步骤可参考其[官方文档](https://docs.anaconda.com/anaconda/install)
执行如下Anaconda命令为ModelScope创建对应的python环境（要求python版本 >=3.7）

```python
conda create -n modelscope python=3.7
conda activate modelscope
```
检查python和pip命令是否切换到conda环境下：

```python
which python
# ~/anaconda3/envs/modelscope/bin/python
which pip
# ~/anaconda3/envs/modelscope/bin/pip
```

由于anaconda环境默认的pip版本较低，建议先升级到新版：

```python
python -m pip install --upgrade pip
```

#### 安装PyTorch
本模型已经在PyTorch 1.8~1.11下测试通过，可执行以下命令指定安装PyTorch v1.11：（如果下载安装速度较慢可指定阿里云、清华等国内pypi镜像）
```
pip install torch==1.11 torchaudio torchvision
```

#### 安装libsndfile1
本模型的pipeline中使用了三方库SoundFile进行wav文件处理，**在Linux系统上用户需要手动安装SoundFile的底层依赖库libsndfile**，在Windows和MacOS上会自动安装不需要用户操作。详细信息可参考[SoundFile官网](https://github.com/bastibe/python-soundfile#installation)。以Ubuntu系统为例，用户需要执行如下命令:

```shell
sudo apt-get update
sudo apt-get install libsndfile1
```

#### 安装ModelScope和语音模型相关依赖

```
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

### 验证内置唤醒模型推理
如果以上步骤执行正常，环境安装成功的话，以下python代码应该能运行通过，并且打印出5次唤醒信息。
**注意：**以下代码运行时需要从modelscope网站下载模型数据，因此需要确保网络正常。

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


kws = pipeline(
    Tasks.keyword_spotting,
    model='damo/speech_dfsmn_kws_char_farfield_16k_nihaomiya')
# you can also use local file path
result = kws('https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/3ch_nihaomiya.wav')
print(result)
```

### 准备训练套件
本训练套件在ModelScope的模型训练能力之外封装了数据打标，配置，模型转换，测试，训练流程组织等功能，使用户可以一键完成训练。

#### 训练mini模型
训练套件提供了**try_me.py**脚本，可以自动下载一个不到200M大小的数据包，并生成对应的配置文件。使用这个配置文件启动训练，可以在1小时内完成训练流程，得到一个可以在安静场景下唤醒的模型，唤醒词是“你好米雅”。
运行命令如下，请把其中的`/your/test_dir`替换为您的真实路径，下载和生成的所有数据都会保存在其中。

```shell
# 进入训练套件目录
cd kws-training-scripts
# 运行脚本，准备数据和配置文件
# 参数threads指定训练时的并发线程数，请根据您实际可用的cpu数量配置
python try_me.py threads /your/test_dir
# 运行训练套件
python pipeline.py -1 /your/test_dir/config.yml
```

##### 运行成功后输出信息如下：
前面很多行是模型唤醒率(51/57)和误唤醒率(5/0)测试信息，正式训练时会从几十个模型中测试挑选最佳模型，会输出很多类似信息，一般不需要关注。
红框中是最终选择的最优模型信息：

- .pth是pytorch格式保存的模型，可以使用此模型继续训练
- .txt是唤醒工具使用的模型参数文件
- model kw frr and level一行表示
   - 建议此模型使用时唤醒阈值设置为`0.86`
   - 此时唤醒词'0_ni_hao_mi_ya'在所有测试场景中的综合拒识率为`0.14035087719298245`，换算成唤醒率约为`1 - 0.14035087719298245 = 0.86`

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/2639/1670226175459-10893a4c-6f62-4815-930e-25784a63c84e.png#clientId=uafcf068e-9738-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=794&id=u993996b4&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1588&originWidth=2146&originalType=binary&ratio=1&rotation=0&showTitle=false&size=372161&status=done&style=none&taskId=u50c56780-d0b0-4909-a9c2-ddf530241ad&title=&width=1073)

## 配置和运行
### 配置
各配置项和说明，参见《KWS训练套件配置说明》。
### 运行
进入kws-training-scripts目录，运行以下命令：

```shell
# 通过设置环境变量指定希望使用的GPU的id序号，从 0 开始
export CUDA_VISIBLE_DEVICES=gpu_id
# config.yml 为训练配置文件
# --remote_dataset 指定需要下载第三方开源数据集
# /data/open_dataset 是用户指定的数据集存放目录，需要至少300G磁盘空间
# 程序支持断点续传和智能判断，之前已经下载过的话不会重复下载
python pipeline.py config.yml --remote_dataset /data/open_dataset
```

### 步骤和产物

- 检查数据，生成最终训练配置
- 训练阶段，实时读取原始数据，生成训练数据，训练模型，每轮生成的模型checkpoint保存成.pth文件，放在$work_dir/first，默认训练500轮
- 训练完毕后，从所有模型checkpoint中挑选loss最小的一批（约20%），转换为推理格式.txt文件，保存在$work_dir/first_txt
- 每个模型都用测试集测试各场景唤醒率和误唤醒率，汇总结果存放在$work_dir/first_roc，详细结果存放在$work_dir/first_roc_eval
- 综合唤醒率和误唤醒率结果对模型进行排序后存放在$work_dir/first_roc_sort
- 给出排序第一名的模型

第一轮和第二轮产物相同，第一轮存放路径前缀为first，第二轮为second

## 其他工具
### kws模型打标工具
套件中提供的kws_align.py脚本可以利用唤醒模型对线上音频做筛选和打标，处理后的数据即可用来训练正式模型。

脚本调用方式如下：

```
kws_align.py [-h] -m MODEL_TXT [-o OUT_DIR] [-t THREADS] input keyword_desc
其中：
input			是线上音频所在目录
keyword_desc	是唤醒词描述符，针对”小爱同学“模型是 0_xiao_ai_tong_xue,1,2,3,4
-m MODEL_TXT	从附件中zip包解压得到txt模型参数文件
-o OUT_DIR		指定生成的音频数据输出目录
-t THREADS		并发处理线程数
```
例如：
`python kws_align.py /your/audio/data/ 0_xiao_ai_tong_xue,1,2,3,4 -m top_28_checkpoint_0089.txt`

