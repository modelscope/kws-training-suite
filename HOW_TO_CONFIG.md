## 配置文件格式
配置文件为yaml格式，列表类配置项的值无论是一条还是多条，每一条前面都应当加 - 。
以下是一个最精简的配置文件例子：

```yaml
work_dir: /mnt/data/data/ckpt_0923
workers: 60
kws_decode_desc:
- ni_hao_tian_mao,1,2,3,4
- jin_tian_de_tian_qi,5,3,6,3,7
test_pos_data_dir: /mnt/data/himiya.dev/origin_pos_4mic
test_pos_anno_dir: /mnt/data/himiya.dev/annotation_pos_4mic
test_neg_data_dir: /mnt/data/himiya.dev/origin_neg_4mic
train_pos_list:
- /mnt/data/wakeup/himiya_align_list_train.txt,0.2
- /mnt/data/wakeup/himiya_align_list_aug.txt,0.8
```

## 必选配置
### 基础配置
#### `work_dir (str)`
程序工作目录，所有生成的数据都会存放在下面。
包括

- 训练生成的模型
- 模拟负样本标注
- 测试生成的中间音频和log等

#### `workers (int)`
训练和测试时并发工作线程数，可以加快程序处理速度，配置值应接近可用CPU总数，建议不低于60。

#### `kws_decode_desc (list[str])`
关键词描述字段，支持设置多个关键词。
单个关键词描述分为两部分以逗号分隔，第一部分是关键词字符串；第二部分是字符串每个音节的分类值，有几个音节就写几个值，之间也以逗号分隔。例如`ni_hao_mi_ya,1,2,3,4` 
此处的分类值必须与唤醒音频标注中一致（注：当不一致时可参考[关键词分类映射配置](#NVepB)）。

##### Tips
关键词不限制4字短语，也可以是“下一首”“音量调大些”等n字短语，则映射标记也对应n个。
多个关键词的分类值定义规则：

- 所有关键词每个字按读音分类，不区分声调，同一个读音的字定义为同一个分类，各关键词共享此分类
- 例如 2个关键词： 你好天猫，今天的天气
- 得到如下分类对应表，其中"天"字为两个关键词共享
- 那么 "你好天猫"的对应分类是 1,2,3,4 "今天的天气"对应分类是  5,3,6,3,7（配置示例参见文档开头）

| ni | hao | tian | mao | jin | de | qi |
| --- | --- | --- | --- | --- | --- | --- |
| 你 | 好 | 天 | 猫 | 今 | 的 | 气 |
| 1 | 2 | 3 | 4 | 5 | 6 | 7 |


### 测试数据配置
包括 `test_pos_data_dir, test_neg_data_dir, test_pos_anno_dir`

#### 数据格式
此处配置的音频格式应该是从实际设备上采集的包括参考信号的多通道音频。
按默认2麦1参考的设置应该是3通道，采样率16000Hz的wav音频文件。

#### `test_pos_data_dir (str)`
测试数据集中正样本数据所在目录。
目录下应该有1到n个子文件夹，每个文件夹代表一个测试场景，下面存放测试音频wav文件。

#### `test_neg_data_dir (str)`
测试数据集中负样本所在目录，格式要求同上。

#### `test_pos_anno_dir (str)`
测试数据集中正样本标注所在目录。
目录下应该有1到n个子文件夹，每个文件夹代表一个测试场景，测试结果会按场景汇总统计。标注分粗标和精标两种，目前只支持粗标
粗标数据格式：每个子文件夹下存放一个test.txt文件，其中每一行是音频文件路径、对应关键词名称和发生次数，以空格分隔，例如
`/mnt/data/haierxiongdi/haier-xyxy-20220902/spk60db_001.wav hai_er_xiong_di 200`
**注意：**此处关键词名称必须是上面 kws_decode_desc 中配置过的

### 训练数据配置
包括 

```
train_pos_list(关键词&标签)
train_neg_list(非关键词短句)
single_noise1_list, multi_noise1_list, train_noise2_list (噪声音频)
train_interf_list (用来模拟干扰音源的音频)
train_ref_list (用来模拟回声的音频)
```

注意：以上配置项默认为必填，但在使用 --remote_dataset 参数时除第一项外都可以选填，程序会自动下载远程服务器上的开源数据集。用户在使用开源数据的同时仍然可以配置这些项目，程序会把用户配置的数据与开源数据合并使用。

#### 数据格式
以下配置中的音频文件格式除特殊说明外要求为采样率16000Hz的单声道PCM编码.wav文件。
除关键词音频以外的音频应切分成60秒一段的短文件。套件中提供了切分工具，只需指定音频存放路径和并发线程数，切分后的音频默认存放在 /your/wav/dir_seg60目录下，也可以用-o 参数指定输出目录。
运行示例：

```
python split_wav.py /your/wav/dir -t thread_number
```


#### 配置格式
以上配置项都可以配置多行，其中每一行是一个文件路径和对应比重。
文件路径指向个文本文件，其中每一行为一条wav音频文件的路径，各配置项指向的音频内容有区别。
比重值表示取用数据时，从本行取的占总数据量的比例，数值建议每100小时音频的权重设为1，其他时长按比例换算，比如10小时音频则配置权重0.1。
例如您准备了两种音频数据音乐噪声10小时和家居噪声20小时，那么可以如下配置：

```
- /mnt/data/noise/list/music_list.txt,0.1
- /mnt/data/noise/list/homework_list.txt,0.2
```
各种音频数据用途如下图：
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/2639/1668502083753-ffe1827f-47b3-4843-baf9-7fae4104ad3c.png#clientId=u098dba5b-1c65-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=378&id=u4d80cefc&margin=%5Bobject%20Object%5D&name=image.png&originHeight=755&originWidth=1755&originalType=binary&ratio=1&rotation=0&showTitle=false&size=637888&status=done&style=none&taskId=ue8e0ec92-f09b-485f-b3e2-18188edb497&title=&width=877.5)

#### `train_pos_list (list[str])`
配置正样本，关键词音频文件，通常是众包采集的背景安静，发音清晰的关键词语音，而且经过 force align 增加了关键词标注信息。
:::info
可选：关键词分类映射配置
当正样本文件中的标注信息与关键词描述(kws_decode_desc)中的分类定义不一致时可以配置映射关系。方法是在原有配置项后添加 下划线分隔的关键词对应分类值。
例如针对上面例子中“今天的天气” 关键词可如下配置
`/mnt/data/list/jin_tian_de_tian_qi_pos_list.txt,0.6,5_3_6_3_7`
:::

#### `train_neg_list (list[str])`
配置负样本，不含唤醒词的人声文件。
#### `single_noise1_list, multi_noise1_list (list[str])`
配置单通道噪声音频，可包括音乐，扫地机，吸尘器等各种噪声，通常这两项可以配置相同值。有经验的训练者可以采取不同配置。
#### `train_noise2_list (list[str])`
配置多通道噪声音频，应当是真实设备录制的多通道音频
#### `train_interf_list (list[str])`
配置干扰声音频
#### `train_ref_list (list[str])`
配置参考音频，可以是设备中播放出的音乐、故事、电视节目等。

## 可选配置
### 训练量配置
#### `max_epochs (int)`
训练轮数，默认500
#### `val_iters_per_epoch (int)`
验证数据集有几个batch，默认为150
#### `train_iters_per_epoch (int)`
每轮训练有几个batch，默认为300

### 其他配置
#### `main_keyword (str)`
当需要多个唤醒词时，可以在这里配置其中一个作为主唤醒词。
那么从训练出的模型中选择最优模型时主要会以这个主唤醒词的效果为准。
#### `mic_number (int)`
硬件设备麦克风数量，默认为2
用来计算输入音频通道数，即2 mic + 1 ref = 3ch。目前通道顺序不支持自定义，默认为mic, mic, ref 

#### `max_far (float)`
误唤醒阈值，单位 次/小时，模型效果测试排序时，高于阈值的数据不参与计算。
最终返回给用户的唤醒率和kws_level配置组合 是保证误唤醒率低于此阈值的所有组合中唤醒率最高的。

