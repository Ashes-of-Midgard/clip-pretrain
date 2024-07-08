# Installation

```shell
conda create -n clip-pretrain python=3.8 -y
conda activate clip-pretrain
conda install -y -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm packaging
```

# Prepare dataset
数据集链接: https://pan.baidu.com/s/1D-gw8qkoJt3UEpelYa3oqQ
提取码: 4kxd
下载后提取解压，按照如下目录放置在data文件夹下
|- data
    |- train
    |- val
    |- gen_txt.py
    |- gen_image.py

搜索所有的desktop.ini，并删除；修改gen_txt.py当中的root为'./'，然后进入data目录下
```shell
cd data
python gen_txt.py
```

# Train
```shell
python main.py
```