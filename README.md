# Installation

```shell
conda create -n clip-pretrain python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate clip-pretrain
pip install openmim
mim install mmyolo
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

# Prepare pretrained models
RN50不需要手动下载，YOLOv8可以从链接：https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_095938-ce3c1b3f.pth，下载
下载后删除文件名最后的日期后缀（即yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.pth），放置在model/pretrained目录下.
如果想用其他的预训练权重，需要根据对应预训练权重的模型参数设置更改./model/model.py中load函数里与YOLOv8相关部分.

# Train
```shell
python main.py --backbone RN50
python main.py --backbone YOLOv8
```

# Convert Model Weights
想要利用训练后的darknet作为YOLOv8模型的backbone，需要将保存在logs目录下的checkpoint文件转化为和yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.pth相同的格式。这里采用的方式就是加载yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.pth作为基础模板，将其中backbone部分的权重替换为logs目录下checkpoint中的权重.
```shell
python convert_checkpoint.py
```