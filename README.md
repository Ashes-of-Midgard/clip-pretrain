# Installation

```shell
conda create -n clip-pretrain python=3.8 -y
conda activate clip-pretrain
conda install -y -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm packaging
```

# Prepare dataset
下载cifar10数据集（暂时用于测试效果）
把cifar-10-batches-py放在data/cifar10

# Train
```shell
python main.py
```