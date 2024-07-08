import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Union, Callable, Tuple
from PIL import Image

CIFAR10_CATEGORIES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                      'frog', 'horse', 'ship', 'truck')

SHIP_CATEGORIES = ('aircraft carrier', 'cruiser', 'destroyer', 'cruise ship')


class InfraredShipDataset(Dataset):
    def __init__(self, root: str, split: str, transform_img: Callable=None, transform_target: Callable=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform_img = transform_img
        self.transform_target = transform_target
        self.data = []
        with open(f'{self.root}/data_old_{self.split}.txt', 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            img_path = self.root + '/' + line.split(' ')[0]
            label = int(line.split(' ')[1].strip())
            self.data.append({'img_path': img_path, 'label': label})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Union[Image.Image, torch.Tensor]:
        img_path = self.data[index]['img_path']
        label = self.data[index]['label']
        if self.transform_target is not None:
            label = self.transform_target(label)

        img = Image.open(img_path).convert('RGB') # The image is expected to be Infrared image
        if self.transform_img is not None:
            img = self.transform_img(img)

        return img, label
    
    def get_mean_std(self) -> Tuple[torch.tensor, torch.Tensor]:
        # record the transforms setting
        keep_transform_img = self.transform_img
        keep_transform_target = self.transform_target

        # Set the transforms to prevent interfering the images' mean and std
        self.transform_img = transforms.ToTensor()
        self.transform_target = None

        # calculate the mean
        value_sum = torch.tensor([0., 0., 0.])
        num_pix = 0
        for i in range(self.__len__()):
            img, label = self.__getitem__(i)
            num_pix += img.shape[1] * img.shape[2]
            img = img.sum(dim=-1)
            img = img.sum(dim=-1)
            value_sum += img
        mean = value_sum / num_pix

        # calculate the std
        value_sum = torch.tensor([0., 0., 0.])
        for i in range(self.__len__()):
            img, label = self.__getitem__(i)
            diff = img - mean.unsqueeze(1).unsqueeze(2)
            diff_sq = diff.square()
            diff_sq = diff_sq.sum(-1)
            diff_sq = diff_sq.sum(-1)
            value_sum += diff_sq
        std = value_sum / num_pix

        # restore transforms
        self.transform_img = keep_transform_img
        self.transform_target = keep_transform_target

        return mean, std