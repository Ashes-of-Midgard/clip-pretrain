import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, RandomResizedCrop, Normalize, Resize
import os

import model
from train import train_epoch
from eval import eval_epoch
from dataset import CIFAR10_CATEGORIES


if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'
    print('Using device %s' % device)
    epoch_num = 100
    batch_size_train = 32
    batch_size_eval = 32
    lr = 0.05
    topk = (1,5)

    model_train,_ = model.load('RN50', device=device, cls_categories=CIFAR10_CATEGORIES)
    model_train.frozen_text_backbone()

    train_transform = Compose([ToTensor(),
                               Normalize(mean=[125.307, 122.961, 113.8575],
                                         std=[51.5865, 50.847, 51.255]),
                               RandomResizedCrop(size=(224,224)),
                               RandomHorizontalFlip()])
    eval_transform = Compose([ToTensor(),
                              Resize((224,224)),
                              Normalize(mean=[125.307, 122.961, 113.8575],
                                        std=[51.5865, 50.847, 51.255])])
    train_set = datasets.CIFAR10('./data/cifar10',train=True,download=True,transform=train_transform)
    eval_set = datasets.CIFAR10('./data/cifar10',train=False,download=True,transform=eval_transform)

    
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size_eval, shuffle=False)

    optim = Adam(filter(lambda p: p.requires_grad, model_train.parameters()), lr=lr)
    scheduler = MultiStepLR(optimizer=optim, milestones=[30,80], gamma=0.1)

    criterion = CrossEntropyLoss()

    best_acc = 0.
    checkpoint = None
    for epoch in range(epoch_num):
        
        model_train.train()
        train_epoch(epoch,
                    model_train,
                    optim,
                    scheduler,
                    train_loader,
                    criterion,
                    device)
        
        model_train.eval()
        acc, loss = eval_epoch(epoch,
                               model_train,
                               eval_loader,
                               criterion,
                               topk,
                               device)
        print('Epoch %d, eval loss %.4f, acc@1 %.2f\%, acc@5 %.2f\%' % (epoch, loss, 100 * acc[1], 100 * acc[5]))
        if acc[1] > best_acc:
            best_acc = acc[1]
            checkpoint = {'epoch':epoch,
                          'acc':acc[1],
                          'state_dict':model_train.state_dict()}
            os.makedirs('checkpoints',exist_ok=True)
            with open('checkpoints/best_acc.pth.tar', 'w') as f:
                torch.save(checkpoint, f)
    
    print('Training finished, best acc: %.2f\%, best epoch: %d' % 100*checkpoint['acc'], checkpoint['epoch'])
            