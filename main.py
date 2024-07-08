import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, RandomResizedCrop, Normalize, Resize
import os
import argparse

import model
from train import train_epoch
from eval import eval_epoch
from dataset import SHIP_CATEGORIES, InfraredShipDataset


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone, -b', type=str, default='RN50')
    parser.add_argument('--batch_size_train, -bst', type=int, default=16)
    parser.add_argument('--batch_size_val, -bsv', type=int, default=16)
    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'
    print('Using device %s' % device)
    epoch_num = 20
    batch_size_train = args.batch_size_train
    batch_size_eval = args.batch_size_val
    lr = 1e-5
    topk = (1,)

    model_train,_ = model.load(args.backbone, device=device, cls_categories=SHIP_CATEGORIES)
    model_train.frozen_text_backbone()

    train_transform = Compose([ToTensor(),
                               Normalize(mean=[0.1168, 0.1168, 0.1168],
                                         std=[0.0282, 0.0282, 0.0282]),
                               RandomResizedCrop(size=(224,224)),
                               RandomHorizontalFlip()])
    eval_transform = Compose([ToTensor(),
                              Resize((224,224)),
                              Normalize(mean=[0.1168, 0.1168, 0.1168],
                                        std=[0.0282, 0.0282, 0.0282])])
    train_set = InfraredShipDataset('./data',split='train',transform_img=train_transform)
    eval_set = InfraredShipDataset('./data',split='val',transform_img=eval_transform)

    # train_set_mean, train_set_std = train_set.get_mean_std()
    # print('train_set_mean: ', train_set_mean)
    # print('train_set_std: ', train_set_std)
    
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size_eval, shuffle=False)

    optim = Adam(filter(lambda p: p.requires_grad, model_train.parameters()), lr=lr)
    scheduler = MultiStepLR(optimizer=optim, milestones=[5,15], gamma=0.1)

    criterion = CrossEntropyLoss()

    best_acc = 0.
    checkpoint = None
    for epoch in range(epoch_num):
        model_train.eval()
        acc, loss = eval_epoch(epoch,
                               model_train,
                               eval_loader,
                               criterion,
                               topk,
                               device)
        print('Epoch %d, eval loss %.4f, acc@ %.2f%%' % (epoch, loss, 100 * acc[1]))
        if acc[1] > best_acc:
            best_acc = acc[1]
            checkpoint = {'epoch':epoch,
                          'acc':acc[1],
                          'state_dict':model_train.state_dict()}
            os.makedirs('checkpoints',exist_ok=True)
            with open('checkpoints/best_acc.pth.tar', 'wb') as f:
                torch.save(checkpoint, f)
        
        model_train.train()
        train_epoch(epoch,
                    model_train,
                    optim,
                    scheduler,
                    train_loader,
                    criterion,
                    device)
    
    print('Training finished, best acc: %.2f%%, best epoch: %d' % (100*checkpoint['acc'], checkpoint['epoch']))
