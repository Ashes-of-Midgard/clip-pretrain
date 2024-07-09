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
from typing import List
from datetime import datetime

import model
from train import train_epoch
from eval import eval_epoch
from dataset import SHIP_CATEGORIES, InfraredShipDataset


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='RN50')
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_eval', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epoch_num', type=int, default=20)
    parser.add_argument('--milestones', type=List, default=[5, 15])
    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'
    print('Using device %s' % device)
    
    epoch_num = args.epoch_num
    batch_size_train = args.batch_size_train
    batch_size_eval = args.batch_size_eval
    lr = args.learning_rate

    topk = (1,)
    image_size = (224, 224) if args.backbone != 'YOLOv8' else (640, 640)

    time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.makedirs('./logs', exist_ok=True)
    log_file = './logs/' + "log_"+time_stamp + '.log'
    with open(log_file, 'w') as f:
        f.write('backbone: '+args.backbone+'\n')
        f.write('learning_rate: '+str(lr)+'\n')
        f.write('batch_size_train: '+str(batch_size_train)+'\n')
        f.write('batch_size_eval: '+str(batch_size_eval)+'\n')
        f.write('epoch_num: '+str(epoch_num)+'\n')
        f.write('image_size: ' + str(image_size[0]) + '\n')

    model_train,_ = model.load(args.backbone, device=device, cls_categories=SHIP_CATEGORIES)
    model_train.frozen_text_backbone()

    train_transform = Compose([ToTensor(),
                               Normalize(mean=[0.1168, 0.1168, 0.1168],
                                         std=[0.0282, 0.0282, 0.0282]),
                               RandomResizedCrop(size=image_size),
                               RandomHorizontalFlip()])
    eval_transform = Compose([ToTensor(),
                              Resize(image_size),
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
    scheduler = MultiStepLR(optimizer=optim, milestones=args.milestones, gamma=0.1)

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
        with open(log_file, 'a') as f:
            f.write('Epoch %d, eval loss %.4f, acc@ %.2f%%' % (epoch, loss, 100 * acc[1]))
        if acc[1] > best_acc:
            best_acc = acc[1]
            checkpoint = {'epoch':epoch,
                          'acc':acc[1],
                          'state_dict':model_train.state_dict()}
            os.makedirs('checkpoints',exist_ok=True)
            with open(f'logs/best_acc_epoch_{epoch}_{args.backbone}_{time_stamp}.pth.tar', 'wb') as f:
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
