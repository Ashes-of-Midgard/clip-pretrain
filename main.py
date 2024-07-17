import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import (Compose, RandomHorizontalFlip, ToTensor,
                                    RandomResizedCrop, Normalize, Resize,
                                    RandomRotation, RandomAutocontrast)
import os
import argparse
from typing import List, Dict
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
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device %s' % device)
    
    backbone = args.backbone
    batch_size_train = args.batch_size_train
    batch_size_eval = args.batch_size_eval
    lr = args.learning_rate
    epoch_num = args.epoch_num if args.mode=='train' else 1
    milestones = args.milestones
    checkpoint_path = args.checkpoint

    topk = (1,)
    image_size = (224, 224) if backbone != 'YOLOv8' else (640, 640)

    time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.makedirs('./logs', exist_ok=True)
    log_file = './logs/' + "log_" + backbone + '_' + args.mode + '_' + time_stamp + '.log'
    with open(log_file, 'w') as f:
        f.write('backbone: '+backbone+'\n')
        f.write('learning_rate: '+str(lr)+'\n')
        f.write('batch_size_train: '+str(batch_size_train)+'\n')
        f.write('batch_size_eval: '+str(batch_size_eval)+'\n')
        f.write('epoch_num: '+str(epoch_num)+'\n')
        f.write('image_size: ' + str(image_size[0]) + '\n')

    model_train,_ = model.load(backbone, device=device, cls_categories=SHIP_CATEGORIES)
    model_train.frozen_text_backbone()

    train_transform = Compose([ToTensor(),
                               Normalize(mean=[0.1168, 0.1168, 0.1168],
                                         std=[0.0282, 0.0282, 0.0282]),
                               RandomResizedCrop(size=image_size),
                               RandomHorizontalFlip(),
                               RandomAutocontrast()])
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
    eval_loader = DataLoader(eval_set, batch_size=batch_size_eval, shuffle=True)

    optim = Adam(filter(lambda p: p.requires_grad, model_train.parameters()), lr=lr)
    scheduler = MultiStepLR(optimizer=optim, milestones=args.milestones, gamma=0.1)

    criterion = CrossEntropyLoss()

    best_acc = 0.
    # load checkpoint
    if checkpoint_path is not None:
        loaded_checkpoint = torch.load(checkpoint_path)
        loaded_state_dict: Dict = loaded_checkpoint['state_dict']
        converted_state_dict = {}
        for key, value in loaded_state_dict.items():
            value: Tensor
            converted_state_dict[key] = value.to(device)
        model_train.load_state_dict(converted_state_dict)

    for epoch in range(epoch_num):
        model_train.eval()
        acc, loss = eval_epoch(epoch,
                               model_train,
                               eval_loader,
                               criterion,
                               topk,
                               len(SHIP_CATEGORIES),
                               device)
        acc_of_categories = {}
        for i in range(len(SHIP_CATEGORIES)):
            acc_of_categories[SHIP_CATEGORIES[i]] = f'{100*acc[i][1]:.2f}%'
        print('Epoch %d, eval loss %.4f, acc %.2f%%' % (epoch, loss, 100 * acc['all'][1]))
        print(f'Acc of categories: {acc_of_categories}')
        with open(log_file, 'a') as f:
<<<<<<< HEAD
            f.write('Epoch %d, eval loss %.4f, acc %.2f%%\n' % (epoch, loss, 100 * acc['all'][1]))
            f.write(f'Acc of categories: {acc_of_categories}\n')
=======
            f.write('Epoch %d, eval loss %.4f, acc@ %.2f%%\n' % (epoch, loss, 100 * acc[1]))
>>>>>>> a55bcd1c1304207016a3ee130edfe93a31d81f3e

        if acc['all'][1] > best_acc:
            best_acc = acc['all'][1]
            checkpoint = {'epoch':epoch,
                          'acc':acc['all'][1],
                          'state_dict':model_train.state_dict()}
            os.makedirs('checkpoints',exist_ok=True)
            with open(f'logs/best_acc_{args.backbone}_{time_stamp}.pth.tar', 'wb') as f:
                torch.save(checkpoint, f)

        if args.mode=='train':
            model_train.train()
            train_epoch(epoch,
                        model_train,
                        optim,
                        scheduler,
                        train_loader,
                        criterion,
                        device)
    
    if args.mode=='train':
        print('Training finished, best acc: %.2f%%, best epoch: %d' % (100*checkpoint['acc'], checkpoint['epoch']))
