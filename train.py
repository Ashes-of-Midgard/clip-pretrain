from typing import Callable
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import tqdm
import math
import warnings

from model import CLIPZeroshotClassifier
from utils import AverageMeter


def train_epoch(epoch:int,
                model:CLIPZeroshotClassifier,
                optim:Optimizer,
                scheduler:_LRScheduler,
                train_loader:DataLoader,
                criterion:Callable,
                device:torch.device) -> float:
    avg_loss = AverageMeter()
    tbar = tqdm.tqdm(train_loader)
    for i, (images, labels) in enumerate(tbar):
        images = images.to(device, model.dtype)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        loss: Tensor

        if math.isnan(loss.item()):
            warnings.warn('nan value appeared in loss')
            continue
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss.update(loss.item())

        tbar.set_description('Epoch %d, training loss %.4f' % (epoch, avg_loss.avg))
    if scheduler is not None:
        scheduler.step()
    return avg_loss.avg