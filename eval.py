from typing import Tuple, Callable, List, Union, Dict
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import tqdm
import math
import warnings

from model import CLIPZeroshotClassifier
from utils import AverageMeter


def get_acc(preds:Tensor,
            labels:Tensor,
            topk:Union[List[int], Tuple[int, ...]]) -> Dict[int, float]:
    if len(labels.shape) == 2:
        labels = torch.argmax(labels, dim=1)

    maxk = max(topk)
    _, pred_indices = preds.topk(maxk, dim=1)
      
    acc = {}
    for k in topk:
        correct = pred_indices[:, :k].eq(labels.view(-1, 1).expand_as(pred_indices[:, :k])).any(dim=1)
        acc[k] = correct.float().mean().item()

    return acc


def eval_epoch(epoch:int,
               model:CLIPZeroshotClassifier,
               eval_loader:DataLoader,
               criterion:Callable,
               topk:Union[List[int], Tuple[int, ...]],
               device:torch.device) -> Tuple[Dict[int, float], float]:
    avg_loss = AverageMeter()
    avg_acc = {k: AverageMeter() for k in topk}
    tbar = tqdm.tqdm(eval_loader)
    with torch.no_grad():
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
                avg_loss.update(loss.item())

            acc = get_acc(logits, labels, topk)
            for k in topk:
                avg_acc[k].update(acc[k])
            
            tbar.set_description('Epoch %d, eval loss %.4f, eval acc %.2f\%' % (epoch, avg_loss.avg, 100 * avg_acc[1].avg))
    return {avg_acc[k].avg for k in topk}, avg_loss.avg