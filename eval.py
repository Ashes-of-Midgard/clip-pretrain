from typing import Tuple, Callable, List, Union, Dict, Optional
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import tqdm
import math
import warnings

from model import CLIPZeroshotClassifier
from utils import AverageMeter


def get_correct(preds:Tensor,
                labels:Tensor,
                topk:Union[List[int], Tuple[int, ...]],
                num_cls:Optional[int]=None) -> Dict[Union[int, str], Dict[int, float]]:
    if len(labels.shape) == 2:
        labels = torch.argmax(labels, dim=1)

    id_mask_of_cls = {}
    id_mask_of_cls['all'] = torch.tensor([True for _ in range(len(labels))])
    if num_cls is not None:
        for i in range(num_cls):
            id_mask_of_cls[i] = (labels==i)
            
    correct = {}
    for key, id_mask in id_mask_of_cls.items():
        maxk = max(topk)
        _, pred_indices = preds.topk(maxk, dim=1)

        correct[key] = {}
        pred_indices_selected = pred_indices[id_mask]
        labels_selected = labels[id_mask]
        
        for k in topk:
            correct[key][k] = pred_indices_selected[:, :k].eq(labels_selected.view(-1, 1).expand_as(pred_indices_selected[:, :k])).any(dim=1)

    return correct


def eval_epoch(epoch:int,
               model:CLIPZeroshotClassifier,
               eval_loader:DataLoader,
               criterion:Callable,
               topk:Union[List[int], Tuple[int, ...]],
               num_cls:Optional[int],
               device:torch.device) -> Tuple[Dict[int, float], float]:
    avg_loss = AverageMeter()
    correct_whole_set = {}
    correct_whole_set['all'] = {k: torch.empty([0], dtype=torch.bool, device=device) for k in topk}
    if num_cls is not None:
        for i in range(num_cls):
            correct_whole_set[i] = {k: torch.empty([0], dtype=torch.bool, device=device) for k in topk}
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

            correct = get_correct(logits, labels, topk, num_cls)
            for key in correct.keys():
                for k in topk:
                    correct_whole_set[key][k] = torch.cat((correct_whole_set[key][k], correct[key][k]), dim=0)

            tbar.set_description('Epoch %d, eval loss %.4f, eval acc %.2f%%' % (epoch, avg_loss.avg, 100 * correct_whole_set['all'][1].float().mean().item()))
    
    return {key: {k: correct_whole_set[key][k].float().mean().item() for k in topk} for key in correct_whole_set.keys()}, avg_loss.avg