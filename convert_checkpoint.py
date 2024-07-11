# convert the checkpoint that 'main.py' saves to mmyolo's format
import torch
from typing import Dict


converted_save_path = 'model/pretrained/yolov8_s_clip-pretrain.pth'
load_checkpoint_path = 'logs/best_acc_epoch_3_YOLOv8_11_07_2024_13_21_04.pth.tar'
reference_checkpoint_path = 'model/pretrained/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.pth'

checkpoint_loaded = torch.load(load_checkpoint_path)
state_dict_loaded = checkpoint_loaded['state_dict']
checkpoint_converted = torch.load(reference_checkpoint_path)

state_dict_loaded: Dict
for key, value in state_dict_loaded.items():
    key: str
    if key.startswith('visual.darknet'):
        converted_key = 'backbone'+key[14:]
        checkpoint_converted['state_dict'][converted_key] = value
        print(f'Convert key {key} to {converted_key}')

torch.save(checkpoint_converted, converted_save_path)