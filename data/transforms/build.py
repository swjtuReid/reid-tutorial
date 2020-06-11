# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, Cutout


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if cfg.INPUT.DATA_AUGMENTATION == 'RE':
        data_augmentation = RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    elif cfg.INPUT.DATA_AUGMENTATION == 'CUT':
        data_augmentation = Cutout(probability = 0.5, size=64, mean=cfg.INPUT.PIXEL_MEAN) # original code uses mean=[0.0, 0.0, 0.0]? 
    if is_train:
        if cfg.INPUT.DATA_AUGMENTATION == 'NO':
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                #RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                data_augmentation
                #RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])   
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
