# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .part import Part
from .multibranch import MultiBranch
def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    if cfg.MODEL.TYPE == 'base':
        print('model: base')
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    elif cfg.MODEL.TYPE == 'part':
        print('model: part')
        model = Part(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    elif cfg.MODEL.TYPE == 'multibranch':
        print('model: multibranch')
        model = MultiBranch(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)                  
    return model
