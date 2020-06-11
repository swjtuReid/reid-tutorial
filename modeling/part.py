# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import random
import numpy as np
import torch.nn.functional as F
import copy
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.attentions import PAM_Module, CAM_Module


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _init_reduction(reduction):  # 初始化降维层
    # conv
    nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
    # bn
    nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
    nn.init.constant_(reduction[1].bias, 0.)


class Part(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, class_block=True):
        super(Part, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        # init
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.class_block = class_block

        # 池化层
        self.gap = nn.AdaptiveAvgPool2d((6, 1)) 

        # 1x1卷积层，降维
        reduction = nn.Sequential(nn.Conv2d(2048, self.in_planes, 1, bias=False), nn.BatchNorm2d(self.in_planes), nn.ReLU())
        _init_reduction(reduction)

        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        
        # 全连接层
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_5 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_6 = nn.Linear(self.in_planes, self.num_classes, bias=False)        
        self.classifier_1.apply(weights_init_classifier)   
        self.classifier_2.apply(weights_init_classifier)   
        self.classifier_3.apply(weights_init_classifier)   
        self.classifier_4.apply(weights_init_classifier)   
        self.classifier_5.apply(weights_init_classifier)   
        self.classifier_6.apply(weights_init_classifier)    
        
    def forward(self, x):

        part_feat = self.gap(self.base(x))  # (b, 2048, 6, 1)
        # print(global_feat.shape)
        
        part_1 = part_feat[:, :, 0:1, :]  # (b, 2048, 1, 1)
        part_2 = part_feat[:, :, 1:2, :]
        part_3 = part_feat[:, :, 2:3, :]
        part_4 = part_feat[:, :, 3:4, :]
        part_5 = part_feat[:, :, 4:5, :]
        part_6 = part_feat[:, :, 5:6, :]
        
        feat_1 = self.reduction_1(part_1).squeeze(dim=3).squeeze(dim=2)  # (b, 256, 1, 1)
        feat_2 = self.reduction_2(part_2).squeeze(dim=3).squeeze(dim=2)
        feat_3 = self.reduction_3(part_3).squeeze(dim=3).squeeze(dim=2)
        feat_4 = self.reduction_4(part_4).squeeze(dim=3).squeeze(dim=2)
        feat_5 = self.reduction_5(part_5).squeeze(dim=3).squeeze(dim=2)
        feat_6 = self.reduction_6(part_6).squeeze(dim=3).squeeze(dim=2)
        
        cls_score_1 = self.classifier_1(feat_1)
        cls_score_2 = self.classifier_1(feat_1)
        cls_score_3 = self.classifier_1(feat_1)
        cls_score_4 = self.classifier_1(feat_1)
        cls_score_5 = self.classifier_1(feat_1)
        cls_score_6 = self.classifier_1(feat_1)
        
        predict = torch.cat([feat_1, feat_2, feat_3, feat_4, feat_5, feat_6], dim=1)                    
        classifier_feats = [cls_score_1, cls_score_2, cls_score_3, cls_score_4, cls_score_5, cls_score_6]
        if self.training:
            return classifier_feats , None
        else:
            return predict
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])   
