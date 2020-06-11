# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import random
import numpy as np
import torch.nn.functional as F
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

class BatchDrop(nn.Module):
    def __init__(self, h_ratio=0.2, w_ratio=1.0):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x

def block_weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        #init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)

def block_weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal(m.weight.data, std=0.001)
    #init.constant(m.bias.data, 0.0)
    
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num,  relu=True, num_bottleneck=2048):
        super(ClassBlock, self).__init__()
        #add_block = []
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck,bias = False)] 
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]
       
        
        #add_block = nn.Sequential(*add_block)
        #add_block.apply(weights_init_kaiming)
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(block_weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(block_weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num,bias = False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(block_weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block1(x)
        x1 = self.add_block2(x)
        x2 = self.classifier(x1)
        return x1, x2
class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, class_block=True):
        super(Baseline, self).__init__()
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

        #init
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.class_block = class_block
        
        #global branch
        self.global_gap = nn.AdaptiveAvgPool2d(1)
        self.global_gmp = nn.AdaptiveMaxPool2d(1)
        self.global_reduction = nn.Sequential(
                                nn.Conv2d(2048, 512, 1), 
                                nn.BatchNorm2d(512), 
                                nn.ReLU()
                                )
        self.global_reduction.apply(weights_init_kaiming)                        
        if self.class_block:
            self.global_classifier = ClassBlock(2048, self.num_classes, 2048)
        else :
            self.global_classifier = nn.Linear(512, self.num_classes, bias=False)
            self.global_classifier.apply(weights_init_classifier)
        
        #bdb branch
        self.part = Bottleneck(2048, 512)
        self.part_gap = nn.AdaptiveAvgPool2d(1)
        self.part_gmp = nn.AdaptiveMaxPool2d(1)
        self.batch_drop = BatchDrop()
        self.part_reduction = nn.Sequential(
                                nn.Conv2d(2048, 1024, 1), 
                                nn.BatchNorm2d(1024), 
                                nn.ReLU()
                                )
        self.part_reduction.apply(weights_init_kaiming)
        
        #attention
        #self.attentions = SELayer(2048)
        self.pam = PAM_Module(2048)
        self.cam = PAM_Module(2048)
        #self.attentions.apply(attention_init)
        self.K = 32
                                   
        if self.class_block :
            self.part_classifier = ClassBlock(2048, self.num_classes, 2048)
        else :
            self.part_classifier = nn.Linear(1024, self.num_classes, bias=False)
            self.part_classifier.apply(weights_init_classifier)
        
        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

    #返回总亮度前K大的通道特征
    def getTopK(self, x, K):
        batch_size, c, h, w = x.data.size()
        matrix = x.sum(dim=(2, 3))
        _, index = torch.sort(matrix, 1, True)
        index = index.cpu().detach().numpy()
        attention_maps = []
        for i in range(batch_size):
            attention_maps.append(x[i, index[i][:K], ...])
        return torch.stack(attention_maps)

    def batch_augmentation(self, images, attention_map, mode='drop', theta=0.5, padding_ratio=0.1):
        batches, _, imgH, imgW = images.size()

        if mode == 'crop':
            crop_images = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_c = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = theta * atten_map.max()

                crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
                nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
                height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
                height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
                width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
                width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

                crop_images.append(
                    F.upsample_bilinear(
                        images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                        size=(imgH, imgW)))
            crop_images = torch.cat(crop_images, dim=0)
            return crop_images

        elif mode == 'drop':
            drop_masks = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_d = random.uniform(*theta) * atten_map.max()
                else:
                    theta_d = theta * atten_map.max()
                # 只保留小于阈值的单元,即把显著相关的区域drop掉
                drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
            drop_masks = torch.cat(drop_masks, dim=0)
            drop_images = images * drop_masks.float()
            return drop_images

        else:
            raise ValueError(
                'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)            
            
    def forward(self, x):
        
        batch_size = x.size(0)
        x = self.base(x)
        predict = []
        triplet_feats = []
        classifier_feats = []
        
        #global branch
        global_feat = self.global_gap(x)
        global_triplet_feat = self.global_reduction(global_feat).squeeze()
        if self.class_block :
            global_predict, global_classifier_feat = self.global_classifier(global_feat.squeeze())
        else :
            global_classifier_feat = self.global_classifier(global_triplet_feat) 
            global_predict = global_triplet_feat  
            
        triplet_feats.append(global_triplet_feat)
        classifier_feats.append(global_classifier_feat)
        predict.append(global_predict)
        
        #bdb branch
        part_feat = self.part(x)
        part_feat = self.batch_drop(part_feat)
        #attention
        #attention_maps = self.attentions(x)
        attention_maps = self.pam(x) + self.cam(x)
        attention_maps = self.getTopK(attention_maps, self.K)
        
        '''
        #general attention map
        if self.training:
            # 随机选择一个通道作为attention mask
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + 1e-12)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.K, 1, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)
            #print(attention_map.shape)
            part_feat = self.batch_augmentation(part_feat, attention_map, mode='drop', theta=0.5)        
        '''
        part_feat = self.part_gmp(part_feat)
        
        part_triplet_feat = self.part_reduction(part_feat).squeeze()
        if self.class_block :
            part_predict, part_classifier_feat = self.part_classifier(part_feat.squeeze())
        else :
            part_classifier_feat = self.part_classifier(part_triplet_feat)
            part_predict = part_triplet_feat
            
        triplet_feats.append(part_triplet_feat)
        classifier_feats.append(part_classifier_feat)
        predict.append(part_predict)
        

        if self.training:
            return classifier_feats, triplet_feats  
        else:
            return torch.cat(predict, 1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
            
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def attention_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_in')
            if 'bn' in key:
                nn.init.constant_(m.state_dict()[key][...], 1.)
        elif key.split('.')[-1] == 'bias':
            nn.init.constant_(m.state_dict()[key][...], 0.)            
