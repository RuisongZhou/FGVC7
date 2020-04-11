#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 4:08 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
from efficientnet_pytorch import EfficientNet
from torch import nn
size_dict = {
        'b0':'efficientnet-b0',
        'b1': 'efficientnet-b1',
        'b2': 'efficientnet-b2',
        'b3': 'efficientnet-b3',
        'b4': 'efficientnet-b4',
        'b5': 'efficientnet-b5',
        'b6': 'efficientnet-b6',
        'b7': 'efficientnet-b7',
    }

class efficientnet(nn.Module):
    def __init__(self, num_classes, size):
        super(efficientnet, self).__init__()
        self.model = EfficientNet.from_pretrained(size_dict[size])
        self.extra_feature = nn.Sequential(
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280,4)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.extra_feature(x)
        x = self.pooling(x)
        x = x.contiguous().view(-1, x.size(1))
        x = self.fc(x)
        return x

    def get_features(self):
        return self.model.extract_features