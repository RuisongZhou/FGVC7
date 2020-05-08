#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 8:10 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.densenet import *

import math
import torch.nn.functional as F
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        features = features.contiguous().view(features.size(0), -1)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class DenseNet121(nn.Module):
    def __init__(self, num_classes=4, metric=False):
        super(DenseNet121, self).__init__()
        self.backbone = densenet121(pretrained=True)

        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(512 * 4, num_classes)
        self.arc = ArcMarginProduct(2048, num_classes)

        self.EX = 2
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.arc_margin_product = ArcMarginProduct(512, num_classes)
        self.bn1 = nn.BatchNorm1d(1024 * self.EX)
        self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
        self.bn2 = nn.BatchNorm1d(512 * self.EX)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512 * self.EX, 512)
        self.bn3 = nn.BatchNorm1d(512)


    def logits(self, x):
        x = torch.cat((nn.AdaptiveAvgPool2d(1)(x), nn.AdaptiveMaxPool2d(1)(x)), dim=1)
        x = self.avg_pool(x)
        arc = self.arc(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x, arc

    def metric(self, x):
        x = torch.cat((nn.AdaptiveAvgPool2d(1)(x), nn.AdaptiveMaxPool2d(1)(x)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = F.dropout(x, p=0.25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.5)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        feature = self.bn3(x)
        cosine = self.arc_margin_product(feature)
        return cosine

    def forward(self, x):

        x = self.backbone.features(x)
        out, arc = self.logits(x)

        return out, arc
