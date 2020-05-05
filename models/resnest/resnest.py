##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from .resnet import ResNet, Bottleneck
import torch.nn as nn
from torch.utils import model_zoo

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model

from torch.nn.utils.weight_norm import WeightNorm
class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores

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

class Resnest50(nn.Module):
    def __init__(self, num_classes=4):
        super(Resnest50, self).__init__()
        self.model = resnest50(pretrained=True)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(512* 4, num_classes)
        self.arc = ArcMarginProduct(2048, num_classes)
    def logits(self, x):
        x = self.avg_pool(x)
        arc = self.arc(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x, arc

    def forward(self, x):
        x = self.model.extract_feature(x)
        x, arc = self.logits(x)

        return x, arc

class Resnest101(nn.Module):
    def __init__(self, num_classes=4):
        super(Resnest101, self).__init__()
        self.model = resnest101(pretrained=True)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = distLinear(512 * 4, num_classes)
        self.arc = ArcMarginProduct(2048, num_classes)
    def logits(self, x):
        x = self.avg_pool(x)
        arc = self.arc(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x, arc

    def forward(self, x):
        x = self.model.extract_feature(x)
        x, arc = self.logits(x)

        return x, arc

class Resnest200(nn.Module):
    def __init__(self, num_classes=4):
        super(Resnest200, self).__init__()
        self.model = resnest200(pretrained=True)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(512 * 4, num_classes)
        self.arc = ArcMarginProduct(2048, num_classes)
    def logits(self, x):
        x = self.avg_pool(x)
        arc = self.arc(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x,arc = self.last_linear(x)
        return x, arc

    def forward(self, x):
        x = self.model.extract_feature(x)
        x,arc = self.logits(x)

        return x, arc

class Resnest269(nn.Module):
    def __init__(self, num_classes=4):
        super(Resnest269, self).__init__()
        self.model = resnest269(pretrained=True)
        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(512 * 4, num_classes)
        self.arc = ArcMarginProduct(2048, num_classes)
    def logits(self, x):
        x = self.avg_pool(x)
        arc = self.arc(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x, arc

    def forward(self, x):
        x = self.model.extract_feature(x)
        x,arc = self.logits(x)

        return x, arc