#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 9:41 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F

class HS(nn.Module):

    def __init__(self):
        super(HS, self).__init__()

    def forward(self, inputs):
        clip = torch.clamp(inputs + 3, 0, 6) / 6
        return inputs * clip

def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding=kernel//2, bias=False),
        nn.BatchNorm2d(oup),
        HS()
    )
class FPN(nn.Module):
    def __init__(self, in_channels_list,out_channels):
        super(FPN, self).__init__()

        self.output1 = conv_bn(in_channels_list[0], out_channels, 1,1)
        self.output2 = conv_bn(in_channels_list[1], out_channels, 1,1)
        self.output3 = conv_bn(in_channels_list[2], out_channels, 1,1)
        self.output4 = conv_bn(in_channels_list[3], out_channels, 1,1)

        self.merge1 = conv_bn(out_channels, out_channels, 3,1)
        self.merge2 = conv_bn(out_channels, out_channels, 3,1)

    def forward(self, input):
        # names = list(input.keys())
        #input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        output4 = self.output4(input[3])

        up = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + up
        output3 = self.merge2(output3)

        up = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1

class UpSample(nn.Module):
    def __init__(self, channels, scale=2, keep_channel=False):
        super(UpSample, self).__init__()
        if keep_channel == False:
            assert channels % (scale**2) == 0

        self.upsampleblock = nn.Sequential(
            nn.Conv2d(channels, channels*scale*scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ) if  keep_channel else   \
        nn.Sequential(
            nn.PixelShuffle(scale),
            nn.BatchNorm2d(int(channels/(scale**2))),
            nn.ReLU()
        )
        self.init()

    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.upsampleblock(x)

class Rebuild(nn.Module):
    def __init__(self, in_channels, block):
        super(Rebuild,self).__init__()
        self.block = block

        self.upsampleblock = nn.Sequential(
            UpSample(in_channels, keep_channel=True),       # x64
            conv_bn(in_channels, in_channels,3,1),
            UpSample(in_channels, keep_channel=False),    # x16build_img
            conv_bn(in_channels//4, in_channels//4,3,1),
            nn.Conv2d(in_channels//4,out_channels=3,kernel_size=1,padding=0, bias=True)
        )
        self.init()
    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.block(x)
        x = self.upsampleblock(x)
        return x
