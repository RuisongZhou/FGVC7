#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:14 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label , dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5, reduction='mean'):
        super(ArcFaceLoss, self).__init__()
        self.reduction = reduction
        self.s = s
        self.cos_m = math.cos(m)             #  0.87758
        self.sin_m = math.sin(m)             #  0.47943
        self.th = math.cos(math.pi - m)      # -0.87758
        self.mm = math.sin(math.pi - m) * m  #  0.23971

    def cross_entropy(preds, trues, class_weights=1.0, reduction='mean', **kwargs):
        class_weights = torch.tensor(class_weights).to(preds)
        ce_loss = -torch.sum(class_weights * trues * F.log_softmax(preds, dim=1), dim=1)
        if reduction == 'mean':
            return ce_loss.mean()
        elif reduction == 'sum':
            return ce_loss.sum()
        elif reduction == 'none':
            return ce_loss

    def forward(self, logits, labels):
        logits = logits.float()  # float16 to float32 (if used float16)
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # equals to **2
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        bs = logits.size(0)
        y_onehot = torch.zeros([bs, 4]).to(labels).scatter_(1, labels.view(-1,1), 1)

        output = (y_onehot * phi) + ((1.0 - y_onehot) * cosine)
        output *= self.s
        loss = self.cross_entropy(output, y_onehot, reduction = self.reduction)
        return loss / 2


class Criterion(nn.Module):
    def __init__(self, weight_arcface=1.0, weight_ce=1.0):
        super(Criterion, self).__init__()

        self.arcfaceloss = ArcFaceLoss()
        ceweight = [1, 4, 1, 1]
        self.weight = torch.tensor(ceweight)
        self.weight_arcface = weight_arcface
        self.weight_ce = weight_ce

    def forward(self, logits, labels):
        loss1 = self.arcfaceloss(logits, labels)
        loss2 = F.cross_entropy(logits,labels, weight=self.weight.to(logits))

        return loss1 * self.weight_arcface + loss2 * self.weight_ce

    def ce_forward(self, logits, labels):
        return F.cross_entropy(logits,labels, weight=self.weight.to(logits))