#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:14 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import torch
import math
import torch.nn.functional as F
from torch import nn, Tensor

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

    def cross_entropy(self, preds, trues, class_weights=1.0, reduction='mean', **kwargs):
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
        y_onehot = torch.zeros([bs, 4]).to(logits).scatter_(1, labels.view(-1,1), 1)
        output = (y_onehot * phi) + ((1.0 - y_onehot) * cosine)
        output *= self.s
        loss = self.cross_entropy(output, y_onehot, reduction = self.reduction)
        return loss / 2

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=80) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def convert_label_to_similarity(self, normed_feature: Tensor, label: Tensor):
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        sp, sn = self.convert_label_to_similarity(sp, sn)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

class Criterion(nn.Module):
    def __init__(self, weight_arcface=1.0, weight_ce=1.0):
        super(Criterion, self).__init__()

        self.arcfaceloss = ArcFaceLoss()
        ceweight = [1, 4, 1, 1]
        self.weight = torch.tensor(ceweight)
        self.weight_arcface = weight_arcface
        self.weight_ce = weight_ce

    def forward(self, out, labels, image=None):
        if len(out) ==3 :
            logits, arc_metric, build_img = out
        else :
            logits, arc_metric = out
            build_img = None
        loss1 = self.arcfaceloss(arc_metric, labels) if self.weight_ce else 0
        loss2 = F.cross_entropy(logits,labels, weight=self.weight.to(logits))
        loss = loss1 * self.weight_arcface + loss2 * self.weight_ce
        if image is not None and build_img is not None:
             loss += F.mse_loss(build_img.flatten(), image.flatten())
        return loss

    def ce_forward(self, logits, labels):
        return F.cross_entropy(logits,labels, weight=self.weight.to(logits))