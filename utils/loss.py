#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:14 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import torch
import math
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
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

        self.focal_loss= FocalLoss(class_num=4)
        self.weight_arcface = weight_arcface
        self.weight_ce = weight_ce

    def forward(self, out, labels, image=None):
        if len(out) ==3 :
            logits, arc_metric, build_img = out
        else :
            logits, arc_metric = out
            build_img = None
        loss1 = self.arcfaceloss(arc_metric, labels) if self.weight_arcface else 0
        #loss2 = F.cross_entropy(logits,labels, weight=self.weight.to(logits))
        loss2 = self.focal_loss(logits, labels)
        loss = loss1 * self.weight_arcface + loss2 * self.weight_ce
        if image is not None and build_img is not None:
             loss += F.mse_loss(build_img.flatten(), image.flatten())
        return loss

    def ce_forward(self, logits, labels):
        return F.cross_entropy(logits,labels, weight=self.weight.to(logits))

    def arc_forward(self, logits, labels):
        return self.arcfaceloss(logits, labels)