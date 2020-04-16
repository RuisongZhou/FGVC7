#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:14 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import os
import pdb
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import itertools
import random

class FGVC7Data(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = root
        self.transform = transform
        self.phase = phase
        self.ids, self.labels = self._read_img_ids(root)

    def _read_img_ids(self, root):
        df = pd.read_csv(os.path.join(self.root, '{}.csv'.format(self.phase)))
        labels = []
        if self.phase == 'train':
            ids = [[],[],[],[]]
            for i in range(len(df)):
                label = np.argmax(df.iloc[i].values[1:])
                labels.append(label)
                ids[label].append(os.path.join(self.root, 'images', df.iloc[i].values[0]+'.jpg'))
            ids[1] = ids[1] * 3
            labels = [[i]*len(ids[i]) for i in range(4)]
            labels = list(itertools.chain.from_iterable(labels))
            ids = list(itertools.chain.from_iterable(ids))
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(ids)
            random.seed(randnum)
            random.shuffle(labels)
        else:
            ids = df['image_id']
            ids = [os.path.join(self.root, 'images', e+'.jpg')for e in ids]
        return ids, labels

    def __len__(self):
        return len(self.ids)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, item):
        id = self.ids[item]
        image = cv2.imread(id)  # (C, H, W)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[item] if self.phase =='train' else 0
        if self.transform != None:
            image, _ = self.transform(image, label)
        return image, label


if __name__ == '__main__':
    dataset = FGVC7Data('./data/', phase='train')
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset,batch_size=16)
    loader = iter(loader)
    x, y = next(loader)
    x, y = next(loader)