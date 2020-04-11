#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:14 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import os
import pdb
from PIL import Image
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
            ids[1] = ids[1] * 5
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

    def __getitem__(self, item):
        id = self.ids[item]
        image = Image.open(id).convert('RGB')  # (C, H, W)
        if self.transform != None:
            image = self.transform(image)
        if self.phase == 'test':
            return image
        else:
            label = self.labels[item]
            return image, label


if __name__ == '__main__':
    dataset = FGVC7Data('./data/plant-pathology-2020-fgvc7', phase='test')
    print(dataset[0])
