#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:14 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset


class FGVC7Data(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = root
        self.transform = transform
        self.phase = phase
        self.ids, self.labels = self._read_img_ids(root)

    def _read_img_ids(self, root):
        assert self.phase == 'train' or self.phase == 'test' or self.phase == 'valid'
        if self.phase != 'test':
            file_path = os.path.join(root, 'train.csv')
            ids = []
            labels = []
            with open(file_path, 'rb') as f:
                f.readline()
                for line in f.readlines():
                    line = str(line.strip(), encoding='UTF-8').split(',')
                    ids.append(line[0])
                    labels.append((line.index('1')) - 1)
            ids = [os.path.join(self.root, 'images', e + '.jpg') for e in ids[:1600]] if self.phase == 'train' else \
                [os.path.join(self.root, 'images', e + '.jpg') for e in ids[1600:]]
        else:
            file_path = os.path.join(root, 'test.csv')
            ids = []
            labels = []
            with open(file_path, 'rb') as f:
                f.readline()
                for line in f.readlines():
                    line = str(line.strip(), encoding='UTF-8').split(',')
                    ids.append(line[0])
            ids = [os.path.join(self.root, 'images', e + '.jpg') for e in ids]
        return ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        image = Image.open(id).convert('RGB')  # (C, H, W)
        if self.transform != None:
            image = self.transform(image)
        if self.phase == 'test':
            return id
        else:
            label = self.labels[item]
            return image, label


if __name__ == '__main__':
    dataset = FGVC7Data('./data/plant-pathology-2020-fgvc7', phase='test')
    print(dataset[0])
