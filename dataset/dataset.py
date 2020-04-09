#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:14 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import get_transform

class FGVC7Data(Dataset):
    def __init__(self,root,transform=None, phase='train' ):
        self.root = root
        self.transform = transform
        self.ids, self.labels = self._read_img_ids(root)
        self.phase = phase

    def _read_img_ids(self, root):
        assert self.phase =='train' or self.phase == 'test'
        if self.phase == 'train':
            file_path = os.path.join(root,'train.csv')
            ids = []
            labels = []
            with open(file_path, 'rb') as f:
                for line in f.readlines():
                    line = str(line.strip(), encoding='UTF-8').split(',')
                    ids.append(line[0])
                    labels.append((line.index('1')) - 1)
        else :
            file_path = os.path.join(root, 'test.csv')
            ids = []
            labels = []
            with open(file_path, 'rb') as f:
                for line in f.readlines():
                    line = str(line.strip(), encoding='UTF-8').split(',')
                    ids.append(line[0])
        return ids, labels


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        pass