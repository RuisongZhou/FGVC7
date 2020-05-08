#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 9:05 上午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com

import numpy as np
import pandas as pd
import os, sys

sample_submission = pd.read_csv(os.path.join('./data/plant-pathology-2020-fgvc7', 'sample_submission.csv'))

ans1 = pd.read_csv('b4_97.5_all.csv')
ans2 = pd.read_csv('b6_97.5_ce.csv')    #97.8
ans3 = pd.read_csv('resnest101_97.3_all.csv')   #98.3
ans4 = pd.read_csv('b6_97.7.csv')       #98.4
ans5 = pd.read_csv('b7_97.2_all.csv')   #98.5
ans6 = pd.read_csv('97.7.csv')      #98.6

ans7 = pd.read_csv('97.0_b5.csv')   #98.6 -- 掉点
ans8 = pd.read_csv('b7_dense201_97.5.csv')  #98.6  -- 掉点

ans9 = pd.read_csv('b7_1_95.1.csv')
ans10 = pd.read_csv('seresnext101_96.7.csv')
ans11 = pd.read_csv('96.7_b4_kaggle.csv')
ans12 = pd.read_csv('b7_dense201_97.5.csv')

ans13 = pd.read_csv('merge_97.7.csv')
anss =  [ans1, ans2, ans3, ans4, ans5, ans6]
#anss = [ans9, ans10, ans11, ans12]
healthy = np.zeros_like(ans1.healthy)
multiple_diseases= np.zeros_like(ans1.healthy)
rust= np.zeros_like(ans1.healthy)
scab= np.zeros_like(ans1.healthy)
alpha = 0.15
for i, ans in enumerate(anss):
    healthy += ans.healthy * (1 + alpha*i)
    multiple_diseases +=  ans.multiple_diseases* (1 + alpha*i)
    rust +=  ans.rust* (1 + alpha*i)
    scab +=  ans.scab* (1 + alpha*i)


sample_submission['healthy'] = healthy / len(anss)
sample_submission['multiple_diseases'] = multiple_diseases/ len(anss)
sample_submission['rust'] = rust/ len(anss)
sample_submission['scab'] = scab/ len(anss)
sample_submission.to_csv('merge.csv', index=False)

