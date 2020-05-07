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
ans2 = pd.read_csv('b6_97.5_ce.csv')
ans3 = pd.read_csv('resnest101_97.3_all.csv')
ans4 = pd.read_csv('b6_97.7.csv')       #98.4
ans5 = pd.read_csv('b7_97.2_all.csv')   #98.5
ans6 = pd.read_csv('97.7.csv')      #98.6
ans7 = pd.read_csv('97.0_b5.csv')   #98.6
anss =  [ans1, ans2, ans3, ans4, ans5, ans6, ans7]
healthy = np.zeros_like(ans1.healthy)
multiple_diseases= np.zeros_like(ans1.healthy)
rust= np.zeros_like(ans1.healthy)
scab= np.zeros_like(ans1.healthy)

for ans in anss:
    healthy += ans.healthy
    multiple_diseases +=  ans.multiple_diseases
    rust +=  ans.rust
    scab +=  ans.scab


sample_submission['healthy'] = healthy / len(anss)
sample_submission['multiple_diseases'] = multiple_diseases/ len(anss)
sample_submission['rust'] = rust/ len(anss)
sample_submission['scab'] = scab/ len(anss)
sample_submission.to_csv('merge.csv', index=False)

