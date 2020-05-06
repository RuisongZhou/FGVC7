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
ans4 = pd.read_csv('97.7.csv')
healthy = (ans1.healthy + ans2.healthy + ans3.healthy+ ans4.healthy) /4
multiple_diseases =  (ans1.multiple_diseases + ans2.multiple_diseases + ans3.multiple_diseases+ans4.multiple_diseases) /4
rust =  (ans1.rust + ans2.rust + ans3.rust+ans4.rust) /4
scab =  (ans1.scab + ans2.scab + ans3.scab+ans4.scab) /4

sample_submission['healthy'] = healthy
sample_submission['multiple_diseases'] = multiple_diseases
sample_submission['rust'] = rust
sample_submission['scab'] = scab
sample_submission.to_csv('merge.csv', index=False)

