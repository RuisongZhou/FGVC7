#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 8:25 上午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
"""EVALUATION
Created: Nov 22,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os, sys
import logging
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import Config
from models import *
from dataset.dataset import FGVC7Data
from utils.utils import TopKAccuracyMetric, batch_augment, get_transform
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='./data/', help='Train Dataset directory path')
parser.add_argument('--net', default='inception_mixed_6e', help='Choose net to use')
args = parser.parse_args()
config = Config()
config.net = args.net
config.refresh()

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

def choose_net(name: str):
    if len(name) == 2 and name[0] == 'b':
        model = efficientnet(size=name)
    elif name.lower() == 'seresnext50':
        model = se_resnext50()
    elif name.lower() == 'seresnext101':
        model = se_resnext101()
    elif name.lower() == 'resnest101':
        model = Resnest101()
    elif name.lower() == 'resnest269':
        model = Resnest269()
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    return model

def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    try:
        ckpt = config.eval_ckpt
    except:
        logging.info('Set ckpt for evaluation in config.py')
        return

    ##################################
    # Dataset for testing
    ##################################
    test_dataset = FGVC7Data(root=args.datasets, phase='test',
                             transform=get_transform(config.image_size, 'tta')[0])
    import pandas as pd
    sample_submission = pd.read_csv(os.path.join(args.datasets, 'sample_submission.csv'))
    ##################################
    # Initialize model
    ##################################
    net = choose_net(args.net)

    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    ##################################
    # use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    results = []
    for index in range(3):
        test_dataset.set_transform(get_transform([config.image_size[0], config.image_size[1]], 'tta')[index])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        net.eval()
        result = []
        with torch.no_grad():
            pbar = tqdm(total=len(test_loader), unit='batches')
            pbar.set_description('Validation TTA {}'.format(index))
            for i, input in enumerate(test_loader):
                X, _ = input
                X = X.to(device)
                y_pred, y_metric = net(X)

                # 处理结果
                y_pred = F.softmax(y_pred, dim=1).cpu().numpy()
                result.append(y_pred)

                batch_info = 'Val step {}'.format((i + 1))
                pbar.update()
                pbar.set_postfix_str(batch_info)
            pbar.close()
        results.append(result)

    healthy = []
    multiple_disease = []
    rust = []
    scab = []
    for i in tqdm(range(len(results[0]))):
        h_ans, m_ans, r_ans, s_ans = 0,0,0,0
        for k in range(len(results)):
            h_ans += results[k][i][0][0]
            m_ans += results[k][i][0][1]
            r_ans += results[k][i][0][2]
            s_ans += results[k][i][0][3]

        healthy.append(h_ans/len(results))
        multiple_disease.append(m_ans/len(results))
        rust.append(r_ans/len(results))
        scab.append(s_ans/len(results))

    sample_submission['healthy'] = healthy
    sample_submission['multiple_diseases'] = multiple_disease
    sample_submission['rust'] = rust
    sample_submission['scab'] = scab
    sample_submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
