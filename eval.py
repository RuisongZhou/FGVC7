"""EVALUATION
Created: Nov 22,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models import WSDAN
from dataset.dataset import FGVC7Data
from utils.utils import TopKAccuracyMetric, batch_augment, get_transform
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='./data/', help='Train Dataset directory path')
parser.add_argument('--net', default='inception_mixed_6e', help='Choose net to use')
args = parser.parse_args()

config.net = args.net
config.refresh()

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# visualize
visualize = config.visualize
savepath = config.eval_savepath
if visualize:
    os.makedirs(savepath, exist_ok=True)

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


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
    test_dataset = FGVC7Data(root=args.datasets, phase='test', transform=get_transform(config.image_size, 'test'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)
    import pandas as pd
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    ##################################
    # Initialize model
    ##################################
    net = WSDAN(num_classes=4, M=config.num_attentions, net=args.net)

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

    ##################################
    # Prediction
    ##################################
    raw_accuracy = TopKAccuracyMetric(topk=(1, 2))
    ref_accuracy = TopKAccuracyMetric(topk=(1, 2))
    raw_accuracy.reset()
    ref_accuracy.reset()

    net.eval()
    result = []
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i,X in enumerate(test_loader):

            X = X.to(device)

            # WS-DAN
            y_pred_raw, _, attention_maps = net(X)

            # Augmentation with crop_mask
            crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)

            y_pred_crop, _, _ = net(crop_image)
            y_pred = (y_pred_raw + y_pred_crop) / 2.

            if visualize:
                # reshape attention maps
                attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
                attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

                # get heat attention maps
                heat_attention_maps = generate_heatmap(attention_maps)

                # raw_image, heat_attention, raw_attention
                raw_image = X.cpu() * STD + MEAN
                heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
                raw_attention_image = raw_image * attention_maps

                for batch_idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * config.batch_size + batch_idx)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * config.batch_size + batch_idx)))

            # end of this batch
            batch_info = 'Val step {}'.format((i+1))
            pbar.update()
            pbar.set_postfix_str(batch_info)

            # 处理结果
            y_pred = F.softmax(y_pred,dim=1).cpu().numpy()
            result.append(y_pred)
        pbar.close()


    healthy = []
    multiple_disease = []
    rust = []
    scab = []
    for i in tqdm(range(len(result))):
        healthy.append(result[i][0][0])
        multiple_disease.append(result[i][0][1])
        rust.append(result[i][0][2])
        scab.append(result[i][0][3])
    sample_submission['healthy'] = healthy
    sample_submission['multiple_diseases'] = multiple_disease
    sample_submission['rust'] = rust
    sample_submission['scab'] = scab
    sample_submission.to_csv('submission.csv', index=False)
if __name__ == '__main__':
    main()
