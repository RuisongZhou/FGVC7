#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 5:11 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com

import os, sys
import time
import logging
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from config import Config
from models import *
from dataset.dataset import FGVC7Data
from utils.utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment, get_transform
from utils.loss import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='./data/', help='Train Dataset directory path')
parser.add_argument('--net', default='b0', help='Choose net to use')
parser.add_argument('--bs', default=24, type=int,  help='batch size')
parser.add_argument('--ckpt', default=None, type=str, help='resume train')
parser.add_argument('--epochs', default=40, type=int,  help='epoch size')
parser.add_argument('--loss', default='all', type=str, help='choose loss')
args = parser.parse_args()

config = Config()
#others
config.batch_size = args.bs
config.net = args.net
config.ckpt = args.ckpt
config.epochs = args.epochs
config.refresh()

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

# General loss functions
ce_weight = 1.0
arc_weight = 0
if args.loss == 'all':
    ce_weight = arc_weight = 0.5
elif args.loss == 'ce':
    ce_weight = 1.0
    arc_weight = 0
elif args.loss == 'arc' :
    ce_weight = 0
    arc_weight = 1.0
criterion = Criterion(weight_arcface=arc_weight, weight_ce=ce_weight)
# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1,2))

def choose_net(name: str):
    if len(name) == 2 and name[0] == 'b':
        model = efficientnet(size=name)
    elif name.lower() == 'seresnext50':
        model = se_resnext50()
    elif name.lower() == 'seresnext101':
        model = se_resnext101()
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    return model

def main():
    ##################################
    # Initialize saving directory
    ##################################
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Load dataset  TODO: 10-fold cross validation
    ##################################
    train_dataset = FGVC7Data(root='./data/', phase='train', transform=get_transform(config.image_size, 'train'))
    indices = range(len(train_dataset))
    split = int(0.1 * len(train_dataset))
    train_indices = indices[split:]
    test_indices = indices[:split]
    #train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True)
    validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=valid_sampler,
                              num_workers=config.workers, pin_memory=True)

    num_classes = 4
    print('Train Size: {}'.format(len(train_indices)))
    print('Valid Size: {}'.format(len(test_indices)))
    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = choose_net(args.net)

    if config.ckpt:
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])
        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################x
    # Use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Optimizer, LR Schedulerextract_features(img)
    ##################################
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(net.parameters(),lr=learning_rate, amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, eta_min = 1e-6)

    ##################################
    # ModelCheckpoint
    ##################################
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

        ##################################
        # TRAINING
        ##################################
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_indices), len(test_indices)))
    logging.info('')

    for epoch in range(start_epoch, config.epochs):
        callback.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, LR {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(logs=logs,
              data_loader=train_loader,
              net=net,
              optimizer=optimizer,
              pbar=pbar)
        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()

        callback.on_epoch_end(logs, net)
        pbar.close()

def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    for i, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()

        # obtain data for training
        X = X.to(device)
        y = y.to(device)
        out = net(X)
        # loss
        batch_loss = criterion(out, y)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            y_pred_raw, _ = out
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)

        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f} {:.2f})'.format(
            epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1])
        pbar.update()
        pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))

def validate(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.to(device)
            y = y.to(device)

            ##################################
            # Raw Image
            ##################################
            y_pred , y_arc = net(X)
            # loss
            batch_loss = criterion.ce_forward(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


if __name__ == '__main__':
    main()
