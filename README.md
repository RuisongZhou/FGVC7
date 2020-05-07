# FGVC7
This repository is for Kaggle Competetion [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/).
And our rank is 15/1115.

## score

|model|score(AUC)|
|:---: | :---: | 
|SeResNet50|96.0|
|SeResNet101|96.7|
|EffcientNet b4| 97.5|
|EffcientNet b6| 97.7|
|ResNest101| 97.3|
|ResNest200| 97.3|
|WSDAN*| 97.0|
|merge**| 98.4|

*. WSDAN model is from pytorch [implement](https://github.com/GuYuc/WS-DAN.PyTorch) of 
[See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification](https://arxiv.org/abs/1901.09891v2)

**. merge model is the merge of EffcientNet b4, EffcientNet b6(2 models), ResNest101. 
#

# RUN
```shell
python train.py --net $YOURNET --loss all
```
YOURNET should be in b0~b7, seresnet50, seresnet101,resnest101, resnest200, resnest269


# EVAL

```shell
python eval_tta.py --net $YOURNET
```