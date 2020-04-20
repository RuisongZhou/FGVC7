#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 5:35 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
# from https://github.com/amdegroot/ssd.pytorch
import types
import cv2
import numpy as np
import torch
import numpy as np
from imgaug.augmenters import CoarseDropout, AdditiveGaussianNoise, GaussianBlur
import random

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):

        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels

class HorizontalFilp(object):
    def __call__(self, image, labels=None):
        if np.random.randint(2):
            image = cv2.flip(image, 1)
        return image, labels

class VerticalFlip(object):
    def __call__(self, image, labels=None):
        if np.random.randint(2):
            image = cv2.flip(image, 0)
        return image, labels

class dropout(object):
    def __init__(self,rate,size=None):
        self.trans = CoarseDropout(rate, size_percent=size)
    def __call__(self, image, label=None):
        return self.trans.augment_image(image), label

class addnoise(object):
    def __init__(self,scale):
        self.trans = AdditiveGaussianNoise(scale=scale)
    def __call__(self, image, label=None):
        return self.trans.augment_image(image), label

class addblur(object):
    def __init__(self, simga):
        self.trans = GaussianBlur(sigma=simga)
    def __call__(self, image, label=None):
        return self.trans.augment_image(image), label

class ConvertFromInts(object):
    def __call__(self, image, labels=None):
        return image.astype(np.float32), labels

class Resize(object):
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, image, labels=None):
        image = cv2.resize(image, (self.size[1],
                                   self.size[0]))
        return image, labels


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, labels=None):
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        return image, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, labels=None):
        if np.random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, labels


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, labels=None):
        if np.random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, labels=None):
        if np.random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, labels


class ToCV2Image(object):
    def __call__(self, tensor, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), labels


class ToTensor(object):
    def __call__(self, cvimage, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, labels=None):
        im = image.copy()
        im, labels = self.rand_brightness(im, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, labels = distort(im, labels)
        im, labels = self.rand_light_noise(im, labels)
        return im, labels

class RandomCrop(object):
    def __init__(self,size):
        self.size = size
    def __call__(self, image, labels=None):
        height, width, _ = image.shape      #400*640  -> 320*512
        x = random.randrange(height-self.size[0])
        y = random.randrange(width-self.size[1])
        return image[x:x+self.size[0], y:y+self.size[1]], labels

class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, image, labels=None):
        h, w,_  = image.shape
        center = (w // 2, h // 2)
        angle = random.randrange(self.angle)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated, labels