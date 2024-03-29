# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from augly.image import functional as aug_functional

import torch
from torchvision import transforms
from torchvision.transforms import functional

from . import augment_queries

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.size][::-1]    
    return functional.resize(x, new_edges_size)

def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x, y: Images tensor with imagenet normalization
    """  
    delta = 255 * (x - y) * image_std.to(x.device)
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    return psnr

def linf(x, y):
    """ 
    Return Linf distance 
    Args:
        x, y: Images tensor with imagenet normalization
    """  
    return torch.max(torch.abs(255 * (x - y) * image_std.to(x.device)))

attacks_dict = {
    "none": lambda x : x,
    "rotation": lambda x, angle: functional.rotate(x, angle, functional.InterpolationMode('bilinear'), expand=True),
    "grayscale": functional.rgb_to_grayscale,
    "contrast": functional.adjust_contrast,
    "brightness": functional.adjust_brightness,
    "hue": functional.adjust_hue,
    "hflip": functional.hflip,
    "vflip": functional.vflip,
    "blur": functional.gaussian_blur, # sigma = ksize*0.15 + 0.35  - ksize = (sigma-0.35)/0.15
    "jpeg": aug_functional.encoding_quality,
    "resize": resize,
    "center_crop": center_crop,
    "meme_format": aug_functional.meme_format,
    "overlay_emoji": aug_functional.overlay_emoji,
    "overlay_onto_screenshot": aug_functional.overlay_onto_screenshot,
    "auto": augment_queries.augment_img,
}

attacks = [{'attack': 'none'}] \
    + [{'attack': 'auto'}] \
    + [{'attack': 'meme_format'}] \
    + [{'attack': 'overlay_onto_screenshot'}] \
    + [{'attack': 'rotation', 'angle': angle} for angle in [25,90]] \
    + [{'attack': 'center_crop', 'scale': 0.5}] \
    + [{'attack': 'resize', 'scale': 0.5}] \
    + [{'attack': 'blur', 'kernel_size': 11}] \
    + [{'attack': 'jpeg', 'quality': 50}] \
    + [{'attack': 'hue', 'hue_factor': 0.2}] \
    + [{'attack': 'contrast', 'contrast_factor': cf} for cf in [0.5, 2.0]] \
    + [{'attack': 'brightness', 'brightness_factor': bf} for bf in [0.5, 2.0]] \

# more attacks for the full evaluation
attacks_2 = [{'attack': 'rotation', 'angle': jj} for jj in range(-90, 100,10)] \
    + [{'attack': 'center_crop', 'scale': 0.1*jj} for jj in range(1,11)] \
    + [{'attack': 'resize', 'scale': 0.1*jj} for jj in range(1,11)] \
    + [{'attack': 'blur', 'kernel_size': 1+2*jj} for jj in range(1,15)] \
    + [{'attack': 'jpeg', 'quality': 10*jj} for jj in range(1,11)] \
    + [{'attack': 'contrast', 'contrast_factor': 0.5 + 0.1*jj} for jj in range(15)] \
    + [{'attack': 'brightness', 'brightness_factor': 0.5 + 0.1*jj} for jj in range(15)] \
    + [{'attack': 'hue', 'hue_factor': -0.5 + 0.1*jj} for jj in range(0,11)] \

def generate_attacks(img, attacks=attacks):
    """ Generate a list of attacked images from a PIL image. """
    attacked_imgs = []
    for attack in attacks:
        attack = attack.copy()
        attack_name = attack.pop('attack')
        attacked_imgs.append(attacks_dict[attack_name](img, **attack))
    return attacked_imgs