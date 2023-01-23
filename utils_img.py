
import numpy as np

import torch

from torchvision.transforms import functional
from augly.image import functional as aug_functional

import augment_queries


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

attacks_dict = {
    "none": lambda x : x,
    "rotation": lambda x, angle: functional.rotate(x, angle, functional.InterpolationMode('bilinear'), expand=True),
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