# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, functools, logging

import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import DataLoader
from torchvision.datasets.folder import is_image_file, default_loader

import faiss
import timm
from timm import optim as timm_optim
from timm import scheduler as timm_scheduler

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Index

def build_index_factory(idx_str, quant, fts_path, idx_path=None) -> faiss.Index:
    """ 
    Builds index from string and fts_path. see https://github.com/facebookresearch/faiss/wiki/The-index-factory 
    Args:
        idx_str: string describing the index
        quant: quantization type, either "L2" or "IP" (Inner Product)
        fts_path: path to the train features as a torch tensor .pt file
        idx_path: path to save the index
    """
    fts = torch.load(fts_path)
    fts = fts.numpy() # b d
    D = fts.shape[-1]
    metric = faiss.METRIC_L2 if quant == 'L2' else faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(D, idx_str, metric)
    index.train(fts)
    if idx_path is not None:
        print(f'Saving Index to {idx_path}...')
        faiss.write_index(index, idx_path)
    return index

# Arguments helpers

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example: 
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]]=float(x[1])
    return params

# Optimizer and Scheduler

def build_optimizer(name, model_params, **optim_params):
    """ Build optimizer from a dictionary of parameters """
    tim_optimizers = sorted(name for name in timm_optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(timm_optim.__dict__[name]))
    torch_optimizers = sorted(name for name in torch.optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.__dict__[name]))
    if name in tim_optimizers:
        return getattr(timm_optim, name)(model_params, **optim_params)
    elif name in torch_optimizers:
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}", choose among {str(tim_optimizers+torch_optimizers)}')

def build_scheduler(name, optimizer, **lr_scheduler_params):
    """ 
    Build scheduler from a dictionary of parameters 
    Args:
        name: name of the scheduler
        optimizer: optimizer to be used with the scheduler
        params: dictionary of scheduler parameters
    Ex:
        CosineLRScheduler, optimizer {t_initial=50, cycle_mul=2, cycle_limit=3, cycle_decay=0.5, warmup_lr_init=1e-6, warmup_t=5}
    """
    tim_schedulers = sorted(name for name in timm_scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(timm_scheduler.__dict__[name]))
    torch_schedulers = sorted(name for name in torch.optim.lr_scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.lr_scheduler.__dict__[name]))
    if name in tim_schedulers:
        return getattr(timm_scheduler, name)(optimizer, **lr_scheduler_params)
    elif hasattr(torch.optim.lr_scheduler, name):
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **lr_scheduler_params)
    raise ValueError(f'Unknown scheduler "{name}", choose among {str(tim_schedulers+torch_schedulers)}')

# Model

def build_backbone(path, name):
    """ Build a pretrained torchvision backbone from its name.
    Args:
        path: path to the checkpoint, can be an URL
        name: "torchscript" or name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html) 
        or timm (see https://rwightman.github.io/pytorch-image-models/models/). 
    Returns:
        model: nn.Module
    """
    if name == 'torchscript':
        model = torch.jit.load(path)
        return model
    else:
        if hasattr(models, name):
            model = getattr(models, name)(pretrained=True)
        elif name in timm.list_models():
            model = timm.models.create_model(name, num_classes=0)
        else:
            raise NotImplementedError('Model %s does not exist in torchvision'%name)
        model.head = nn.Identity()
        model.fc = nn.Identity()
        if path is not None:
            if path.startswith("http"):
                checkpoint = torch.hub.load_state_dict_from_url(path, progress=False)
            else:
                checkpoint = torch.load(path)
            state_dict = checkpoint
            for ckpt_key in ['state_dict', 'model_state_dict', 'teacher']:
                if ckpt_key in checkpoint:
                    state_dict = checkpoint[ckpt_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
        return model

# Data loading

@functools.lru_cache()
def get_image_paths(path):
    logging.info(f"Resolving files in: {path}")
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])

class ImageFolder:
    """An image folder dataset without classes"""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch

def get_dataloader(data_dir, transform, batch_size=128, num_workers=8, collate_fn=collate_fn):
    """ Get dataloader for the images in the data_dir. """
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False, pin_memory=True, drop_last=False)
    return dataloader
