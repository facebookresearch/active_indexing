

import os
import subprocess
import timm
import numpy as np

import functools, logging

import torch
import torch.nn as nn
from torchvision import models, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import is_image_file, default_loader

from timm import optim, scheduler
import faiss

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)


## Index

def build_index_factory(idx_str, quant, fts_path, idx_path=None) -> faiss.Index:
    """ 
    Builds index from string and fts_path.
    https://github.com/facebookresearch/faiss/wiki/The-index-factory 
    """
    print(f'Index not found. Building Index with fts from {fts_path}...')
    fts = torch.load(fts_path)
    fts = fts.detach().cpu().numpy()
    D = fts.shape[1]
    metric = faiss.METRIC_L2 if quant == 'L2' else faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(D, idx_str, metric)
    index.train(fts)
    if idx_path is not None:
        print(f'Saving Index to {idx_path}...')
        faiss.write_index(index, idx_path)
    return index

### Exp

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

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

def build_optimizer(name, model_params, **optim_params):
    """ Build optimizer from a dictionary of parameters """
    tim_optimizers = sorted(name for name in optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(optim.__dict__[name]))
    torch_optimizers = sorted(name for name in torch.optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.__dict__[name]))
    if hasattr(optim, name):
        return getattr(optim, name)(model_params, **optim_params)
    elif hasattr(torch.optim, name):
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
    tim_schedulers = sorted(name for name in scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(scheduler.__dict__[name]))
    torch_schedulers = sorted(name for name in torch.optim.lr_scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.lr_scheduler.__dict__[name]))
    if hasattr(scheduler, name):
        return getattr(scheduler, name)(optimizer, **lr_scheduler_params)
    elif hasattr(torch.optim.lr_scheduler, name):
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **lr_scheduler_params)
    raise ValueError(f'Unknown scheduler "{name}", choose among {str(tim_schedulers+torch_schedulers)}')

### Model

def build_backbone(path, name):
    """ Build a pretrained torchvision backbone from its name.

    Args:
        path: path to the checkpoint, can be an URL
        name: name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html) 
        or timm (see https://rwightman.github.io/pytorch-image-models/models/). 
    """
    if name == 'torchscript':
        model = torch.jit.load(path)
        return model.to(device, non_blocking=True)
    if hasattr(models, name):
        model = getattr(models, name)(pretrained=True)
    else:
        if name in timm.list_models():
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
    return model.to(device, non_blocking=True)

### Data loading

@functools.lru_cache()
def get_image_paths(path):
    logging.info(f"Resolving files in: {path}")
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])

class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

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

def get_dataloader(data_dir, transform, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_fn):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    custom = True
    dataset = ImageFolder(data_dir, transform=transform) if custom else datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    return dataloader

def pil_imgs_from_folder(folder):
    """ Get all images in the folder as PIL images """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder,filename))
            if img is not None:
                filenames.append(filename)
                images.append(img)
        except:
            print("Error opening image: ", filename)
    return images, filenames

### Image

def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """  
    delta = 255 * (x - y) * image_std.to(x.device)
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    return psnr

def linf(x, y):
    """ 
    Return Linf distance 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """  
    return torch.max(torch.abs(255 * (x - y) * image_std.to(x.device)))



