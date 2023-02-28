# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable
import argparse
import json
import time

import faiss
import numpy as np

import torch
from torch import nn
from torchvision.transforms import functional

import utils
import utils_img
from attenuations import JND

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_targets(
        target: str, 
        index: faiss.Index,
        fts: torch.Tensor,
        ivf_centroids: np.ndarray = None
    ) -> torch.Tensor:
    """
    Get the target representations for the features.
    Args:
        target (str): Target representation to use.
        index (faiss.Index): Index to use for retrieval.
        fts (torch.Tensor): Features to get the targets for. batch_size x feature_dim
        ivf_centroids (np.ndarray): Centroids of the IVF index.
    Returns:
        targets (torch.Tensor): Target representations for the features. batch_size x feature_dim
    """
    if target == 'pq_recons':
        targets = index.reconstruct_n(index.ntotal-fts.shape[0], fts.shape[0]) # reconstruct the PQ codes that have just been added
        targets = torch.tensor(targets)
    elif target == 'ori_ft':
        fts.clone()
    elif target == 'ivf_cluster':
        ivf_D, ivf_I = index.quantizer.search(fts.detach().cpu().numpy(), k=1) # find the closest cluster center for each feature
        targets = ivf_centroids.take(ivf_I.flatten(), axis=0) # get the cluster representation for each feature
        targets = torch.tensor(targets)
    elif target == 'ivf_cluster_half':
        ivf_D, ivf_I = index.quantizer.search(fts.detach().cpu().numpy(), k=1)
        centroids = ivf_centroids.take(ivf_I.flatten(), axis=0)
        targets = (torch.tensor(centroids) + fts.clone() / 2)
    else:
        raise NotImplementedError(f'Invalid target: {target}')
    return targets


def activate_images(
        imgs: list[torch.Tensor],
        ori_fts: torch.Tensor,
        model: nn.Module, 
        index: faiss.Index, 
        ivf_centroids: np.ndarray, 
        attenuation: JND, 
        loss_f: Callable,
        loss_i: Callable, 
        params: argparse.Namespace
    ) -> list[torch.Tensor]:
    """
    Activate images.
    Args:
        imgs (list of torch.Tensor): Images to activate. batch_size * [3 x height x width]
        model (torch.nn.Module): Model for feature extraction.
        index (faiss.Index): Index to use for retrieval.
        ivf_centroids (np.ndarray): Centroids of the IVF index.
        attenuation (JND): To create Just Noticeable Difference heatmaps.
        loss_f (Callable): Loss function to use for the indexation loss.
        loss_i (Callable): Loss function to use for the image loss.
        params (argparse.Namespace): Parameters.
    Returns:
        activated images (list of torch.Tensor): Activated images. batch_size * [3 x height x width]
    """
    targets = get_targets(params.target, index, ori_fts, ivf_centroids)
    targets = targets.to(device)

    # Just noticeable difference heatmaps
    alpha = torch.tensor([0.072*(1/0.299), 0.072*(1/0.587), 0.072*(1/0.114)])
    alpha = alpha[:,None,None].to(device) # 3 x 1 x 1
    heatmaps = [params.scaling * attenuation.heatmaps(img) for img in imgs]

    # init distortion + optimizer + scheduler
    deltas = [1e-6 * torch.randn_like(img).to(device) for img in imgs] # b (1 c h w)
    for distortion in deltas:
        distortion.requires_grad = True
    optim_params = utils.parse_params(params.optimizer)
    optimizer = utils.build_optimizer(model_params=deltas, **optim_params)
    if params.scheduler is not None:
        scheduler = utils.build_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler))
    
    # begin optim
    iter_time = time.time()
    log_stats = []
    for gd_it in range(params.iterations):
        gd_it_time = time.time()
        if params.scheduler is not None:
            scheduler.step(gd_it)

        # perceptual constraints
        percep_deltas = [torch.tanh(delta) for delta in deltas] if params.use_tanh else deltas
        percep_deltas = [delta * alpha for delta in percep_deltas] if params.scale_channels else percep_deltas
        imgs_t = [img + hm * delta for img, hm, delta in zip(imgs, heatmaps, percep_deltas)]

        # get features
        batch_imgs = [functional.resize(img_t, (params.resize_size, params.resize_size)) for img_t in imgs_t]
        batch_imgs = torch.stack(batch_imgs)
        fts = model(batch_imgs) # b d

        # compute losses
        lf = loss_f(fts, targets)
        li = loss_i(imgs_t, imgs)
        loss = params.lambda_f * lf + params.lambda_i * li

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log stats
        psnrs = torch.tensor([utils_img.psnr(img_t, img) for img_t, img in zip(imgs_t, imgs)])
        linfs = torch.tensor([utils_img.linf(img_t, img) for img_t, img in zip(imgs_t, imgs)])
        log_stats.append({
            'gd_it': gd_it,
            'loss': loss.item(),
            'loss_f': lf.item(),
            'loss_i': li.item(),
            'psnr': torch.nanmean(psnrs).item(),
            'linf': torch.nanmean(linfs).item(),
            'lr': optimizer.param_groups[0]['lr'],
            'gd_it_time': time.time() - gd_it_time,
            'iter_time': time.time() - iter_time,
            'max_mem': torch.cuda.max_memory_allocated() / (1024*1024),
            'kw': 'optim',
        })
        if (gd_it+1) % params.log_freq == 0:
            print(json.dumps(log_stats[-1]))
            # tqdm.tqdm.write(json.dumps(log_stats[-1]))
    
    # perceptual constraints
    percep_deltas = [torch.tanh(delta) for delta in deltas] if params.use_tanh else deltas
    percep_deltas = [delta * alpha for delta in percep_deltas] if params.scale_channels else percep_deltas
    imgs_t = [img + hm * delta for img, hm, delta in zip(imgs, heatmaps, percep_deltas)]

    return imgs_t
