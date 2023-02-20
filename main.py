# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import tqdm

import faiss
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image

import attenuations
import data.augment_queries as augment_queries
import utils
import utils_img
from engine import activate_images

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--verbose", type=int, default=1)
    aa("--seed", type=int, default=0)

    group = parser.add_argument_group('Data parameters')
    aa("--fts_training_path", type=str, default="path/to/train/fts.pth")
    aa("--fts_reference_path", type=str, default="path/to/train/ref_990k.pth")
    aa("--data_dir", type=str, default="/path/to/disc/ref_10k.pth")
    aa("--query_nonmatch_dir", type=str, default="/path/to/disc/queries_40k")
    aa("--batch_size", type=int, default=16)
    aa("--batch_size_eval", type=int, default=128)
    aa("--resize_size", type=int, default=288, help="Resize images to this size. (Default: 288)")

    group = parser.add_argument_group('Model parameters')
    aa("--model_name", type=str, default="torchscript")
    aa("--model_path", type=str, default="/path/to/model.torchscript.pt")

    group = parser.add_argument_group('Index parameters')
    aa("--idx_dir", type=str, default="indexes", help="Directory where to save the index. (Default: index_disc_sscd288)")
    aa("--idx_factory", type=str, default="IVF4096,PQ8x8", help="String to create index from index factory. (Default: IVF4096,PQ8x8)")
    aa("--quant", type=str, default="L2", help="Quantizer type if IVF (L2, IP, etc.)")
    aa("--nprobe", type=int, default=1, help="Number of probes per query if IVF.")
    aa("--kneighbors", type=int, default=100, help="Number of nearest neighbors to return")

    group = parser.add_argument_group('Optimization parameters')
    aa("--iterations", type=int, default=10, help="Number of iterations for image optimization. (Default: 10)")
    aa("--optimizer", type=str, default="Adam,lr=1e-0", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--target", type=str, default="pq_recons", help="Target to use. (Default: pq_recons)")
    aa("--loss_f", type=str, default="cossim", help="Loss w to use. Choose among mse, cossim (Default: cossim)")
    aa("--lambda_f", type=float, default=1.0, help="Weight of the feature loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1e-2, help="Weight of the image loss. (Default: 1.0)")

    group = parser.add_argument_group('Distortion & Attenuation parameters')
    aa("--use_attenuation", type=utils.bool_inst, default=True, help="Use heatmap attenuation")
    aa("--scaling", type=float, default=3.0, help="Scaling factor for the heatmap attenuation")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Scale the RGB channels of the heatmap attenuation")
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh for the heatmap attenuation")

    group = parser.add_argument_group('Evaluation parameters')
    aa("--use_attacks_2", type=utils.bool_inst, default=False, help="Use attacks_2 for augmentation evaluation. (Default: False)")
    aa("--eval_retrieval", type=utils.bool_inst, default=True, help="Evaluate retrieval. (Default: True)")
    aa("--eval_icd", type=utils.bool_inst, default=True, help="Evaluate icd. (Default: True)")

    group = parser.add_argument_group('Misc parameters')
    aa("--active", type=utils.bool_inst, default=True, help="Activate images")
    aa("--save_imgs", type=utils.bool_inst, default=True, help="Save images")
    aa("--log_freq", type=int, default=11, help="Log every n iterations. (Default: 1)")
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode. (Default: False)")

    return parser

@torch.no_grad()
def eval_retrieval(img_loader, image_indices, transform, model, index, kneighbors, use_attacks_2=False):
    """
    Evaluate retrieval on the activated images.
    Args:
        img_loader (torch.utils.data.DataLoader): Data loader for the images.
        image_indices (list): List of ground-truth image indices.
        transform (torchvision.transforms): Transform to apply to the images.
        model (torch.nn.Module): Model to use for feature extraction.
        index (faiss.Index): Index to use for retrieval.
        kneighbors (int): Number of nearest neighbors to return.
        use_attacks_2 (bool): Use attacks_2 for augmentation evaluation. (Default: False)
    Returns:
        df (pandas.DataFrame): Dataframe with the results.
    """
    logs = []
    attacks = utils_img.attacks_2 if use_attacks_2 else utils_img.attacks 
    base_count = 0
    for ii, imgs in enumerate(tqdm.tqdm(img_loader)):

        # create attacks for each image of the batch
        attacked_imgs = [utils_img.generate_attacks(pil_img, attacks) for pil_img in imgs] # batchsize nattacks

        # create batches for each attack
        batch_attacked_imgs = [[] for _ in range(len(attacks))] # nattacks 0
        for jj, attacked_img_jj in enumerate(attacked_imgs):
            for kk in range(len(attacks)): # nattacks 0 -> nattacks batchsize
                img_jj_attack_kk = transform(attacked_img_jj[kk]).unsqueeze(0).to(device)
                batch_attacked_imgs[kk].append(img_jj_attack_kk) 
        batch_attacked_imgs = [torch.cat(batch_attacked_img, dim=0) for batch_attacked_img in batch_attacked_imgs] # nattacks batchsize

        # iterate over attacks
        for kk in range(len(attacks)):
            # create attack param
            attack = attacks[kk].copy()
            attack_name = attack.pop('attack')
            param_names = ['attack_param' for _ in range(len(attack.keys()))]
            attack_params = dict(zip(param_names,list(attack.values())))
            # extract features
            fts = model(batch_attacked_imgs[kk])
            fts = fts.detach().cpu().numpy()
            # retrieve nearest neighbors
            retrieved_Ds, retrieved_Is = index.search(fts, k=kneighbors)
            # iterate over images of the batch
            for jj in range(len(batch_attacked_imgs[kk])):
                image_index = image_indices[base_count+jj]
                retrieved_D, retrieved_I = retrieved_Ds[jj], retrieved_Is[jj]
                rank = [kk for kk in range(len(retrieved_I)) if retrieved_I[kk]==image_index]
                rank = rank[0] if rank else len(retrieved_I)
                logs.append({
                    'batch': ii, 
                    'image_index': image_index, 
                    "attack": attack_name,
                    **attack_params,
                    'retrieved_distances': retrieved_D,
                    'retrieved_indices': retrieved_I,
                    'rank': rank,
                    'r@1': 1 if rank<1 else 0,
                    'r@10': 1 if rank<10 else 0,
                    'r@100': 1 if rank<100 else 0,
                    'ap': 1/(rank+1),
                    "kw": "evaluation",
                })  
        
        # update count of images
        base_count += len(imgs)

    df = pd.DataFrame(logs).drop(columns='kw')
    return df

@torch.no_grad()
def eval_icd(img_loader, img_nonmatch_loader, image_indices, transform, model, index, kneighbors, seed=0):
    """
    Evaluate icd on the activated images.
    Args:
        img_loader (torch.utils.data.DataLoader): Data loader for the images.
        img_nonmatch_loader (torch.utils.data.DataLoader): Data loader for the non-matching images.
        image_indices (list): List of ground-truth image indices.
        transform (torchvision.transforms): Transform to apply to the images.
        model (torch.nn.Module): Model to use for feature extraction.
        index (faiss.Index): Index to use for retrieval.
        kneighbors (int): Number of nearest neighbors to return.
        query_nonmatch_dir (str): Directory where the non-matching images are stored.
        seed (int): Seed for the random number generator. (Default: 0)
    Returns:
        df (pandas.DataFrame): Dataframe with the results.
    """
    # stats on matching images
    rng = np.random.RandomState(seed)
    logs = []
    ct_match = 0 # count of matching images
    for ii, imgs in enumerate(tqdm.tqdm(img_loader)):
        # create attack for each image of the batch
        attacked_imgs = []
        attack_names = []
        for jj, pil_img in enumerate(imgs):
            attacked_img, aug_params = augment_queries.augment_img_wrapper(pil_img, rng, return_params=True)
            attack_name = "[" + ", ".join([str(ftr) for ftr in aug_params])
            attacked_img = transform(attacked_img).unsqueeze(0).to(device)
            attack_names.append(attack_name)
            attacked_imgs.append(attacked_img)
        attacked_imgs = torch.cat(attacked_imgs, dim=0)
        # extract features
        fts = model(attacked_imgs)
        fts = fts.detach().cpu().numpy()
        # nearest neighbors search
        retrieved_Ds, retrieved_Is = index.search(fts, k=kneighbors)
        # iterate over images of the batch
        for jj in range(len(imgs)):
            retrieved_D, retrieved_I = retrieved_Ds[jj], retrieved_Is[jj]
            image_index = image_indices[ct_match + jj]
            logs.append({
                'batch': ii, 
                'image_index': image_index, 
                'attack': attack_names[jj],
                'scores': retrieved_D,
                'retrieved_ids': retrieved_I,
                "kw": "icd_evaluation",
            })

        # update count of matching images
        ct_match += len(imgs)

    # stats non matching images
    for ii, imgs in enumerate(tqdm.tqdm(img_nonmatch_loader)):
        # create attack for each image of the batch
        attacked_imgs = []
        attack_names = []
        for jj, pil_img in enumerate(imgs):
            attacked_img, aug_params = augment_queries.augment_img_wrapper(pil_img, rng, return_params=True)
            attack_name = "[" + ", ".join([str(ftr) for ftr in aug_params])
            attacked_img = transform(attacked_img).unsqueeze(0).to(device)
            attack_names.append(attack_name)
            attacked_imgs.append(attacked_img)
        attacked_imgs = torch.cat(attacked_imgs, dim=0)
        # extract features
        fts = model(attacked_imgs)
        fts = fts.detach().cpu().numpy()
        # nearest neighbors search
        retrieved_Ds, retrieved_Is = index.search(fts, k=kneighbors)
        # iterate over images of the batch
        for jj in range(len(imgs)):
            retrieved_D, retrieved_I = retrieved_Ds[jj], retrieved_Is[jj]
            logs.append({
                'batch': ii, 
                'image_index': -1, 
                'attack': attack_names[jj],
                'scores': retrieved_D,
                'retrieved_ids': retrieved_I,
                "kw": "icd_evaluation",
            })

    icd_df = pd.DataFrame(logs).drop(columns='kw')
    return icd_df



def main(params):

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    # Create the directories
    os.makedirs(params.idx_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    os.makedirs(imgs_dir, exist_ok=True)
    print(f'>>> Starting. \n \t Index will be saved in {params.idx_dir} - images will be saved in {imgs_dir} - evaluation logs in {params.output_dir}')

    # Build Index - see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    print(f'>>> Building Index')
    idx_path = os.path.join(params.idx_dir, f'idx={params.idx_factory}_quant={params.quant}.index')
    if os.path.exists(idx_path):
        print(f'>>> Loading Index from {idx_path}')
        index = faiss.read_index(idx_path)
    else:
        print(f'>>> Index not found. Building Index with fts from {params.fts_training_path}...')
        index = utils.build_index_factory(params.idx_factory, params.quant, params.fts_training_path, idx_path)
    index.nprobe = params.nprobe
    if 'IVF' in params.idx_factory:  # optionally get the centroids
        ivf = faiss.extract_index_ivf(index)
        ivf_centroids = ivf.quantizer.reconstruct_n(0, ivf.nlist)
    else:
        ivf_centroids = None
    
    # Adding reference images to the index
    print(f'>>> Adding reference images to the index from {params.fts_reference_path}...')
    fts = torch.load(params.fts_reference_path)
    index.add(fts.detach().cpu().numpy())
    n_index_ref = index.ntotal
    if 'IVF' in params.idx_factory:
        ivf.make_direct_map()

    # Build the feature extractor model
    print(f'>>> Building backbone from {params.model_path}...')
    model = utils.build_backbone(path=params.model_path, name=params.model_name)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # loss for feature 
    cossim = nn.CosineSimilarity(dim=-1)
    pdist = nn.PairwiseDistance(p=2)
    def loss_f(ft, target):
        if params.loss_f == 'cossim':
            dists = -cossim(ft, target)
        else:
            dists = pdist(ft, target)**2
        return torch.mean(dists)
    
    # loss for image
    mse = nn.MSELoss()
    def loss_i(imgs, imgs_ori):
        li = 0
        bb = len(imgs)
        for ii in range(bb): # imgs do not have same size so we cannot use batch mse
            li += mse(imgs[ii], imgs_ori[ii])
        return li/bb

    # build perceptual attenuation
    attenuation = attenuations.JND(preprocess = utils_img.UNNORMALIZE_IMAGENET).to(device)
    attenuation.requires_grad = False

    # Load images to activate
    print(f'>>> Loading images from {params.data_dir}...')
    transform = transforms.Compose([
        transforms.ToTensor(), 
        utils_img.NORMALIZE_IMAGENET,
    ])
    transform_with_resize = transforms.Compose([
        transforms.ToTensor(), 
        utils_img.NORMALIZE_IMAGENET,
        transforms.Resize((params.resize_size, params.resize_size)),
    ])
    data_loader = utils.get_dataloader(params.data_dir, transform, params.batch_size)

    print(f'>>> Activating images...')
    all_imgs = []
    for it, imgs in enumerate(tqdm.tqdm(data_loader)):

        if params.debug and it > 5:
            break

        imgs = [img.to(device) for img in imgs]

        # Add to index
        resized_imgs = [functional.resize(img, (params.resize_size, params.resize_size)) for img in imgs]
        batch_imgs = torch.stack([img for img in resized_imgs])
        fts = model(batch_imgs)
        index.add(fts.detach().cpu().numpy())
        if 'IVF' in params.idx_factory:
            ivf.make_direct_map() # update the direct map if needed

        # Activate
        if params.active:
            imgs = activate_images(imgs, fts, model, index, ivf_centroids, attenuation, loss_f, loss_i, params)

        # Save images
        for ii, img in enumerate(imgs):
            img = torch.clamp(utils_img.UNNORMALIZE_IMAGENET(img), 0, 1) 
            img = torch.round(255 * img)/255 
            img = img.detach().cpu() 
            if params.save_imgs:
                save_image(img, os.path.join(imgs_dir, f'{it*params.batch_size + ii:05d}.png'))
            else:
                all_imgs.append(transforms.ToPILImage()(img))
    
    if params.save_imgs:
        # create loader from saved images
        img_loader = utils.get_dataloader(imgs_dir, transform=None, batch_size=params.batch_size_eval)
    else:
        # list of images to list of batches
        img_loader = [all_imgs[ii:ii + params.batch_size_eval] for ii in range(0, len(all_imgs), params.batch_size_eval)] 

    if params.eval_retrieval:
        print(f'>>> Evaluating nearest neighbors search...')
        image_indices = range(n_index_ref, index.ntotal)
        df = eval_retrieval(img_loader, image_indices, transform_with_resize, model, index, params.kneighbors, params.use_attacks_2)
        df.to_csv(os.path.join(params.output_dir, 'retr_df.csv'), index=False)
        df.fillna(0, inplace=True)
        df_mean = df.groupby(['attack', 'attack_param'], as_index=False).mean()
        print(f'\n{df_mean}')

    if params.eval_icd:
        print(f'>>> Evaluating copy detection on query set...')
        image_indices = range(n_index_ref, index.ntotal)
        img_nonatch_loader = utils.get_dataloader(params.query_nonmatch_dir, transform=None, batch_size=params.batch_size_eval)
        icd_df = eval_icd(img_loader, img_nonatch_loader, image_indices, transform_with_resize, model, index, params.kneighbors)
        icd_df_path = os.path.join(params.output_dir,'icd_df.csv')
        icd_df.to_csv(icd_df_path, index=False)
        print(f'\n{icd_df}')


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)


#  python main.py --fts_training_path /checkpoint/pfz/2023_logs/0216_extractfts_sscd/_model_path=0_data_dir=0/fts.pth --fts_reference_path /checkpoint/pfz/2023_logs/0216_extractfts_sscd/_model_path=0_data_dir=1/fts.pth --data_dir /checkpoint/pfz/datasets/disc_prepared/references_10k --query_nonmatch_dir /checkpoint/pfz/datasets/disc_prepared/queries_40k --model_path /checkpoint/pfz/watermarking/models/sscd/sscd_disc_advanced.torchscript.pt --batch_size 16 --debug True --active True