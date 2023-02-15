import argparse, random, os, tqdm

import numpy as np

import faiss
import pandas as pd
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image

import attenuations
import utils
import utils_img
import data.augment_queries as augment_queries
from activate import activate_images

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
    aa("--query_match_dir", type=str, default=None)
    aa("--query_nonmatch_dir", type=str, default="/path/to/disc/queries_40k")
    aa("--batch_size", type=int, default=16)
    aa("--resize", type=utils.bool_inst, default=True, help="Resize images before feature extraction. (Default: True)")
    aa("--resize_size", type=int, default=288, help="Resize images to this size. (Default: 288)")

    group = parser.add_argument_group('Model parameters')
    aa("--model_path", type=str, default="/path/to/model.torchscript.pt")
    aa("--model_name", type=str, default="torchscript")

    group = parser.add_argument_group('Index parameters')
    aa("--idx_dir", type=str, default="index_disc_sscd288", help="Directory where to save the index. (Default: index_disc_sscd288)")
    aa("--idx_factory", type=str, default="IVF4096,PQ8x8", help="String to create index from index factory. (Default: IVF4096,PQ8x8)")
    aa("--quant", type=str, default="L2", help="Quantizer type if IVF (L2, IP, etc.)")
    aa("--nprobe", type=int, default=1, help="Number of probes per query if IVF.")
    aa("--kneighbors", type=int, default=100, help="Number of nearest neighbors to return")

    group = parser.add_argument_group('Optimization parameters')
    aa("--iterations", type=int, default=10, help="Number of iterations for image optimization. (Default: 10)")
    aa("--optimizer", type=str, default="Adam,lr=1e-0", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--target", type=str, default="pq_recons", help="Target to use. (Default: pq_recons)")
    aa("--loss_f", type=str, default="cossim", help="Loss w to use. Choose among mse, eot, mot, smot (Default: cossim)")
    aa("--lambda_f", type=float, default=1.0, help="Weight of the feature loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss. (Default: 1.0)")

    group = parser.add_argument_group('Distortion & Attenuation parameters')
    aa("--use_attenuation", type=utils.bool_inst, default=True, help="Use heatmap attenuation")
    aa("--scaling", type=float, default=3.0, help="Scaling factor for the heatmap attenuation")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Scale the RGB channels of the heatmap attenuation")
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh for the heatmap attenuation")

    group = parser.add_argument_group('Evaluation parameters')
    aa("--only_1k", type=utils.bool_inst, default=False, help="Only evaluate on 1k images. (Default: False)")
    aa("--use_attacks_2", type=utils.bool_inst, default=False, help="Use attacks_2 for augmentation evaluation. (Default: False)")
    aa("--eval_retrieval", type=utils.bool_inst, default=True, help="Evaluate retrieval. (Default: True)")
    aa("--eval_icd", type=utils.bool_inst, default=True, help="Evaluate icd. (Default: True)")

    group = parser.add_argument_group('Misc parameters')
    aa("--save_imgs", type=utils.bool_inst, default=True, help="Save images")
    aa("--post_process", type=utils.bool_inst, default=True, help="Post process images")
    aa("--log_freq", type=int, default=1, help="Log every n iterations. (Default: 1)")
    aa("--active", type=utils.bool_inst, default=True, help="Activate images")
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def eval_retrieval(imgs_dir, image_indices, transform, model, index, kneighbors, use_attacks_2=False):
    """
    Evaluate retrieval on the activated images.
    Args:
        imgs_dir (str): Directory where the images are stored.
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
    data_loader = utils.get_dataloader(imgs_dir, transform=None, batch_size=1, shuffle=False)
    attacks = utils_img.attacks_2 if use_attacks_2 else utils_img.attacks 
    for ii, img in enumerate(tqdm.tqdm(data_loader)):
        image_index = image_indices[ii]
        pil_img = img[0]
        # pil_img = transforms.ToPILImage()(img)
        attacked_imgs = utils_img.generate_attacks(pil_img, attacks)
        if ii==0:
            for jj in range(len(utils_img.attacks)):
                attacked_imgs[jj].save(os.path.join(imgs_dir,"%i_%s.jpg"%(ii, str(utils_img.attacks[jj])) ))
        for jj, attacked_img in enumerate(attacked_imgs):
            attacked_img = transform(attacked_img).unsqueeze(0).to(device)
            ft = model(attacked_img)
            ft = ft.detach().cpu().numpy()
            retrieved_D, retrieved_I = index.search(ft, k=kneighbors)
            retrieved_D, retrieved_I = retrieved_D[0], retrieved_I[0]
            rank = [kk for kk in range(len(retrieved_I)) if retrieved_I[kk]==image_index]
            rank = rank[0] if rank else len(retrieved_I)
            attack = attacks[jj].copy()
            attack_name = attack.pop('attack')
            param_names = ['param%i'%kk for kk in range(len(attack.keys()))]
            attack_params = dict(zip(param_names,list(attack.values())))
            logs.append({
                'image': ii, 
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
    df = pd.DataFrame(logs).drop(columns='kw')
    return df


def eval_icd(imgs_dir, image_indices, transform, model, index, kneighbors, query_nonmatch_dir, seed=0):
    """
    Evaluate icd on the activated images.
    Args:
        imgs_dir (str): Directory where the images are stored.
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
    query_match_dir = imgs_dir
    data_loader = utils.get_dataloader(query_match_dir, transform=None, batch_size=1, shuffle=False)
    rng = np.random.RandomState(seed)
    logs = []
    ct_match = 0
    for ii, img in enumerate(tqdm.tqdm(data_loader)):
        image_index = image_indices[ii]
        pil_img = img[0]
        attacked_img, aug_params = augment_queries.augment_img(pil_img, rng, return_params=True)
        attack_name = "[" + ", ".join([str(ftr) for ftr in aug_params])
        attacked_img = transform(attacked_img).unsqueeze(0).to(device)
        ft = model(attacked_img)
        ft = ft.detach().cpu().numpy()
        retrieved_D, retrieved_I = index.search(ft, k=kneighbors)
        retrieved_D, retrieved_I = retrieved_D[0], retrieved_I[0]
        logs.append({
            'image': ii, 
            'image_index': image_index, 
            'attack': attack_name,
            'scores': retrieved_D,
            'retrieved_ids': retrieved_I,
            "kw": "icd_evaluation",
        })
        ct_match +=1
    # stats non matching images
    data_loader = utils.get_dataloader(query_nonmatch_dir, transform=None, batch_size=1, shuffle=False)
    attack_names = []
    with open(os.path.join(query_nonmatch_dir, 'query_40k_augmentations.txt'), 'r') as f:
        for line in f:
            attack_names.append(line)
    for ii, img in enumerate(tqdm.tqdm(data_loader)):
        attack_name = attack_names[ii]
        pil_img = img[0]
        attacked_img = transform(pil_img).unsqueeze(0).to(device)
        ft = model(attacked_img)
        retrieved_D, retrieved_I = index.search(ft.detach().cpu().numpy(), k=kneighbors)
        retrieved_D, retrieved_I = retrieved_D[0], retrieved_I[0]
        logs.append({
            'image': ct_match + ii, 
            'image_index': -1, 
            'attack': attack_name,
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
    os.makedirs(params.output_dir)
    marking_dir = os.path.join(params.output_dir, 'marking')
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    os.makedirs(marking_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    print(f'>>> Starting. Images will be saved in {imgs_dir} and evaluation logs in {params.output_dir}')

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
    if 'IVF' in params.target:  # optionally get the centroids
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
    data_loader = utils.get_dataloader(params.data_dir, transform, params.batch_size, shuffle=False)

    print('>>> Marking images and saving them into %s...'%imgs_dir)
    for it, imgs in enumerate(tqdm.tqdm(data_loader)):

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
            imgs = activate_images(imgs, model, index, ivf_centroids, attenuation, loss_f, loss_i, params)

        # Save images
        for ii, img in enumerate(imgs):
            img = torch.clamp(utils_img.UNNORMALIZE_IMAGENET(img), 0, 1) 
            img = torch.round(255 * img)/255 
            img = img.detach().cpu() 
            save_image(img, os.path.join(imgs_dir, f'{it*params.batch_size + ii:05d}.png'))
    
    if params.eval_retrieval:
        print(f'>>> Evaluating nearest neighbors search...')
        image_indices = range(n_index_ref, len(index.ntotal))
        df = eval_retrieval(imgs_dir, image_indices, transform_with_resize, model, index, params.kneighbors, params.use_attacks_2)
        df.to_csv(os.path.join(params.output_dir, 'df.csv'), index=False)
        df.fillna(0, inplace=True)
        df_mean = df.groupby(['attack', 'param0'], as_index=False).mean()
        print(f'\n{df_mean}')

    if params.eval_icd:
        print(f'>>> Evaluating copy detection on query set...')
        image_indices = range(n_index_ref, len(index.ntotal))
        icd_df = eval_icd(imgs_dir, image_indices, transform_with_resize, model, index, params.kneighbors, params.query_nonmatch_dir)
        icd_df_path = os.path.join(params.output_dir,'icd_df.csv')
        icd_df.to_csv(icd_df_path, index=False)
        print(f'\n{icd_df}')


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
