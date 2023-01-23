import argparse, json, random, os, time, tqdm

import numpy as np

import faiss
import pandas as pd
import torch
import torch.nn as nn

from torch import optim

from torchvision import transforms
from torchvision.transforms import functional

import data_augs
import attenuations
import utils
import utils_img
import augment_queries


from torchvision.utils import save_image

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
    aa("--resize_at_eval", type=utils.bool_inst, default=True, help="Resize images at evaluation. (Default: True)")
    aa("--resize_size", type=int, default=288, help="Resize images to this size. (Default: 288)")

    group = parser.add_argument_group('Model parameters')
    aa("--model_path", type=str, default="/path/to/sscd_disc_advanced.torchscript.pt")
    aa("--model_name", type=str, default="custom")

    group = parser.add_argument_group('Index parameters')
    aa("--idx_path", type=str, default=None, help="Path to the index. (Default: None)")
    aa("--idx_dir", type=str, default="index_disc_sscd288", help="Directory where to save the index. (Default: index_disc_sscd288)")
    aa("--idx_factory", type=str, default="IVF4096,PQ8x8", help="String to create index from index factory. (Default: IVF4096,PQ8x8)")
    aa("--quant", type=str, default="L2", help="Quantizer type if IVF (L2, IP, etc.)")
    aa("--nprobe", type=int, default=1, help="Number of probes per query")
    aa("--kneighbors", type=int, default=100, help="Number of nearest neighbors to return")

    group = parser.add_argument_group('Data augmentation parameters')
    aa("--use_da", type=utils.bool_inst, default=False, help="Use data augmentation")
    aa("--degrees", type=int, default=90, help="Rotation range for the rotation augmentation. (Default: 90)")
    aa("--crop_scale", type=float, nargs='+', default=(0.2, 1.0), help="Crop scale range for the crop augmentation. (Default: (0.2, 1.0))")
    aa("--crop_ratio", type=float, nargs='+', default=(3/4, 4/3), help="Crop ratio for the crop augmentation. (Default: (3/4, 4/3))")
    aa("--blur_size", type=float, default=7, help="Blur scale range for the blur augmentation. (Default: 17)")
    aa("--color_jitter", type=float, nargs='+', default=(0.5, 0.5, 0.5, 0.2), help="Color jitter range for the color augmentation. (Default: (1.0, 1.0, 1.0, 0.3))")
    aa("--p_blur", type=float, default=1.0, help="Probability of the blur augmentation. (Default: 0.5)")
    aa("--p_aff", type=float, default=1.0, help="Probability of the rotation augmentation. (Default: 0.5)")
    aa("--p_crop", type=float, default=0.0, help="Probability of the crop augmentation. (Default: 0.5)")
    aa("--p_color_jitter", type=float, default=1.0, help="Probability of the color augmentation. (Default: 0.5)")
    aa("--p_diff_jpeg", type=float, default=1.0, help="Probability of the diffjpeg augmentation. (Default: 0.5)")
    aa("--low_jpeg", type=int, default=40, help="Lower bound in diffjpeg augmentation. (Default: 40)")
    aa("--n_aug_imgs", type=int, default=7, help="Number of augmented images per image. If 0 only one augmentation. (Default: 0)")

    group = parser.add_argument_group('Optimization parameters')
    aa("--iterations", type=int, default=10, help="Number of iterations for image optimization. (Default: 10)")
    aa("--optimizer", type=str, default="Adam,lr=1e-0", help="Optimizer to use. (Default: Adam)")
    aa("--scheduler", type=str, default=None, help="Scheduler to use. (Default: None)")
    aa("--target", type=str, default="pq_recons", help="Target to use. (Default: pq_recons)")
    aa("--loss_w", type=str, default="cossim", help="Loss w to use. Choose among mse, eot, mot, smot (Default: cossim)")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss. (Default: 1.0)")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss. (Default: 1.0)")

    group = parser.add_argument_group('Distortion & Attenuation parameters')
    aa("--use_attenuation", type=utils.bool_inst, default=True, help="Use heatmap attenuation")
    aa("--scaling", type=float, default=3.0, help="Scaling factor for the heatmap attenuation")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Scale the channels of the heatmap attenuation")
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh for the heatmap attenuation")
    aa("--linf", type=int, default=0, help="Use linf for the heatmap attenuation")
    aa("--distortion_init", type=float, default=0, help="Initial distortion value. Will be multiplied by a random noise. (Default: 0)")

    group = parser.add_argument_group('Evaluation parameters')
    aa("--only_1k", type=utils.bool_inst, default=False, help="Only evaluate on 1k images. (Default: False)")
    aa("--use_attacks_2", type=utils.bool_inst, default=False, help="Use attacks_2 for augmentation evaluation. (Default: False)")
    aa("--eval_retrieval", type=utils.bool_inst, default=True, help="Evaluate retrieval. (Default: True)")
    aa("--eval_icd", type=utils.bool_inst, default=True, help="Evaluate icd. (Default: True)")

    group = parser.add_argument_group('Misc parameters')
    aa("--save_imgs", type=utils.bool_inst, default=True, help="Save images")
    aa("--post_process", type=utils.bool_inst, default=True, help="Post process images")
    aa("--log_freq", type=int, default=1, help="Log every n iterations. (Default: 1)")
    aa("--passive", type=utils.bool_inst, default=False, help="Passive mode")
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def main(params):

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # None param for clutils
    if params.scheduler is not None:
        if params.scheduler.lower() == 'none':
            params.scheduler = None
    if params.idx_path is not None:
        if params.idx_path.lower() == 'none':
            params.idx_path = None
    if params.query_match_dir is not None:
        if params.query_match_dir.lower() == 'none':
            params.query_match_dir = None

    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    if params.idx_dir is not None:
        os.makedirs(params.idx_dir, exist_ok=True)
    marking_dir = os.path.join(params.output_dir, 'marking')
    if not os.path.exists(marking_dir):
        os.makedirs(marking_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # Build the model
    print(f'>>> Building backbone from {params.model_path}...')
    model = utils.build_backbone(path=params.model_path, name=params.model_name)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # Build Index - https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    print(f'>>> Building Index')
    if params.idx_path is None:
        if params.idx_factory is not None:
            params.idx_path = os.path.join(params.idx_dir, f'idx={params.idx_factory}_quant={params.quant}.index')
        else:
            raise ValueError('idx_factory or idx_path are not specified')
    if os.path.exists(params.idx_path):
        print(f'>>> Loading Index from {params.idx_path}')
        index = faiss.read_index(params.idx_path)
    else:
        index = utils.build_index_factory(params.idx_factory, params.quant, params.fts_training_path, params.idx_path)
    index.nprobe = params.nprobe
    n_index_training = index.ntotal
    if 'IVF' in params.idx_factory:
        ivf = faiss.extract_index_ivf(index)
        ivf_centroids = ivf.quantizer.reconstruct_n(0, ivf.nlist)
    
    # Adding reference images to the index
    print(f'>>> Adding reference images to the index from {params.fts_reference_path}...')
    fts = torch.load(params.fts_reference_path)
    index.add(fts.detach().cpu().numpy())
    n_index_ref = index.ntotal
    if 'IVF' in params.idx_factory:
        ivf.make_direct_map()

    # Load images to mark and index
    print(f'>>> Loading images from {params.data_dir}...')
    NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    transform = transforms.Compose([
        transforms.ToTensor(), NORMALIZE_IMAGENET,
    ])
    data_loader = utils.get_dataloader(params.data_dir, transform, params.batch_size, shuffle=False)

    same_size = True
    for ii in range(len(data_loader.dataset)-1):
        if not os.path.getsize(data_loader.dataset.samples[ii]) == os.path.getsize(data_loader.dataset.samples[ii+1]):
            same_size = False
            print(f'Images in folder do not have same size. Resizing them at {params.resize_size} before feature extraction...')
            break
    same_size = same_size or (params.batch_size==1)
    assert (same_size or params.resize), f'Images of a batch must have same size. Set "resize" to True or "batch_size" to 1'

    # Build data augmentation to apply at marking time
    print('>>> Building data augmentation...')
    if params.use_da:
        data_aug = data_augs.KorniaAug(
            degrees=params.degrees,
            crop_scale=params.crop_scale,
            crop_ratio=params.crop_ratio,
            blur_size=params.blur_size,
            color_jitter=params.color_jitter,
            diff_jpeg = params.low_jpeg,
            p_crop=params.p_crop,p_blur=params.p_blur,p_color_jitter=params.p_color_jitter,p_aff=params.p_aff,p_diff_jpeg=params.p_diff_jpeg,
        ).to(device)
    else:
        data_aug = nn.Identity().to(device)

    # loss for watermark and image
    loss_w = params.loss_w.lower()
    is_mse = loss_w.startswith('mse')
    is_cossim = loss_w.startswith('cossim')
    assert is_mse or is_cossim, f'Invalid loss_w: {loss_w}'
    CosSim = nn.CosineSimilarity(dim=-1)
    pDist = nn.PairwiseDistance(p=2)
    mse = nn.MSELoss()
    def LossW(ft, target):
        dists = -CosSim(ft, target) if is_cossim else pDist(ft, target)**2
        return torch.mean(dists)
    def LossI(imgs, imgs_ori):
        loss_i = 0
        bb = len(imgs)
        for ii in range(bb):
            loss_i += mse(imgs[ii], imgs_ori[ii])
        return loss_i/bb
    print('>>> Loss f is %s...' % 'mse' if is_mse else 'cosine similarity')

    print('>>> Marking images and saving them into %s...'%imgs_dir)
    for it, imgs in enumerate(tqdm.tqdm(data_loader)):

        if params.debug:
            if it >1: 
                break
        if params.only_1k and it*params.batch_size > 1000:
            break

        imgs = [el.to(device) for el in imgs]


        # Index
        n_total = index.ntotal
        if params.resize:
            batch_imgs = torch.stack([functional.resize(el, (params.resize_size, params.resize_size)) for el in imgs])
        else:
            batch_imgs = torch.stack(imgs)
        fts = model(batch_imgs)
        index.add(fts.detach().cpu().numpy())
        if 'IVF' in params.idx_factory:
            ivf.make_direct_map()

        if params.passive:
            clip_imgs = [el.detach().cpu() for el in imgs]

        else:
            targets = []
            # for jj in range(len(imgs)):
            if params.target == 'pq_recons':
                targets.append(torch.tensor(index.reconstruct_n(n_total, fts.shape[0])).to(device))
            elif params.target == 'ori_ft':
                targets.append(fts.clone())
            elif params.target == 'ivf_cluster':
                ivf_D, ivf_I = index.quantizer.search(fts.detach().cpu().numpy(), k=1)
                centroids = ivf_centroids.take(ivf_I.flatten(), axis=0)
                targets.append(torch.tensor(centroids, device=device))
            elif params.target == 'ivf_cluster_half':
                ivf_D, ivf_I = index.quantizer.search(fts.detach().cpu().numpy(), k=1)
                centroids = ivf_centroids.take(ivf_I.flatten(), axis=0)
                middle = torch.tensor(centroids, device=device) + fts.clone()
                targets.append(middle/2)
            else:
                raise NotImplementedError(f'Invalid target: {params.target}')
            targets = torch.cat(targets, dim=0)
            # print((fts-targets)[0])
            if params.use_da:
                targets = targets.repeat(params.n_aug_imgs+1, *[1 for _ in targets.shape[1:]])

            # init distortion and channel scaler
            distortion_scaler = torch.tensor([0.072*(1/0.299), 0.072*(1/0.587), 0.072*(1/0.114)]).to(device)
            distortions = [params.distortion_init * torch.randn_like(el).to(device) for el in imgs] # b (1 c h w)
            for distortion in distortions:
                distortion.requires_grad = True
            distortion_scaler = distortion_scaler[:,None,None].to(device) 
            linf = params.linf / 255 / 0.229    # linf but in imnet range - approx (-3,3)

            # attenuation and heatmaps
            if params.use_attenuation:
                attenuation = attenuations.JND(preprocess = UNNORMALIZE_IMAGENET).to(device)
                attenuation.requires_grad = False
                heatmaps = [params.scaling * attenuation.heatmaps(el) for el in imgs]
            else:
                heatmaps = [params.scaling for _ in imgs]

            # optimizer and scheduler
            optim_params = utils.parse_params(params.optimizer)
            optimizer = utils.build_optimizer(model_params=distortions, **optim_params)
            scheduler = utils.build_scheduler(optimizer=optimizer, **utils.parse_params(params.scheduler)) if params.scheduler is not None else None
            
            log_stats = []
            iter_time = time.time()
            for gd_it in range(params.iterations):
                gd_it_time = time.time()
                if scheduler is not None:
                    scheduler.step(gd_it)

                # perceptual constraints
                if params.linf > 0 and not params.use_tanh:
                    for el in distortions:
                        el.data = torch.clamp_( el.data, min=-linf, max=linf) 
                percep_distortions = [torch.tanh(el) for el in distortions] if params.use_tanh else distortions
                percep_distortions = [el * distortion_scaler for el in percep_distortions] if params.scale_channels else percep_distortions
                clip_imgs = [el1 + el2 * el3 for el1, el2, el3 in zip(imgs, heatmaps, percep_distortions)]
                
                if params.resize:
                    batch_imgs = [functional.resize(el, (params.resize_size, params.resize_size)) for el in clip_imgs]
                else:
                    batch_imgs = clip_imgs
                batch_imgs = torch.stack(batch_imgs)

                # augment images
                if params.use_da:
                    if params.n_aug_imgs > 0:
                        augm_imgs = batch_imgs.repeat(params.n_aug_imgs, *[1 for _ in batch_imgs.shape[1:]])
                        augm_imgs = NORMALIZE_IMAGENET(data_aug(UNNORMALIZE_IMAGENET(augm_imgs)))
                        augm_imgs = torch.cat([batch_imgs, augm_imgs], dim=0)
                    else:
                        augm_imgs = NORMALIZE_IMAGENET(data_aug(UNNORMALIZE_IMAGENET(batch_imgs)))
                else:
                    augm_imgs = batch_imgs

                # get features
                fts = model(augm_imgs) # b d

                # compute losses
                loss_w = LossW(fts, targets)
                loss_i = LossI(clip_imgs, imgs)
                loss = params.lambda_w * loss_w + params.lambda_i * loss_i

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    if True:
                        psnrs = torch.tensor([utils.psnr(el1, el2) for el1, el2 in zip(clip_imgs, imgs)])
                        linfs = torch.tensor([utils.linf(el1, el2) for el1, el2 in zip(clip_imgs, imgs)])
                        log_stats.append({
                            'batch_it': it,
                            'gd_it': gd_it,
                            'loss': loss.item(),
                            'loss_w': loss_w.item(),
                            'loss_i': loss_i.item(),
                            'psnr': torch.nanmean(psnrs).item(),
                            'linf': torch.nanmean(linfs).item(),
                            'lr': optimizer.param_groups[0]['lr'],
                            'gd_it_time': time.time() - gd_it_time,
                            'iter_time': time.time() - iter_time,
                            'max_mem': torch.cuda.max_memory_allocated() / (1024*1024),
                            'kw': 'optim',
                        })
                        if gd_it % params.log_freq == 0:
                            print(json.dumps(log_stats[-1]))
            
            # perceptual constraints and postprocessing
            if params.linf > 0 and not params.use_tanh:
                for el in distortions:
                    el.data = torch.clamp_( el.data, min=-linf, max=linf) 
            percep_distortions = [torch.tanh(el) for el in distortions] if params.use_tanh else distortions
            percep_distortions = [el * distortion_scaler for el in percep_distortions] if params.scale_channels else percep_distortions
            clip_imgs = [el1 + el2 * el3 for el1, el2, el3 in zip(imgs, heatmaps, percep_distortions)]

        if params.post_process:
            clip_imgs = [torch.clamp(UNNORMALIZE_IMAGENET(el), 0, 1) for el in clip_imgs]
            clip_imgs = [torch.round(255 * el)/255 for el in clip_imgs]
        else: 
            clip_imgs = [UNNORMALIZE_IMAGENET(el) for el in clip_imgs]
        clip_imgs = [el.detach().cpu() for el in clip_imgs]
        
        for ii, img_out in enumerate(clip_imgs):
            save_image(img_out, os.path.join(imgs_dir, f'{it*params.batch_size + ii:05d}.png'))
    
    if params.eval_retrieval:
        print(f'>>> Evaluating nearest neighbors search...')
        logs = []
        data_loader = utils.get_dataloader(imgs_dir, transform=None, batch_size=1, shuffle=False)
        attacks = utils_img.attacks_2 if params.use_attacks_2 else utils_img.attacks 
        for ii, img in enumerate(tqdm.tqdm(data_loader)):
            image_index = n_index_ref + ii
            pil_img = img[0]
            # pil_img = transforms.ToPILImage()(img)
            attacked_imgs = utils_img.generate_attacks(pil_img, attacks)
            if ii==0:
                for jj in range(len(utils_img.attacks)):
                    attacked_imgs[jj].save(os.path.join(imgs_dir,"%i_%s.jpg"%(ii, str(utils_img.attacks[jj])) ))
            for jj, attacked_img in enumerate(attacked_imgs):
                attacked_img = transform(attacked_img).unsqueeze(0).to(device)
                if params.resize_at_eval:
                    attacked_img = functional.resize(attacked_img, (params.resize_size, params.resize_size))
                ft = model(attacked_img)
                ft = ft.detach().cpu().numpy()
                retrieved_D, retrieved_I = index.search(ft, k=params.kneighbors)
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
                    'r@1': 1 if rank==0 else 0,
                    'r@10': 1 if rank<10 else 0,
                    'r@100': 1 if rank<100 else 0,
                    'ap': 1/(rank+1),
                    "kw": "evaluation",
                })  
        df = pd.DataFrame(logs).drop(columns='kw')
        df_path = os.path.join(params.output_dir,'df.csv')
        df.to_csv(df_path, index=False)
        df.fillna(0, inplace=True)
        df_mean = df.groupby(['attack', 'param0'], as_index=False).mean()
        print(f'\n{df_mean}')

    if params.eval_icd:
        print(f'>>> Evaluating copy detection on query set...')
        # stats on matching images
        print(f'On matches...')
        query_match_dir = imgs_dir if params.query_match_dir is None else params.query_match_dir
        data_loader = utils.get_dataloader(query_match_dir, transform=None, batch_size=1, shuffle=False)
        rng = np.random.RandomState(params.seed)
        logs = []
        ct_match = 0
        for ii, img in enumerate(tqdm.tqdm(data_loader)):
            image_index = n_index_ref + ii
            pil_img = img[0]
            attacked_img, aug_params = augment_queries.augment_img(pil_img, rng, return_params=True)
            attack_name = "[" + ", ".join([str(ftr) for ftr in aug_params])
            attacked_img = transform(attacked_img).unsqueeze(0).to(device)
            if params.resize_at_eval:
                attacked_img = functional.resize(attacked_img, (params.resize_size, params.resize_size))
            ft = model(attacked_img)
            ft = ft.detach().cpu().numpy()
            retrieved_D, retrieved_I = index.search(ft, k=params.kneighbors)
            retrieved_D, retrieved_I = retrieved_D[0], retrieved_I[0]
            rank = [kk for kk in range(len(retrieved_I)) if retrieved_I[kk]==image_index]
            rank = rank[0] if rank else len(retrieved_I)
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
        print(f'On non-matches...')
        data_loader = utils.get_dataloader(params.query_nonmatch_dir, transform=None, batch_size=1, shuffle=False)
        attack_names = []
        with open(os.path.join(params.query_nonmatch_dir, 'query_40k_augmentations.txt'), 'r') as f:
            for line in f:
                attack_names.append(line)
        for ii, img in enumerate(tqdm.tqdm(data_loader)):
            attacked_img = transform(img[0]).unsqueeze(0).to(device)
            attack_name = attack_names[ii]
            if params.resize_at_eval:
                attacked_img = functional.resize(attacked_img, (params.resize_size, params.resize_size))
            ft = model(attacked_img)
            retrieved_D, retrieved_I = index.search(ft.detach().cpu().numpy(), k=params.kneighbors)
            retrieved_D, retrieved_I = retrieved_D[0], retrieved_I[0]
            rank = [kk for kk in range(len(retrieved_I)) if retrieved_I[kk]==image_index]
            rank = rank[0] if rank else len(retrieved_I)
            logs.append({
                'image': ct_match + ii, 
                'image_index': -1, 
                'attack': attack_name,
                'scores': retrieved_D,
                'retrieved_ids': retrieved_I,
                "kw": "icd_evaluation",
            })
        icd_df = pd.DataFrame(logs).drop(columns='kw')
        icd_df_path = os.path.join(params.output_dir,'icd_df.csv')
        icd_df.to_csv(icd_df_path, index=False)
        print(f'\n{icd_df}')


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
