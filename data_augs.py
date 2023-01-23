
import torch
import torch.nn as nn
from diff_jpeg import DiffJPEG
import kornia.augmentation as K

from kornia.augmentation import AugmentationBase2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self, p, low=40) -> None:
        super().__init__(p=p)
        self.diff_jpegs = [DiffJPEG(quality=qf).to(device) for qf in range(low,100,10)]
        # self.diff_jpegs = [DiffJPEG(quality=qf).to(device) for qf in [50,80]]

    def generate_parameters(self, input_shape: torch.Size):
        qf = torch.randint(high=len(self.diff_jpegs), size=input_shape[0:1])
        return dict(qf=qf)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        qf = params['qf']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.diff_jpegs[qf[ii]](input[ii:ii+1])
        return output

class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size, p=1) -> None:
        super().__init__(p=p)
        self.gaussian_blurs = [K.RandomGaussianBlur(kernel_size=(kk,kk), sigma= (kk*0.15 + 0.35, kk*0.15 + 0.35)) for kk in range(1,int(blur_size),2)]

    def generate_parameters(self, input_shape: torch.Size):
        blur_strength = torch.randint(high=len(self.gaussian_blurs), size=input_shape[0:1])
        return dict(blur_strength=blur_strength)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        blur_strength = params['blur_strength']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.gaussian_blurs[blur_strength[ii]](input[ii:ii+1])
        return output

class KorniaAug(nn.Module):
    def __init__(self, degrees=30, crop_scale=(0.2, 1.0), crop_ratio=(3/4, 4/3), blur_size=17, color_jitter=(1.0, 1.0, 1.0, 0.3), diff_jpeg=40,
                p_crop=0.5, p_aff=0.5, p_blur=0.5, p_color_jitter=0.5, p_diff_jpeg=0.5, 
                cropping_mode='slice',
            ):
        super(KorniaAug, self).__init__()
        self.jitter = K.ColorJitter(*color_jitter, p=p_color_jitter).to(device)
        self.aff = K.RandomAffine(degrees=degrees, p=p_aff).to(device)
        self.crop = K.RandomResizedCrop(size=(224,224),scale=crop_scale,ratio=crop_ratio, p=p_crop, cropping_mode=cropping_mode).to(device)
        self.hflip = K.RandomHorizontalFlip().to(device)
        self.blur = RandomBlur(blur_size, p_blur).to(device)
        self.diff_jpeg = RandomDiffJPEG(p=p_diff_jpeg, low=diff_jpeg).to(device)
    
    def forward(self, input):
        input = self.diff_jpeg(input)
        input = self.aff(input)
        input = self.crop(input)
        input = self.blur(input)
        input = self.jitter(input)
        input = self.hflip(input)
        return input
