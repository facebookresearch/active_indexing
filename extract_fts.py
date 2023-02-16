import argparse, os, tqdm, json

import torch
from torch import device
from torchvision import transforms

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default='output')
        parser.add_argument("--data_dir", type=str, default="/img/data/dir")
        parser.add_argument("--model_name", type=str, default="custom")
        parser.add_argument("--model_path", type=str, default="/path/to/model.torchscript.pt")
        parser.add_argument("--resize_size", type=int, default=288, help="Resize images to this size. (Default: 288)")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
        
        return parser

    params = get_parser().parse_args()
    print("__log__:{}".format(json.dumps(vars(params))))
    
    print('>>> Creating output directory...')
    os.makedirs(params.output_dir, exist_ok=True)
    
    print('>>> Building backbone...')
    model = utils.build_backbone(path=params.model_path, name=params.model_name)
    model.eval()
    model.to(device)

    print('>>> Creating dataloader...')
    NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE_IMAGENET,
        transforms.Resize((params.resize_size, params.resize_size)),
    ])
    img_loader = utils.get_dataloader(params.data_dir, default_transform, batch_size=params.batch_size, collate_fn=None)

    print('>>> Extracting features...')
    features = []
    with open(os.path.join(params.output_dir, "filenames.txt"), 'w') as f:
        with torch.no_grad():
            for ii, imgs in enumerate(tqdm.tqdm(img_loader)):
                imgs = imgs.to(device)
                fts = model(imgs)
                features.append(fts.cpu())
                for jj in range(fts.shape[0]):
                    sample_fname = img_loader.dataset.samples[ii*params.batch_size + jj]
                    f.write(sample_fname + "\n")

    print('>>> Saving features...')
    features = torch.concat(features, dim=0)
    torch.save(features, os.path.join(params.output_dir, 'fts.pth'))
        
