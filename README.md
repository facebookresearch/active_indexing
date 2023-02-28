# :pushpin: Active Image Indexing

PyTorch/FAISS implementation and pretrained models for the ICLR 2023 paper.
For details, see [**Active Image Indexing**](https://arxiv.org/abs/2210.10620).  


[[`Webpage`](https://pierrefdz.github.io/publications/activeindexing/)]
[[`arXiv`](https://arxiv.org/abs/2210.10620)]
[[`OpenReview`](https://openreview.net/forum?id=K9RHxPpjn2)]


## Introduction


### Context: Image copy detection & retrieval in large-scale databases

*Goal*: query image $\rightarrow$ find the most similar image in a large database

*Applications*: IP protection, de-duplication, moderation, etc.

### Problem
- Feature extractor that maps images to representation vectors is not completely robust to image transformations
- For large-scale databases, brute-force search note possible $\rightarrow$ we use approximate search with index structures (another source of error)
- this makes the copy detection task very challenging at scale

### Active Indexing
*Idea*: change images before release to make them more *indexing friendly*


<div align="center">
  <img width="100%" style="border-radius: 20px" alt="Illustration" src=".github/illustration.png">
</div>




## Activation

The main code for understanding the activation process is in [`engine.py`](https://github.com/facebookresearch/active_indexing/engine.py), in the `activate_images` function.

The 3 main inputs are:
- the images to be activated (batch of images 3xHxW)
- the index for which the images need to be activated
- the model used to extract features

The algorithm is as follows:
```{r, tidy=FALSE, eval=FALSE }
1. Initialize: 
    distortion δ: small perturbation added to the images to move their features. To be optimized.
    targets: where the features of the activated images should be pushed closer to.
    heatmaps: activation heatmaps that tell where to add the distortion (textured areas). 
2. Optimize
    for i in range(iterations):
        a. Add perceptual constraints to δ                    δ -> δ'
        b. Add δ' to original images                 img_o + δ' -> img
        c. Extract features from images              model(img) -> ft
        d. Compute loss between ft and target     L(ft, target) -> L
        e. Compute gradients of L wrt δ'                  ∇L(δ) -> ∇L
        f. Update δ with ∇L                         δ - lr * ∇L -> δ
    return img_o + δ'
```



## Usage

### Requirements

First, clone the repository locally and move inside the folder:
```cmd
git clone https://github.com/facebookresearch/active_indexing.git
cd active_indexing
```
To install the main dependencies, we recommand using conda.
[PyTorch](https://pytorch.org/) and [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) can be installed with:
```cmd
conda install -c pytorch torchvision pytorch==1.11.0 cudatoolkit=11.3
conda install -c conda-forge faiss-gpu==1.7.2
```
Then, install the remaining dependencies with:
```cmd
pip install -r requirements.txt
```
This codebase has been developed with python version 3.8, PyTorch version 1.11.0, CUDA 11.3 and FAISS 1.7.2.


### Data preparation

Experiments are done on DISC21.
It is available for download at https://ai.facebook.com/datasets/disc21-dataset/.

The dataset is composed of:
- 1M training images
- 1M reference images
- 50k query images, 10k of which came from the reference set.

We assume the dataset has been organized as follows:
```
DISC21
├── train
│   ├── T000000.jpg
│   ├── ...
│   └── T999999.jpg
├── references
│   ├── R000000.jpg
│   ├── ...
│   └── R999999.jpg
└── dev_queries_groundtruth.csv
```

We then provide a script to extract the 10k reference images that are used as queries in the dev set thanks to the ground-truth file:
```
python data/prepare_disc.py --data_path path/to/DISC21 --output_dir path/to/DISC21
```
This should create new folders in the `output_dir` (note that only symlinks are created, the images are not duplicated):
- `references_10k` containing the 10k reference images used as queries in the dev set
- `references_990k` folder containing the remaining 990k reference images
- `queries_40k` folder containing 40k additional query images that are not in the reference set (contrary to the paper, we take images from DISC training set instead of the original images of the query dev set before augmentation - for legal convenience).


### Feature extractor models

We provide the links to some models used as feature extractors:
| Name          | Trunk           | Dimension | TorchVision       |
|---|---|---|---|
| sscd_disc_advanced  | ResNet-50   | 512     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.torchscript.pt) |
| sscd_disc_mixup     | ResNet-50   | 512     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt) |
| sscd_disc_large     | ResNeXt101  | 1024    | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt) |
| dino_r50            | ResNet-50   | 2048  | [link](https://dl.fbaipublicfiles.com/active_indexing/dino_r50.torchscript.pt) |
| dino_vits           | ViT-s       | 384  | [link](https://dl.fbaipublicfiles.com/active_indexing/dino_vits16.torchscript.pt) |
| isc_dt1             | EffNetv2    | 256  | [link](https://dl.fbaipublicfiles.com/active_indexing/isc1.torchscript.pt) |

There are standalone TorchScript models that can be used in any pytorch project without any code corresponding to the networks. 
(We are not the authors of these models, we just provide them for convenience).

For example, to use the `sscd_disc_advanced` model:
```cmd
mkdir -p models
wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.torchscript.pt -O models/sscd_disc_advanced.torchscript.pt
```

Other links:
- SSCD: https://github.com/facebookresearch/sscd-copy-detection/
- DINO: https://github.com/facebookresearch/dino
- ISC-dt1: https://github.com/lyakaap/ISC21-Descriptor-Track-1st


### Feature extraction

We provide a simple script to extract features from a given model and a given image folder.
The features are extracted from the last layer of the model.
```
python extract_fts.py --model_name torchscript --model_path path/to/model --data_dir path/to/folder --output_dir path/to/output
```
This will save in the `--output_dir` folder: 
- `fts.pt`: the features in a torch file, 
- `filenames.txt`: a file containing the list of filenames corresponding to the features.

By default, images are resized to $288 \times 288$ (it can be changed with the `--resize_size` argument). 

To make things faster, the rest of the code assumes that features of the `DISC21/training` and `DISC21/ref_990k` image folders are pre-computed and saved in new folders.


## Reproduce Paper Experiments

To reproduce the results of the paper for IVF4096,PQ8x8, use the following command:
```
python main.py --model_name torchscript --model_path path/to/model \
   --idx_factory IVF4096,PQ8x8 --idx_dir indexes \
   --fts_training_path path/to/train/fts.pth --fts_reference_path path/to/ref/fts.pth \
   --data_dir path/to/DISC21/references_10k --query_nonmatch_dir path/to/DISC21/queries_40k \
   --active True --output_dir output_active
```
Replace the last line by `--active False --output_dir output_passive --save_imgs False` to do the same experiment with passive images. 
This should create:
- `indexes/idx=IVF4096,PQ8x8_quant=L2.index`: the index created and trained with features of `fts_training_path`,
- `output_active/` or `output_passive/`: folder containing the results of the experiment,
- `output_active/imgs`: folder containing the activated images (only if `--save_imgs True`),
- `output_active/retr_df.csv`: a csv file containing the results of the retrieval experiment (see below for more details),
- `output_active/icd_df.csv`: a csv file containing the results of the image copy detection experiment (see below for more details).

Useful arguments:
| Argument | Default | Description |
|---|---|---|
| `output_dir` | output/ | Path to the output folder where images and logs will be saved. |
| `idx_dir` | indexes/ | Path to the folder containing the index files (some of them can be long to create/train, so saving them is useful). |
| `idx_factory` | IVF4096,PQ8x8 | Index string to use to build index. See [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/The-index-factory) for more details. |
| `kneighbors` | 100 | Number of neighbors to retrieve when evaluating. |
| `model_name` | torchscript | Type of model to use. You can alternatively use Torchvision or Timm models.|
| `model_path` | None | Path to the torch file containing the model. |
| `fts_training_path` | None | Path to the torch file containing the features of the training set. |
| `fts_reference_path` | None | Path to the torch file containing the features of the reference set. |
| `save_imgs` | True | Saves the images during the active indexing process. It is useful to visualize the images, but is slower and takes more disk space. |
| `active` | True | If True, uses active indexing. If False, uses passive indexing. |


#### Log files

**`retr_df.csv`**: retrieval results. For every augmented version of the image it stores:

|batch|image_index|attack|attack_param|retrieved_distances|retrieved_indices|
|-----|-----------|------|------------|-------------------|-----------------|
|batch number|image number in the reference set|attack used|attack parameter|distances of the retrieved images (in feature space)|indices of the retrieved images

rank|r@1|r@10|r@100|ap|
----|---|----|-----|--|
rank of the original image in the retrieved images | recall at 1| recall at 10| recall at 100 | average precision|

**`icd_df.csv`**: image copy detection results. For each augmented version of the image it stores:
|batch|image_index|attack|attack_param|retrieved_distances|retrieved_indices|
|-----|-----------|------|------------|-------------------|-----------------|
|batch number|image number in the reference set|attack used|attack parameter|distances of the retrieved images (in feature space)|indices of the retrieved images|

To compute the precision-recall curve, you can use the associated code in the notebook `analysis.ipynb`.


#### Remark: 
- The `--model_path` argument should be the same as the one used to extract the features.
- The overlay onto screenshot transform (from Augly) that is used in the paper is the mobile version (Augly's default: web). To change it, you need to locate the file `augly/utils/base_paths.py` (run `pip show augly` to locate the Augly library). Then change the line "TEMPLATE_PATH = os.path.join(SCREENSHOT_TEMPLATES_DIR, "web.png")" to "TEMPLATE_PATH = os.path.join(SCREENSHOT_TEMPLATES_DIR, "mobile.png")".


## License

active_indexing is CC-BY-NC licensed, as found in the LICENSE file.

## Citation

If you find this repository useful, please consider giving a star :star: and please cite as:


```
@inproceedings{fernandez2022active,
  title={Active Image Indexing},
  author={Fernandez, Pierre and Douze, Matthijs and Jégou, Hervé and Furon, Teddy},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

