import argparse
import json
import os
import random
import shutil

import pandas as pd
import tqdm

if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        # parser.add_argument("--data_dir", type=str, default="/disc/data/dir")
        # parser.add_argument("--output_dir", type=str, default="/disc/data/dir")
        parser.add_argument("--data_dir", type=str, default="/datasets01/disc21/091422/")
        parser.add_argument("--output_dir", type=str, default="/checkpoint/pfz/datasets/disc_prepared")
        
        return parser

    params = get_parser().parse_args()
    copy_mode = 'symlink'
    print("Args:{}".format(json.dumps(vars(params))))

    dest_ref10k_dir = os.path.join(params.output_dir, "references_10k")
    dest_ref990k_dir = os.path.join(params.output_dir, "references_990k")
    dest_query_40k_dir = os.path.join(params.output_dir, "queries_40k")
    os.makedirs(dest_ref10k_dir, exist_ok=True)
    os.makedirs(dest_ref990k_dir, exist_ok=True)
    os.makedirs(dest_query_40k_dir, exist_ok=True)
    print(f"Creating output directories: {dest_ref10k_dir}, {dest_ref990k_dir}, {dest_query_40k_dir}")

    print(f"Copying the reference images")
    reference_dir = os.path.join(params.data_dir, "references")
    filenames = [f'R{ii:06d}.jpg' for ii in range(1000000)]
    csv_path = os.path.join(params.data_dir, "groundtruth_matches.csv")
    df = pd.read_csv(csv_path, header=None, names=['Q', 'R'])
    rs = df['R'].values.tolist()
    rs.sort()
    is_img_in_query = {}
    for filename in filenames:
        is_img_in_query[filename] = False
        if len(rs) == 0:
            continue
        if rs[0] in filename:
            is_img_in_query[filename] = True
            rs.pop(0)
    print(f"Number of reference images that are used in query: {sum(is_img_in_query.values())}")
    for filename in tqdm.tqdm(filenames):
        img_path = os.path.join(reference_dir, filename)
        dest_dir = dest_ref10k_dir if is_img_in_query[filename] else dest_ref990k_dir
        if copy_mode == 'symlink':
            os.symlink(img_path, os.path.join(dest_dir, filename))
        else:
            shutil.copy(img_path, os.path.join(dest_dir, filename))
    
    print(f"Copying the query images")
    train_dir = os.path.join(params.data_dir, "train")
    filenames = [f'T{ii:06d}.jpg' for ii in range(1000000)]
    random.seed(0)
    filenames = random.sample(filenames, 40000)
    for filename in tqdm.tqdm(filenames):
        img_path = os.path.join(train_dir, filename)
        if copy_mode == 'symlink':
            os.symlink(img_path, os.path.join(dest_query_40k_dir, filename))
        else:
            shutil.copy(img_path, os.path.join(dest_query_40k_dir, filename))

