import os, tqdm, shutil, argparse, json
import pandas as pd

if __name__ == "__main__":

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, default="/disc/data/dir")
        parser.add_argument("--output_dir", type=str, default="/disc/data/dir")
        
        return parser

    params = get_parser().parse_args()
    print("Args:{}".format(json.dumps(vars(params))))

    # csv_path = os.path.join(params.data_dir, "groundtruth_matches.csv")
    csv_path = "/checkpoint/pfz/datasets/disc/groundtruth_matches.csv"
    df = pd.read_csv(csv_path, header=None, names=['Q', 'R'])
    rs = df['R'].values.tolist()
    rs.sort()

    reference_dir = os.path.join(params.data_dir, "references")

    dest_ref10k_dir = os.path.join(params.output_dir, "references_10k")
    dest_ref990k_dir = os.path.join(params.output_dir, "references_990k")
    print(f"Creating output directories: {dest_ref10k_dir}, {dest_ref990k_dir}")
    os.makedirs(dest_ref10k_dir, exist_ok=True)
    os.makedirs(dest_ref990k_dir, exist_ok=True)

    print("Getting filenames")
    filenames = [f'R{ii:06d}.jpg' for ii in range(1000000)]

    print("Getting reference images that are used in query")
    is_img_in_query = {}
    for filename in tqdm.tqdm(filenames):
        is_img_in_query[filename] = False
        if len(rs) == 0:
            continue
        if rs[0] in filename:
            is_img_in_query[filename] = True
            rs.pop(0)
    print(f"Number of reference images that are used in query: {sum(is_img_in_query.values())}")

    print(f"Copying the reference images")
    copy_mode = 'symlink'
    for filename in tqdm.tqdm(filenames):
        img_path = os.path.join(reference_dir, filename)
        dest_dir = dest_ref10k_dir if is_img_in_query[filename] else dest_ref990k_dir
        if copy_mode == 'symlink':
            os.symlink(img_path, os.path.join(dest_dir, filename))
        else:
            shutil.copy(img_path, os.path.join(dest_dir, filename))
