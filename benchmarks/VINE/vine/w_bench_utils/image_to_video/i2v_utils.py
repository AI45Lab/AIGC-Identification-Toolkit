import os, shutil, argparse
from tqdm import tqdm


def cluster_svd_images(args):
    source_folder = args.source_folder
    target_folder = args.target_folder

    small_folders = ["5", "7", "9", "11", "13", "15", "17", "19"]
    for folder in small_folders:
        os.makedirs(os.path.join(target_folder, folder), exist_ok=True)

    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith('.png'):
            number = str(int(filename.split('_')[-1].split('.')[0]))

            if number in small_folders:
                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(target_folder, number, filename.split('_')[0] + "_" + filename.split('_')[1] + '.png')
                shutil.copy(src_path, dst_path)
    print(f'ALL SVD Images Are Copied -> `{target_folder}`')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', type=str, default="./output/edited_wmed_wbench/SVD_raw")
    parser.add_argument('--target_folder', type=str, default="./output/edited_wmed_wbench/SVD_1K")
    args = parser.parse_args() 
    
    cluster_svd_images(args)
