import torch
import os
import argparse
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from Wevade import removal, forgery
import torchvision
import yaml
import time
from easydict import EasyDict
from videoseal.utils.cfg import setup_model_from_model_card
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.video")
with open('configs/eval.yaml', 'r') as f:
    cfg = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

class video_dataset(torch.utils.data.Dataset):
    def __init__(self, video_path):
        self.video_files = []
        for video_file in os.listdir(video_path):
            if video_file.endswith('.mp4'):
                self.video_files.append(os.path.join(video_path, video_file))
        self.video_files.sort()
    def load_video(self, video_file):
        video = torchvision.io.read_video(video_file, output_format="TCHW")
        video = video[0].float() / 255.0
        return video
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video = self.load_video(video_file)
        video_name = video_file.split('/')[-1]
        return video, video_name


def main(args):

    model = setup_model_from_model_card('./videoseal/cards/videoseal.yaml')
    model.eval()
    model.blender.scaling_w = model.blender.scaling_w
    model.chunk_size = model.chunk_size
    model.step_size = model.step_size
    if args.cuda:
        model.to(args.device)
    # Load dataset.
    if args.attack_type == 'removal':
        dataset_path = os.path.join(args.dataset[args.video_model], args.genre)
        folder = f'videoseal_wbox/{args.video_model}/{args.genre}'
    else:
        dataset_path = args.dataset['kinetics']
        folder = f'videoseal_wbox/{args.video_model}'
    dataset = video_dataset(dataset_path)
    data = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    criterion = nn.MSELoss().to(device=args.device)
    os.makedirs(folder, exist_ok=True)
    filename = f'wbox_PGD_{args.attack_type}_rb{args.rb}.pkl'
    results_path = os.path.join(folder, filename)
    results_pk = pd.read_pickle(results_path) if os.path.exists(results_path) else {}
    attack = removal if args.attack_type == 'removal' else forgery
    gt_watermark = torch.zeros(1, args.secret_size).to(args.device)
    gt_watermark[:, 1::2] = 1
    for video, video_name in tqdm(data,desc='Video'):
        video_name = video_name[0]
        if video_name not in results_pk:
            results_pk[video_name] = {}
        elif len(results_pk[video_name]) != video.shape[1]:
            results_pk[video_name] = {}
        else:
            print(f'{video_name} already processed')
            continue
        video = video.squeeze()
        if args.attack_type == 'removal':
            video = video.to(args.device)
            video = model.embed(imgs=video, msgs=gt_watermark, is_video=True)["imgs_w"] # NOTE：watermarked video
            video = video.detach().cpu()
        for frame_id, frame in enumerate(video):
            frame = frame.clone().unsqueeze(0).to(args.device)
            bit_acc, bound, ssim, psnr, success, decoded_logits = attack(frame, gt_watermark, model, criterion, args)
            # NOTE: the decoded_logits in approximately in range [-1,1], with > 0 as 1 if rounded
            results_pk[video_name][frame_id] = {
                'bit_acc': bit_acc,
                'bound': bound,
                'ssim': ssim,
                'psnr': psnr,
                'decoded_logits': decoded_logits,
            }
        pd.to_pickle(results_pk, results_path)

def parse_args():
    parser = argparse.ArgumentParser(description='StegaStamp wbox attack')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--attack_type', type=str, default='removal', choices=['removal', 'forgery'])
    parser.add_argument('--secret_size', type=int, default=96)
    parser.add_argument('--tau', default=float(67/96), type=float, help='Detection threshold of the detector.')
    parser.add_argument('--iteration', default=200, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--epsilon', default=4/96, type=float, help='Epsilon used in WEvdae-W.')
    parser.add_argument('--alpha', default=0, type=float, help='Learning rate used in WEvade-W.')
    parser.add_argument('--rb', default=0.1, type=float, help='Upper bound of perturbation.')
    parser.add_argument('--WEvade-type', default='WEvade-W-I', type=str, help='Using WEvade-W-I/II.')
    parser.add_argument('--type', default='PGD', type=str, help='Type of attack.')
    parser.add_argument('--genre', type=str, default='cartoon')
    parser.add_argument('--video_model', type=str, default='sora')
    return vars(parser.parse_args())      

  
if __name__ == '__main__':
    args = EasyDict({**cfg, **parse_args()})
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(args)
    main(args)