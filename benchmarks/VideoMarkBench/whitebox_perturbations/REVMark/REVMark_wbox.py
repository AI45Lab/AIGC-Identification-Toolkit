import torch
import os
import yaml
import torch.nn as nn
import argparse
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
from glob import glob
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from easydict import EasyDict
from Wevade import removal, forgery
from REVMark import Encoder, Decoder, framenorm
import cv2
import warnings
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed")

# Your PyTorch code continues here
with open('cfg/eval.yaml', 'r') as f:
    cfg = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))


class video_dataset(torch.utils.data.Dataset):
    def __init__(self, video_path):
        self.video_files = []
        for video_file in os.listdir(video_path):
            if video_file.endswith('.mp4'):
                self.video_files.append(os.path.join(video_path, video_file))
        self.video_files.sort()
    def resize_frame(self, frame):
        return cv2.resize(frame, (128, 128))
    def reshape_tensor(self, tensor):
        c, y, h, w = tensor.shape
        x = (y + 7) // 8  # Compute x by ensuring divisibility
        pad_size = x * 8 - y  # Compute how many extra slices are needed
        if pad_size > 0:
            tensor = torch.cat([tensor, tensor[:, :pad_size]], dim=1)  # Pad by repeating from the beginning
        tensor = tensor.view(c, x, 8, h, w)  # Reshape
        tensor = tensor.permute(1, 0, 2, 3, 4)  # Reorder to [x, 3, 8, 128, 128]
        return tensor        
    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        video = []
        # for i in range(8): # NOTE: in REVMark, only embed 8 frames
        #     ret, frame = cap.read()
        #     video.append(torch.from_numpy(frame[:128,:128].transpose(2,0,1).astype('float32') / 255))
        # return torch.stack(video,dim=1)#.unsqueeze(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video.append(torch.from_numpy(frame[:128,:128].transpose(2,0,1).astype('float32') / 255))
        return torch.stack(video,dim=1)#.unsqueeze(0)
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video = self.load_video(video_file)
        video = self.reshape_tensor(video)
        video_name = video_file.split('/')[-1]
        return video, video_name


def main(args):
    results_csv = pd.DataFrame(columns=['video_name', 'frame_id', 'Bit_acc', 'Perturbation', 'ssim', 'psnr', 'Evasion_rate', 'decoded_logits'])
    
    encoder = Encoder(96, [8,128,128]).to(args.device).eval()
    decoder = Decoder(96, [8,128,128]).to(args.device).eval()
    encoder.load_state_dict(torch.load('checkpoints/Encoder.pth'))
    decoder.load_state_dict(torch.load('checkpoints/Decoder.pth'))
    encoder.tasblock.enable = False
    decoder.tasblock.enable = False
    # Load dataset.
    if args.attack_type == 'removal':
        dataset_path = os.path.join(args.dataset[args.video_model], args.genre)
        folder = f'REVMark_wbox/{args.video_model}/{args.genre}'
    else:
        dataset_path = args.dataset['kinetics']
        folder = f'REVMark_wbox/kinetics'
    
    dataset = video_dataset(dataset_path)
    data = DataLoader(dataset, batch_size=1, shuffle=False)
    # WEvade.
    criterion = nn.MSELoss().to(args.device)
    os.makedirs(folder, exist_ok=True)
    filename = f'wbox_PGD_{args.attack_type}_rb{args.rb}.csv'
    attack = removal if args.attack_type == 'removal' else forgery
    evate = lambda x: (x <= args.tau) if args.attack_type == 'removal' else (x > args.tau)
    gt_watermark = torch.zeros(1, args.secret_size).to(args.device)
    gt_watermark[:, 1::2] = 1

    for video, video_name in tqdm(data,desc='Video'):
        video = video.squeeze(0)
        if args.attack_type == 'removal':
            video = video.to(args.device)
            cover = video * 2 - 1  # Normalize between [-1, 1]
            m = gt_watermark.repeat(cover.shape[0], 1)
            r = encoder(cover, m)
            video = (cover + 6.2 * framenorm(r)).clamp(-1, 1) # shape: [x, 3, 8, 128, 128]
            video = video.detach().cpu()
            video = (video + 1) / 2
        for frame_id, frames in tqdm(enumerate(video),desc='Frame (total %d)' % len(video)):
            frames = frames[None, ...].clone().to(args.device) # shape: [1, 3, 8, 128, 128]
            bit_acc, bound, ssim, psnr, success, decoded_logits = attack(frames, gt_watermark, decoder, criterion, args)
            evasion = evate(bit_acc)
            # NOTE: the decoded_logits in range [0,1]??
            results_csv.loc[len(results_csv)] = [video_name[0], frame_id, bit_acc, bound, ssim, psnr, evasion, decoded_logits]
            del frames, bit_acc, bound, ssim, success, evasion
        results_csv.to_csv(os.path.join(folder, filename), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='StegaStamp wbox attack')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--attack_type', type=str, default='forgery', choices=['removal', 'forgery'])
    parser.add_argument('--secret_size', type=int, default=96)
    parser.add_argument('--tau', default=float(67/96), type=float, help='Detection threshold of the detector.')
    parser.add_argument('--iteration', default=1000, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--epsilon', default=6/96, type=float, help='Epsilon used in WEvdae-W.')
    parser.add_argument('--alpha', default=0, type=float, help='Learning rate used in WEvade-W.')
    parser.add_argument('--rb', default=0.005, type=float, help='Upper bound of perturbation.')
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