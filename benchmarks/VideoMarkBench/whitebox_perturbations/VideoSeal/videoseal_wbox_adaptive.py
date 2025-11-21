import torch
import os
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from Wevade_adaptive import removal, forgery, detect
import torchvision
import yaml
from easydict import EasyDict
from videoseal.utils.cfg import setup_model_from_model_card

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
        # video = self.reshape_tensor(video)
        video_name = video_file.split('/')[-1]
        return video, video_name


def main(args):
    folder = f'videoseal_wbox_adaptive'
    os.makedirs(folder, exist_ok=True)
    filename = f'{args.video_model}_{args.genre}.pkl' if args.attack_type == 'removal' else f'{args.video_model}.pkl'
    results_path = os.path.join(folder, filename)
    # Load existing results or create new dict
    results_pk = pd.read_pickle(results_path) if os.path.exists(results_path) else {}
    model = setup_model_from_model_card('./videoseal/cards/videoseal.yaml')
    model.eval()
    model.to(args.device)
    # Load dataset
    if args.attack_type == 'removal':
        dataset = video_dataset(os.path.join(args.dataset[args.video_model], args.genre))
    else:
        dataset = video_dataset(args.dataset['kinetics'])
    data = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    # Set up for WEvade attack
    attack = removal if args.attack_type == 'removal' else forgery
    gt_watermark = torch.zeros(1, args.secret_size).to(args.device)
    gt_watermark[:, 1::2] = 1

    for video, video_name in tqdm(data, desc='Video'):
        video_name = video_name[0]
        video = video.squeeze()
        if video_name not in results_pk:
            results_pk[video_name] = {}
        elif len(results_pk[video_name]) != video.shape[0]:
            results_pk[video_name] = {}
        else:
            print(f'{video_name} already processed')
            continue
        # Apply watermark for removal attack
        if args.attack_type == 'removal':
            video = video.to(args.device)
            video = model.embed(imgs=video, msgs=gt_watermark, is_video=True)["imgs_w"]
            video = video.detach().cpu()
        for frame_id, frame in enumerate(video):
            frame = frame.clone().unsqueeze(0).to(args.device)

            # Process both attack and detection for each frame
            methods = {
                'detect': detect(frame, gt_watermark, model, args),
                'attack': attack(frame, gt_watermark, model, args)
            }
            # Store results in a more structured way
            results_pk[video_name][frame_id] = {}
            for method_name, (bit_acc, bound, ssim, psnr, decoded_logits) in methods.items():
                results_pk[video_name][frame_id][method_name] = {
                    'bit_acc': bit_acc,
                    'bound': bound,
                    'ssim': ssim,
                    'psnr': psnr,
                    'decoded_logits': decoded_logits
                }
        # Save results after processing each video
        pd.to_pickle(results_pk, results_path)


def parse_args():
    parser = argparse.ArgumentParser(description='StegaStamp wbox attack')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--attack_type', type=str, default='removal', choices=['removal', 'forgery'])
    parser.add_argument('--secret_size', type=int, default=96)
    parser.add_argument('--WEvade-type', default='WEvade-W-I', type=str, help='Using WEvade-W-I/II.')
    parser.add_argument('--genre', type=str, default='cartoon')
    parser.add_argument('--video_model', type=str, default='hunyuan')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = EasyDict({**cfg, **parse_args()})
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(args)
    main(args)