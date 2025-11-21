import json
import torch
import os
import argparse
import time
import attack_mask as attack
from attack_utils import get_model, read_imagenet_data_specify, save_results
from foolbox.distances import l2, linf
import numpy as np
from PIL import Image
import torch_dct
import gc
from tqdm import tqdm
import torchvision
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="results", help="Output folder")
    parser.add_argument(
        '--seed',
        type=int,
        default=20,
        help='The random seed you choose'
    )
    parser.add_argument(
        '--max_queries',
        type=int,
        default=1000,
        help='The max number of queries in model'
    )
    parser.add_argument(
        '--ratio_mask',
        type=float,
        default=0.1,
        help='ratio of mask'
    )
    parser.add_argument(
        '--dim_num',
        type=int,
        default=1,
        help='the number of picked dimensions'
    )
    parser.add_argument(
        '--max_iter_num_in_2d',
        type=int,
        default=2,
        help='the maximum iteration number of attack algorithm in 2d subspace'
    )
    parser.add_argument(
        '--init_theta',
        type=int,
        default=2,
        help='the initial angle of a subspace=init_theta*np.pi/32'
    )
    parser.add_argument(
        '--init_alpha',
        type=float,
        default=np.pi/2,
        help='the initial angle of alpha'
    )
    parser.add_argument(
        '--plus_learning_rate',
        type=float,
        default=0.1,
        help='plus learning_rate when success'
    )
    parser.add_argument(
        '--minus_learning_rate',
        type=float,
        default=0.005,
        help='minus learning_rate when fail'
    )
    parser.add_argument(
        '--half_range',
        type=float,
        default=0.1,
        help='half range of alpha from pi/2'
    )
    parser.add_argument(
        '--H',
        type=int,
        default=576
    )
    parser.add_argument(
        '--W',
        type=int,
        default=1024
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stable-video_realistic"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="videoseal"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="ba-mean"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="removal"
    )
    parser.add_argument(
        '--num_test_video',
        type=int,
        default=40
    )
    return parser.parse_args()


def find_videos(folder_path):
    video_extensions = ['.mp4']
    video_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files


class video_dataset(torch.utils.data.Dataset):
    def __init__(self, video_path):
        self.video_files = []
        for video_file in os.listdir(video_path):
            if video_file.endswith('.mp4'):
                self.video_files.append(os.path.join(video_path, video_file))
        self.video_files.sort()

    def resize_frame(self, frame):
        return cv2.resize(frame, (512, 512))

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        video = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.resize_frame(frame)
            video.append(torch.from_numpy(frame.transpose(2, 0, 1).astype('float32') / 255))

        return torch.stack(video, dim=1)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video = self.load_video(video_file)
        video = video.permute(1, 0, 2, 3)
        video_name = video_file.split('/')[-1]
        return video, video_name


class video_dataset_revmark(torch.utils.data.Dataset):
    def __init__(self, video_path):
        self.video_files = []
        for video_file in os.listdir(video_path):
            if video_file.endswith('.mp4'):
                self.video_files.append(os.path.join(video_path, video_file))
        self.video_files.sort()

    def resize_frame(self, frame):
        return cv2.resize(frame, (128, 128))

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        video = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video.append(torch.from_numpy(frame[:128, :128].transpose(2, 0, 1).astype('float32') / 255))

        return torch.stack(video, dim=1)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video = self.load_video(video_file)
        video = self.reshape_tensor(video)
        video_name = video_file.split('/')[-1]
        return video, video_name

    def reshape_tensor(self, tensor):
        c, y, h, w = tensor.shape
        x = (y + 7) // 8
        pad_size = x * 8 - y
        if pad_size > 0:
            tensor = torch.cat([tensor, tensor[:, :pad_size]], dim=1)
        tensor = tensor.view(c, x, 8, h, w)
        tensor = tensor.permute(1, 0, 2, 3, 4)
        return tensor


if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ###############################
    ### VideoSeal
    if args.model == 'videoseal':
        print("Load Model")
        import videoseal
        from videoseal.evals.metrics import bit_accuracy
        from videoseal.utils.cfg import setup_dataset, setup_model_from_checkpoint, setup_model_from_model_card
    
        model = setup_model_from_model_card('./videoseal/cards/videoseal.yaml')
        model.eval()
        model.to(device)
        ta_model = attack.TA(model, input_device=device)
    
        print("Load Data")
        folder_path = './kinetics-dataset/'
        videos = find_videos(folder_path)[:args.num_test_video]
        print("{} videos loaded".format(len(videos)))
        for video_path in tqdm(videos):
            video = torchvision.io.read_video(video_path, output_format="TCHW")
            video = video[0].float() / 255.0
            video = video[:14, :, :, :]

            print(video.shape)
            print("Attack !")
            time_start = time.time()
            args.W = video.shape[3]
            args.H = video.shape[2]
            my_advs, q_list, l2, linf = ta_model.attack_video(args, video.to(device))
            print('TA Attack Done')
            print("{:.2f} s to run".format(time.time() - time_start))
            print('L2 distance: {}'.format(l2))
            print('Linf distance: {}'.format(linf))


    ### StegaStamp
    elif args.model == 'stegastamp'
        print("Load Model")
        import yaml
        import StegaStamp
        from easydict import EasyDict
        with open('eval.yaml', 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))
        argss = EasyDict({**cfg})
        model = model.StegaStampDecoder(secret_size=argss.secret_size)
        model.load_state_dict(torch.load(os.path.join(argss.adv_pretrained, 'decoder.pth'), map_location=device))
        if argss.cuda:
            model = model.to(device)
        model.eval()
        ta_model = attack.TA(model, input_device=device)

        print("Load Data")
        dataset = video_dataset('./kinetics-dataset/')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for i, (video, video_name) in enumerate(tqdm(dataloader)):
            if i >= args.num_test_video:
                break
            video = video.to(device)
            video = video[0].float() / 255.0
            video = video[:14, :, :, :]

            print(video.shape)
            print("Attack !")
            time_start = time.time()
            args.W = video.shape[3]
            args.H = video.shape[2]
            my_advs, q_list, l2, linf = ta_model.attack_video(args, video.to(device))
            print('TA Attack Done')
            print("{:.2f} s to run".format(time.time() - time_start))
            print('L2 distance: {}'.format(l2))
            print('Linf distance: {}'.format(linf))


    ### REVMark
    elif args.model == 'revmark':
        print("Load Model")
        from REVMark import Encoder, Decoder, framenorm
        model = Decoder(96, [8,128,128]).to(device).eval()
        model.load_state_dict(torch.load('checkpoints/Decoder.pth'))
        model.tasblock.enable = True
        ta_model = attack.TA(model, input_device=device)

        print("Load Data")
        dataset = video_dataset_revmark('./kinetics-dataset/')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for i, (video, video_name) in enumerate(tqdm(dataloader)):
            if i >= args.num_test_video:
                break
            video = video[0][:2, :, :, :, :]
            F, C, K, H, W = video.shape
            video = video.reshape(F * K, C, H, W)

            print(video.shape)
            print("Attack !")
            time_start = time.time()
            args.W = video.shape[3]
            args.H = video.shape[2]
            my_advs, q_list, l2, linf = ta_model.attack_video(args, video.to(device))
            print('TA Attack Done')
            print("{:.2f} s to run".format(time.time() - time_start))
            print('L2 distance: {}'.format(l2))
            print('Linf distance: {}'.format(linf))