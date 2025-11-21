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
    return parser.parse_args()


def load_videos_from_folder(folder_path):
    video_tensors = []
    video_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.pt')
    ])

    for video_path in video_files:
        tensor = torch.load(video_path)
        video_tensors.append(tensor)

    shapes = [v.shape for v in video_tensors]
    assert len(set(shapes)) == 1, f"Video shapes are not consistent: {set(shapes)}"

    return torch.stack(video_tensors, dim=0)


if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ###############################
    print("Load Data")
    videos = load_videos_from_folder(f'../{args.dataset}/')
    print(videos.shape)
    print("{} videos loaded".format(len(videos)))


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

    ### StegaStamp
    elif args.model == 'stegastamp':
        import yaml
        import StegaStamp
        from easydict import EasyDict
        with open('eval.yaml', 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))
        argss = EasyDict({**cfg})

        decoder = model.StegaStampDecoder(secret_size=argss.secret_size)
        decoder.load_state_dict(torch.load(os.path.join(argss.adv_pretrained, 'decoder.pth'), map_location=device))
        if argss.cuda:
            model = decoder.to(device)
        model.eval()
        ta_model = attack.TA(model, input_device=device)

    ### REVMark
    elif args.model == 'revmark':
        from REVMark import Encoder, Decoder, framenorm
        model = Decoder(96, [8,128,128]).to(device).eval()
        model.load_state_dict(torch.load('checkpoints/Decoder.pth'))
        model.tasblock.enable = True
        B, F, C, K, H, W = videos.shape
        videos = videos.reshape(B, F * K, C, H, W)
        ta_model = attack.TA(model, input_device=device)


    ###############################
    for video in videos:
        print(video.shape)
        print("Attack !")
        time_start = time.time()
        my_advs, q_list, l2, linf = ta_model.attack_video(args, video.to(device))
        print('TA Attack Done')
        print("{:.2f} s to run".format(time.time() - time_start))
        print('L2 distance: {}'.format(l2))
        print('Linf distance: {}'.format(linf))