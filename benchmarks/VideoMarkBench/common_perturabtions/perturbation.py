from jpeg import JPEG
from gaussian import Gaussian
from gaussian_blur import GaussianBlur
from crop import Crop
from frame_average import FrameAverage
from frame_switch import FrameSwitch
from frame_remove import FrameRemove
from mpeg4 import MPEG4


def parse_args():
    parser.add_argument('--perturbation', type=str, default='jpeg')
    parser.add_argument('--parameter', type=float, default=90)
    parser.add_argument('--path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    video = torch.load(args.path)

    if args.perturbation == 'jpeg':
        perturbation = JPEG(args.parameter)
        for i in range(video.shape[0]):
            video[i] = perturbation(video[i].unsqueeze(0))[0]

    elif args.perturbation == 'gaussian':
        perturbation = Gaussian(parameter)
        for i in range(video.shape[0]):
            video[i] = perturbation(video[i].unsqueeze(0))[0]

    elif args.perturbation == 'gaussian_blur':
        perturbation = GaussianBlur(parameter)
        for i in range(video.shape[0]):
            video[i] = perturbation(video[i].unsqueeze(0))[0]

    elif args.perturbation == 'crop':
        perturbation = Crop(parameter)
        video = perturbation(video)

    elif args.perturbation == 'frame_average':
        perturbation = FrameAverage(parameter)
        video = perturbation(video)

    elif args.perturbation == 'frame_switch':
        perturbation = FrameSwitch(parameter)
        video = perturbation(video)

    elif args.perturbation == 'frame_remove':
        perturbation = FrameRemove(parameter)
        video = perturbation(video)

    elif args.perturbation == 'mpeg4':
        save_path = "PATH TO SAVE VIDEO"
        perturbation = MPEG4(save_path, parameter, 'libx264')
        perturbation(video)

    else:
        raise ValueError(f"Invalid perturbation: {args.perturbation}")