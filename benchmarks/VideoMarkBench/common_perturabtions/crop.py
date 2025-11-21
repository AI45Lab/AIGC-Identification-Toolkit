import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop


class Crop(nn.Module):
    def __init__(self, crop_ratio):
        super(Crop, self).__init__()
        self.crop_ratio = crop_ratio ** 0.5

    def forward(self, video_tensor):
        if not isinstance(video_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        T, C, H, W = video_tensor.shape

        crop_height = int(H * self.crop_ratio)
        crop_width = int(W * self.crop_ratio)

        i, j, h, w = RandomCrop.get_params(video_tensor[0], output_size=(crop_height, crop_width))

        cropped_resized_video = []
        for frame in video_tensor:
            cropped_frame = F.crop(frame, i, j, h, w)
            resized_frame = F.resize(cropped_frame, (H, W))
            cropped_resized_video.append(resized_frame)

        return torch.stack(cropped_resized_video, dim=0).clamp(0., 1.)