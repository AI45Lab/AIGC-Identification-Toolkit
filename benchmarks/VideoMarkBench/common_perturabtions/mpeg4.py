import torch
import torch.nn as nn
import numpy as np
import ffmpeg
import os
from tqdm import tqdm
import cv2


class MPEG4(nn.Module):
    def __init__(self, quality, path=None, codec='libx264'):
        super(MPEG4, self).__init__()
        self.path = path
        self.quality = quality
        self.codec = codec

    def forward(self, video):
        frames = (video * 255).byte().cpu().numpy()
        frames = np.transpose(frames, (0, 2, 3, 1))
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='512x512', r=25)
            .output(self.path, vcodec=self.codec, pix_fmt='yuv420p', crf=self.quality)
            .run_async(pipe_stdin=True)
        )
        for frame in frames:
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()