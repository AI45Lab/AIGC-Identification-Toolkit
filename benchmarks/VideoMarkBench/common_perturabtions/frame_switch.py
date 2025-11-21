import torch
import random
import torch.nn as nn

class FrameSwitch(nn.Module):
    def __init__(self, p):
        super(FrameSwitch, self).__init__()
        self.p = p

    def forward(self, video):
        """
        对每一帧以概率 p 和相邻帧（前一帧或后一帧）随机交换。

        参数:
        - video: torch.Tensor, [N, 3, H, W]，输入视频张量
        - p: float, 每帧与邻近帧交换的概率，范围 [0, 1]

        返回:
        - swapped_video: torch.Tensor, 交换后的视频张量，形状与输入相同
        """
        assert video.ndim == 4, "输入视频的形状应为 [N, 3, H, W]"
        assert 0 <= self.p <= 1, "p 应该在 [0, 1] 范围内"

        N = video.shape[0]
        swapped_video = video.clone()

        for i in range(N):
            if random.random() < self.p:
                # 随机选择前一帧 (i-1) 或后一帧 (i+1)
                if i == 0:
                    swap_idx = 1  # 第一帧只能与后一帧交换
                elif i == N - 1:
                    swap_idx = N - 2  # 最后一帧只能与前一帧交换
                else:
                    swap_idx = i + random.choice([-1, 1])

                # 执行帧交换
                swapped_video[i], swapped_video[swap_idx] = swapped_video[swap_idx], swapped_video[i]

        return swapped_video