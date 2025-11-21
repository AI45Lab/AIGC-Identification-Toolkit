import torch
import random
import torch.nn as nn

class FrameRemove(nn.Module):
    def __init__(self, p):
        super(FrameRemove, self).__init__()
        self.p = p

    def forward(self, video):
        """
        以概率 p 移除视频中的每一帧。

        参数:
        - video: torch.Tensor, [N, 3, H, W]，输入视频张量
        - p: float, 每帧被移除的概率，范围 [0, 1]

        返回:
        - reduced_video: torch.Tensor, 移除帧后的新视频，形状为 [M, 3, H, W]
        """
        assert video.ndim == 4, "输入视频的形状应为 [N, 3, H, W]"
        assert 0 <= self.p <= 1, "p 应该在 [0, 1] 范围内"

        N = video.shape[0]

        # 生成一个布尔掩码，True 表示保留，False 表示移除
        mask = torch.tensor([random.random() >= self.p for _ in range(N)])

        # 至少保留一帧，防止全部移除
        if mask.sum() == 0:
            mask[random.randint(0, N - 1)] = True

        # 根据掩码选择需要保留的帧
        reduced_video = video[mask]

        return reduced_video