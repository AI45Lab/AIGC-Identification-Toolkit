import torch
import torch.nn.functional as F
import torch.nn as nn

class FrameAverage(nn.Module):
    def __init__(self, num_frames):
        super(FrameAverage, self).__init__()
        self.m = num_frames

    def forward(self, video):
        """
        对每帧在时间维度取邻近 m 帧的均值，m=1 时保持不变。

        参数:
        - video: torch.Tensor, [N, 3, H, W]，输入视频张量
        - m: int, 取均值的窗口大小 (m=1 到 m=4)

        返回:
        - result: torch.Tensor, [N, 3, H, W]，处理后的张量
        """
        assert video.ndim == 4, "输入视频的 shape 应为 [N, 3, H, W]"

        # 如果 m == 1，直接返回原视频
        if self.m == 1:
            return video.clone()

        N, C, H, W = video.shape

        # 结果张量，初始化为 0
        result = torch.zeros_like(video)

        # 对每一帧进行加和（滑动窗口范围为 m 帧）
        for i in range(N):
            # 计算当前帧的平均窗口范围
            start = max(0, i - (self.m - 1) // 2)  # 防止溢出，向前最多取 (m-1)//2 帧
            end = min(N, i + (self.m // 2) + 1)  # 向后最多取 m//2 帧

            # 累加窗口内的帧
            result[i] = video[start:end].mean(dim=0)

        return result