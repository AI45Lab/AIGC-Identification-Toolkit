"""
VideoSeal 图像水印适配器
将单张图像视作单帧视频，复用 VideoSeal 的 embed/detect 能力
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Dict, Any, Optional

try:
    # 相对导入（当作为包运行时）
    from ..video_watermark.videoseal_wrapper import create_videoseal_wrapper
except ImportError:
    # 绝对导入（当 src 在路径中时）
    from video_watermark.videoseal_wrapper import create_videoseal_wrapper


class VideoSealImageWatermark:
    """基于 VideoSeal 的图像水印处理器"""

    def __init__(self, device: Optional[str] = None, lowres_attenuation: bool = True):
        self.wrapper = create_videoseal_wrapper(device=device)
        self.lowres_attenuation = lowres_attenuation

    def _to_tensor(self, image_input: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """将输入转换为形状 (1, C, H, W)、值域[0,1] 的张量"""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, torch.Tensor):
            t = image_input
            if t.dim() == 3 and t.shape[0] in (1, 3):  # C,H,W
                t = t.unsqueeze(0)
            elif t.dim() == 3 and t.shape[-1] in (1, 3):  # H,W,C
                t = t.permute(2, 0, 1).unsqueeze(0)
            elif t.dim() == 4 and t.shape[0] in (1,):  # 1,C,H,W
                pass
            else:
                raise ValueError(f"Unsupported tensor shape for image: {t.shape}")
            t = t.float()
            if t.max() > 1.0:  # assume [0,255]
                t = t / 255.0
            return t
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        arr = np.array(img).astype(np.float32) / 255.0  # H,W,C in [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        return t

    def _to_pil(self, video_tensor: torch.Tensor) -> Image.Image:
        """将形状 (1, C, H, W)、值域[0,1] 的张量转换为 PIL.Image"""
        t = torch.clamp(video_tensor[0].detach().cpu(), 0.0, 1.0)
        arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # H,W,C
        return Image.fromarray(arr)

    def embed(self, image_input: Union[str, Image.Image, torch.Tensor], message: str, **kwargs) -> Image.Image:
        """在图像中嵌入水印，返回带水印图像 (PIL)"""
        if not isinstance(message, str) or len(message) == 0:
            raise ValueError("VideoSeal requires a non-empty string 'message' to embed.")
        img_tensor = self._to_tensor(image_input)  # 1,C,H,W
        watermarked = self.wrapper.embed_watermark(
            img_tensor, message=message, is_video=False, lowres_attenuation=self.lowres_attenuation
        )
        return self._to_pil(watermarked)

    def extract(self, image_input: Union[str, Image.Image, torch.Tensor], *, chunk_size: int = 16, replicate: int = 32, **kwargs) -> Dict[str, Any]:
        """从图像中提取水印，返回检测结果字典

        Args:
            chunk_size: 分块大小（传递给检测器），默认16平衡效率和精度
            replicate: 将单帧图像重复为多帧以提高平均稳定性，默认32提升检测准确率
        """
        img_tensor = self._to_tensor(image_input)  # 1,C,H,W
        if replicate and replicate > 1:
            # 复制为 (replicate, C, H, W)
            img_tensor = img_tensor.repeat(replicate, 1, 1, 1)
        result = self.wrapper.extract_watermark(img_tensor, is_video=False, chunk_size=int(max(1, chunk_size)))
        return result


