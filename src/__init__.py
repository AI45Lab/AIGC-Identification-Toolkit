"""
Unified Watermark Tool - 多模态水印工具
支持文本和图像的水印嵌入与提取
"""

__version__ = "0.1.0"
__author__ = "码农团队"

from .unified.watermark_tool import WatermarkTool
from .text_watermark.credid_watermark import CredIDWatermark
from .image_watermark.image_watermark import ImageWatermark

__all__ = ['WatermarkTool', 'CredIDWatermark', 'ImageWatermark'] 