# CredID Text Watermarking Algorithm
# 
# This package contains the CredID watermarking algorithm implementation
# copied from the credible_LLM_watermarking project.

__version__ = "0.1.0"
__author__ = "CredID Team"

# 暴露核心组件
from .watermarking.CredID.message_model_processor import WmProcessorMessageModel
from .watermarking.CredID.random_message_model_processor import WmProcessorRandomMessageModel  
from .watermarking.CredID.base_processor import WmProcessorBase

__all__ = [
    'WmProcessorMessageModel',
    'WmProcessorRandomMessageModel', 
    'WmProcessorBase'
] 