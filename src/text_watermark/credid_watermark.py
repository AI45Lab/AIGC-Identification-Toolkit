"""
CredID文本水印算法封装
基于原始CredID实现的高级封装，提供简单易用的嵌入和提取接口
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    LogitsProcessorList, 
    PreTrainedModel, 
    PreTrainedTokenizer
)
import logging

# 导入CredID核心组件
from .credid.watermarking.CredID.message_model_processor import WmProcessorMessageModel
from .credid.watermarking.CredID.random_message_model_processor import WmProcessorRandomMessageModel
from .credid.watermarking.CredID.message_models.lm_message_model import LMMessageModel
from .credid.watermarking.CredID.message_models.random_message_model import RandomMessageModel


class CredIDWatermark:
    """
    CredID文本水印算法封装
    
    这是一个高级封装类，隐藏了CredID算法的复杂性，提供简单易用的API。
    支持两种模式：
    1. LM模式：使用语言模型作为"专家顾问"，质量更高但速度稍慢
    2. Random模式：使用随机化规则，速度更快但质量稍低
    
    Example:
        config = {
            'model_name': 'huggyllama/llama-7b',
            'mode': 'lm',  # or 'random'
            'lm_params': {...},
            'wm_params': {...}
        }
        watermark = CredIDWatermark(config)
        result = watermark.embed(model, tokenizer, "Hello world", "secret")
        extracted = watermark.extract(result['watermarked_text'])
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化CredID水印处理器
        
        Args:
            config: 配置字典，包含以下参数：
                - model_name: 预训练模型名称
                - mode: 'lm' 或 'random'，决定使用哪种CredID模式
                - lm_params: LM模式参数
                - wm_params: 水印处理参数
                - device: 设备设置，默认auto
        """
        self.config = config
        self.mode = config.get('mode', 'lm')  # 默认使用LM模式
        
        # 设备设置
        device_config = config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)
        
        # 提取各种参数
        self.model_name = config.get('model_name', 'huggyllama/llama-7b')
        self.lm_params = config.get('lm_params', {})
        self.wm_params = config.get('wm_params', {})
        
        # 初始化核心组件
        self.message_model = None
        self.wm_processor = None
        self.original_message_length = None  # 记录原始消息的段数，用于循环提取
        self.last_encoded_binary = None  # 存储最后一次编码的binary数据，用于候选消息管理
        
        logging.info(f"CredIDWatermark initialized in {self.mode} mode on {self.device}")
    
    def _reset_message_state(self):
        """
        重置消息相关的状态，用于处理新消息时清理之前的状态
        """
        self.original_message_length = None
        self.last_encoded_binary = None
    
    def _setup_processors(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        设置CredID处理器
        根据配置的模式选择使用LM模式还是Random模式
        
        Args:
            model: 预训练语言模型
            tokenizer: 对应的分词器
        """
        try:
            if self.mode == 'lm':
                # LM模式：使用语言模型作为专家顾问
                self.message_model = LMMessageModel(
                    tokenizer=tokenizer,
                    lm_model=model,
                    lm_tokenizer=tokenizer,
                    delta=self.lm_params.get('delta', 1.5),
                    lm_prefix_len=self.lm_params.get('prefix_len', 5),
                    seed=self.lm_params.get('seed', 42),
                    lm_topk=self.lm_params.get('topk', -1),
                    message_code_len=self.lm_params.get('message_len', 10),
                    random_permutation_num=self.lm_params.get('permutation_num', 50),
                    hash_prefix_len=self.lm_params.get('hash_prefix_len', 1),
               
                    shifts=self.lm_params.get('shifts', [21, 24, 3, 8, 14,2, 4, 28, 31, 3, 8, 14, 2, 4, 28, 16, 7, 19, 25, 11, 33, 1, 0, 8, 34])
                )
                
            elif self.mode == 'random':
                # Random模式：使用随机化规则
                self.message_model = RandomMessageModel(
                    tokenizer=tokenizer,
                    lm_tokenizer=tokenizer,
                    delta=self.lm_params.get('delta', 1.5),
                    seed=self.lm_params.get('seed', 42),
                    message_code_len=self.lm_params.get('message_len', 10),
                    hash_prefix_len=self.lm_params.get('hash_prefix_len', 1),
                    device=self.device,
                    # Random模式需要的额外参数
                    shifts=self.lm_params.get('shifts', [21, 24, 3, 8, 14,2, 4, 28, 31, 3, 8, 14, 2, 4, 28, 16, 7, 19, 25, 11, 33, 1, 0, 8, 34])
                )
            else:
                raise ValueError(f"Unsupported mode: {self.mode}. Use 'lm' or 'random'.")
                
            logging.info(f"Message model ({self.mode}) initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to setup message model: {e}")
            raise
    
    def _create_wm_processor(self, message: str) -> Union[WmProcessorMessageModel, WmProcessorRandomMessageModel]:
        """
        创建水印处理器
        
        Args:
            message: 要嵌入的水印信息
            
        Returns:
            配置好的水印处理器
        """
        try:
            
            original_binary = self._message_to_binary(message, 'auto')
            
            # 记录原始消息长度，用于提取时的循环限制
            self.original_message_length = len(original_binary)
            
            # 启用循环嵌入：扩展消息段以支持长文本生成
            max_tokens = self.config.get('max_new_tokens', 300)  
            encode_len = self.lm_params.get('message_len', 10) * self.wm_params.get('encode_ratio', 4)
            needed_segments = (max_tokens + encode_len - 1) // encode_len

            # 关键保护：限制段数不超过shift模式可支持的数量，避免下游索引越界
            # 注意：shifts数组的长度决定了最大支持的段数
            # 但由于内部实现可能有索引偏移，我们保守地使用 len(shifts) - 1
            default_shifts = [21, 24, 3, 8, 14,2, 4, 28, 31, 3, 8, 14, 2, 4, 28, 16, 7, 19, 25, 11, 33, 1, 0, 8, 34]
            shifts = self.lm_params.get('shifts', default_shifts)
            # 保守限制：使用 len(shifts) - 1 以避免边界问题
            max_supported_segments = self.wm_params.get('max_segments', len(shifts) - 1 if shifts else 10)
            if needed_segments > max_supported_segments:
                needed_segments = max_supported_segments
            
            # 如果需要更多段，循环重复消息
            if needed_segments > len(original_binary):
                binary_message = []
                for i in range(needed_segments):
                    binary_message.append(original_binary[i % len(original_binary)])
                # print(f"循环嵌入: {len(original_binary)} → {len(binary_message)} 段 (支持{max_tokens}tokens)")
            else:
                binary_message = original_binary            
            if self.mode == 'lm':
                processor = WmProcessorMessageModel(
                    message=binary_message,
                    message_model=self.message_model,
                    tokenizer=self.message_model.tokenizer,
                    encode_ratio=self.wm_params.get('encode_ratio', 10),
                    seed=self.wm_params.get('seed', 42),
                    strategy=self.wm_params.get('strategy', 'vanilla'),
                    max_confidence_lbd=self.wm_params.get('max_confidence', 0.5),
                    top_k=self.wm_params.get('top_k', 1000)
                )
            else:  # random mode
                processor = WmProcessorRandomMessageModel(
                    message=binary_message,
                    message_model=self.message_model,
                    tokenizer=self.message_model.tokenizer,
                    encode_ratio=self.wm_params.get('encode_ratio', 10),
                    seed=self.wm_params.get('seed', 42),
                    strategy=self.wm_params.get('strategy', 'vanilla'),
                    max_confidence_lbd=self.wm_params.get('max_confidence', 0.5),
                    top_k=self.wm_params.get('top_k', 1000)
                )
            
            return processor
            
        except Exception as e:
            logging.error(f"Failed to create watermark processor: {e}")
            raise
    
    def embed(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
              prompt: str, message: Union[str, List[int], List[str]], 
              segmentation_mode: str = 'auto') -> Dict[str, Any]:
        """
        在文本生成过程中嵌入水印
        
        这是主要的嵌入接口。它会：
        1. 设置CredID处理器
        2. 将水印处理器集成到生成过程中
        3. 生成带水印的文本
        
        Args:
            model: HuggingFace语言模型
            tokenizer: 对应的分词器
            prompt: 输入提示文本
            message: 要嵌入的水印信息（支持字符串、整数列表、字符串列表）
            segmentation_mode: 分割模式 ('auto', 'smart', 'whole', 'spaces')
            
        Returns:
            包含水印文本和元数据的字典：
            {
                'watermarked_text': str,     # 带水印的生成文本
                'original_message': str,     # 原始水印信息
                'binary_message': List[int], # 二进制水印
                'prompt': str,               # 输入提示
                'success': bool,             # 是否成功
                'metadata': dict             # 额外元数据
            }
        """
        try:
            # 0. 重置消息状态，准备处理新消息
            self._reset_message_state()
            
            # 1. 设置处理器（如果还没设置）
            if self.message_model is None:
                self._setup_processors(model, tokenizer)
            
            # 2. 创建水印处理器
            wm_processor = self._create_wm_processor(message)
            
            # 3. 配置生成参数
            generation_config = {
                'max_new_tokens': self.config.get('max_new_tokens', 50),   # 合理的默认长度
                'num_beams': self.config.get('num_beams', 1),  # 减少beam search加速生成
                'do_sample': self.config.get('do_sample', True),
                'temperature': self.config.get('temperature', 0.7),
                'top_p': self.config.get('top_p', 0.9),
                'top_k': self.config.get('top_k', 50),
                'repetition_penalty': self.config.get('repetition_penalty', 1.2),  # 防止重复
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id
            }
            
            # 4. 创建logits处理器列表
            logits_processors = LogitsProcessorList([wm_processor])
            
            # 5. 编码输入
            input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
            input_ids = input_ids.to(model.device)
            
            # 6. 生成带水印文本
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    logits_processor=logits_processors,
                    **generation_config
                )
            
            # 7. 解码输出
            # 只取新生成的部分（去除输入prompt）
            new_tokens = output_ids[0][input_ids.shape[1]:]
            watermarked_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 8. 准备返回结果
            binary_message = self._message_to_binary(message, segmentation_mode)
            
            # 保存编码后的binary数据供候选消息管理器使用
            self.last_encoded_binary = binary_message.copy()
            
            return {
                'watermarked_text': watermarked_text,
                'original_message': message,
                'binary_message': binary_message,
                'prompt': prompt,
                'success': True,
                'metadata': {
                    'mode': self.mode,
                    'model_name': self.model_name,
                    'input_length': input_ids.shape[1],
                    'output_length': len(new_tokens),
                    'generation_config': generation_config
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to embed watermark: {e}")
            return {
                'watermarked_text': None,
                'original_message': message,
                'prompt': prompt,
                'success': False,
                'error': str(e),
                'metadata': {'mode': self.mode}
            }
    
    def extract(self, watermarked_text: str, 
                model: Optional[PreTrainedModel] = None,
                tokenizer: Optional[PreTrainedTokenizer] = None,
                candidates_messages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        从水印文本中提取水印信息
        
        Args:
            watermarked_text: 可能包含水印的文本
            model: 语言模型（LM模式需要）
            tokenizer: 分词器（LM模式需要）
            candidates_messages: 候选消息列表，如果提供则只搜索这些消息
            
        Returns:
            包含提取信息和置信度的字典
        """
        try:
            # 1. 检查模式和参数
            if self.mode == 'lm' and (model is None or tokenizer is None):
                raise ValueError("LM mode requires both model and tokenizer for extraction")
            
            # 2. 设置处理器（如果还没设置）
            if self.message_model is None:
                if self.mode == 'lm':
                    self._setup_processors(model, tokenizer)
                else:
                    # Random模式需要一个默认的tokenizer
                    from transformers import AutoTokenizer
                    default_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    if default_tokenizer.pad_token is None:
                        default_tokenizer.pad_token = default_tokenizer.eos_token
                    self._setup_processors(model, default_tokenizer)
            
            # 3. 创建一个临时的水印处理器用于解码
            # 🔧 关键修复：保护原始消息长度不被临时处理器覆盖
            saved_original_length = self.original_message_length
            temp_message = "0"  # 占位符
            wm_processor = self._create_wm_processor(temp_message)
            # 立即恢复原始状态
            self.original_message_length = saved_original_length
            
            # 4. 准备候选消息编码
            if candidates_messages is not None:
                # 🔧 简化：收集所有候选消息的所有编码
                candidate_codes = []
                for msg in candidates_messages:
                    encoded = self._message_to_binary(msg, "smart")
                    if encoded:
                        candidate_codes.extend(encoded)  # 添加所有段的编码
                
                # 去重并排序
                candidate_codes = sorted(list(set(candidate_codes)))
                
                # 转换为tensor
                candidate_messages_tensor = torch.tensor(candidate_codes, device=self.message_model.device)
            else:
                # 如果没有提供候选消息，则搜索全范围
                candidate_messages_tensor = None
                print(f"🔍 搜索全范围 0-{2**self.lm_params.get('message_len', 10)-1} 的所有编码")
            
            # 5. 执行解码
            batch_size = self.config.get('decode_batch_size', 16)
            disable_tqdm = self.config.get('disable_tqdm', False)
            
            decoded_messages, (all_log_probs, detailed_confidences) = wm_processor.decode(
                watermarked_text,
                messages=candidate_messages_tensor,  # 传入候选消息
                batch_size=batch_size,
                disable_tqdm=disable_tqdm,
                non_analyze=False
            )
            
            # 6.  循环提取限制：只取原始消息长度的段数
            if decoded_messages and len(decoded_messages) > 0:
                # 如果有记录的原始消息长度，限制解码结果的长度
                if self.original_message_length and self.original_message_length > 0:
                    if len(decoded_messages) > self.original_message_length:
                        decoded_messages = decoded_messages[:self.original_message_length]
                        print(f"🔧 循环提取限制: 截取前{self.original_message_length}段 (原始消息长度)")
                    else:
                        print(f"🔧 提取段数({len(decoded_messages)}) ≤ 原始长度({self.original_message_length})，无需截取")
                else:
                    print("⚠️  未找到原始消息长度记录，使用完整解码结果")
            
            # 7. 智能处理多段解码结果
            if decoded_messages and len(decoded_messages) > 0:
                # 如果有候选消息，进行智能匹配
                if candidates_messages is not None:
                    extracted_message, match_confidence = self._match_decoded_with_candidates(
                        decoded_messages, candidates_messages
                    )
                else:
                    # 没有候选消息，使用原有逻辑
                    if len(decoded_messages) == 1:
                        extracted_message = self._binary_to_message([decoded_messages[0]])
                    else:
                        extracted_message = self._binary_to_message(decoded_messages)
                    match_confidence = 0.0
                
                # 计算平均置信度
                if detailed_confidences and len(detailed_confidences) > 0:
                    confidences = [conf[2] for conf in detailed_confidences if len(conf) > 2]
                    avg_confidence = np.mean(confidences) if confidences else 0.0
                    
                    # 结合匹配置信度
                    if match_confidence > 0:
                        final_confidence = (avg_confidence + match_confidence) / 2
                    else:
                        final_confidence = avg_confidence
                    
                    # 判断是否成功（基于置信度阈值）
                    confidence_threshold = self.config.get('confidence_threshold', 0.6)
                    success = final_confidence > confidence_threshold
                else:
                    avg_confidence = 0.0
                    final_confidence = match_confidence  #  修复：使用匹配置信度
                    success = final_confidence > self.config.get('confidence_threshold', 0.6)
                
                return {
                    'extracted_message': extracted_message,
                    'binary_message': decoded_messages,
                    'confidence': float(final_confidence),
                    'success': success,
                    'detailed_confidence': detailed_confidences,
                    'metadata': {
                        'mode': self.mode,
                        'text_length': len(watermarked_text),
                        'num_decoded_segments': len(decoded_messages),
                        'detection_method': 'CredID',
                        'confidence_threshold': confidence_threshold,
                        'search_space': len(candidate_codes) if candidates_messages else 2**self.lm_params.get('message_len', 10),
                        'candidates_provided': candidates_messages is not None
                    }
                }
            else:
                return {
                    'extracted_message': None,
                    'binary_message': [],
                    'confidence': 0.0,
                    'success': False,
                    'detailed_confidence': [],
                    'error': 'No watermark detected',
                    'metadata': {'mode': self.mode}
                }
                
        except Exception as e:
            logging.error(f"Failed to extract watermark: {e}")
            return {
                'extracted_message': None,
                'binary_message': [],
                'confidence': 0.0,
                'success': False,
                'error': str(e),
                'metadata': {'mode': self.mode}
            }
    
    def _message_to_binary(self, message: Union[str, List[int], List[str]], 
                          segmentation_mode: str = 'auto') -> List[int]:
        """
        将消息转换为CredID兼容的消息格式
        
        支持多种消息格式和分割模式：
        1. 单个字符串：转换为单段消息
        2. 句子字符串：分割为多段消息
        3. 混合字符串：智能分割字母和数字部分
        4. 整数列表：直接使用
        5. 字符串列表：每个字符串转换为一段
        
        Args:
            message: 输入消息，支持多种格式
            segmentation_mode: 分割模式
                - 'auto': 自动判断（默认）
                - 'smart': 智能分割混合字符串  
                - 'whole': 整体处理
                - 'spaces': 按空格分割
                - 'custom': 自定义分割（需要预处理）
            
        Returns:
            消息整数列表
        """
        try:
            message_code_len = self.lm_params.get('message_len', 10)
            max_code = 2**message_code_len - 1
            
            # 处理不同类型的消息
            if isinstance(message, str):
                # 根据分割模式处理字符串
                if segmentation_mode == 'auto':
                    # 自动判断处理方式
                    if ' ' in message and len(message.split()) > 1:
                        # 包含空格：按单词分割
                        segments = message.split()
                    elif self._contains_mixed_content(message):
                        # 混合内容：智能分割
                        segments = self._smart_segment_string(message)
                    else:
                        # 简单字符串：整体处理
                        segments = [message]
                
                elif segmentation_mode == 'smart':
                    # 强制智能分割
                    segments = self._smart_segment_string(message)
                
                elif segmentation_mode == 'spaces':
                    # 按空格分割
                    segments = message.split() if ' ' in message else [message]
                
                elif segmentation_mode == 'whole':
                    # 整体处理
                    segments = [message]
                
                else:
                    # 默认情况
                    segments = [message]
                
                # 转换每个片段为整数
                messages = []
                for segment in segments[:10]:  # 限制最多10段
                    segment_hash = hash(segment) % (2**16)
                    binary_str = format(segment_hash, '016b')
                    segment_code = binary_str[:message_code_len]
                    msg_int = int(segment_code, 2)
                    messages.append(msg_int)
                
                return messages
                    
            elif isinstance(message, list):
                if all(isinstance(item, int) for item in message):
                    # 整数列表：直接使用（确保在范围内）
                    return [min(max_code, max(0, msg)) for msg in message]
                elif all(isinstance(item, str) for item in message):
                    # 字符串列表：每个转换为一段
                    messages = []
                    for item in message:
                        item_hash = hash(item) % (2**16)
                        binary_str = format(item_hash, '016b')
                        segment = binary_str[:message_code_len]
                        msg_int = int(segment, 2)
                        messages.append(msg_int)
                    return messages
                else:
                    raise ValueError("List must contain only integers or only strings")
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
                
        except Exception as e:
            logging.error(f"Failed to convert message to binary: {e}")
            # 返回基于消息字符串简单哈希的默认值
            fallback_hash = hash(str(message)) % (2**message_code_len)
            return [fallback_hash]
    
    def _contains_mixed_content(self, text: str) -> bool:
        """
        检查字符串是否包含混合内容（字母+数字）
        
        Args:
            text: 输入字符串
            
        Returns:
            是否包含混合内容
        """
        import re
        
        # 检查是否同时包含字母和数字
        has_letters = bool(re.search(r'[a-zA-Z]', text))
        has_digits = bool(re.search(r'\d', text))
        
        # 如果字符串较长且包含混合内容，则认为需要分割
        return has_letters and has_digits and len(text) > 6
    
    def _smart_segment_string(self, text: str) -> List[str]:
        """
        智能分割混合字符串
        
        将像"alibaba20250725"这样的字符串分割为有意义的片段
        
        Args:
            text: 输入字符串
            
        Returns:
            分割后的字符串列表
        """
        import re
        
        # 方案1：按字母和数字的边界分割
        segments = re.findall(r'[a-zA-Z]+|\d+', text)
        
        # 方案2：如果数字部分很长，进一步细分
        refined_segments = []
        for segment in segments:
            if segment.isdigit() and len(segment) >= 8:
                # 长数字（如日期时间戳）进一步分割
                # 例如 "20250725" -> ["2025", "0725"] 或 ["20250725"]
                if len(segment) == 8:
                    # 可能是日期格式 YYYYMMDD
                    refined_segments.extend([segment[:4], segment[4:]])
                elif len(segment) >= 10:
                    # 更长的数字，按4位分组
                    for i in range(0, len(segment), 4):
                        refined_segments.append(segment[i:i+4])
                else:
                    refined_segments.append(segment)
            elif len(segment) > 8:
                # 很长的字母串，分割为较小的片段
                for i in range(0, len(segment), 6):
                    refined_segments.append(segment[i:i+6])
            else:
                refined_segments.append(segment)
        
        # 过滤空片段并限制长度
        result = [seg for seg in refined_segments if seg and len(seg) > 0]
        return result[:10]  # 最多10段
    
    def _binary_to_message(self, binary: List[int]) -> Union[str, List[str]]:
        """
        将CredID解码的整数消息转换回原始格式
        
        Args:
            binary: CredID解码的整数列表
            
        Returns:
            转换后的消息（字符串或字符串列表）
        """
        try:
            if not binary:
                return "unknown"
            
            # 尝试匹配单段消息
            if len(binary) == 1:
                # 简单的测试消息列表
                test_messages = ["hello", "test", "AI", "tech", "demo"]
                for test_msg in test_messages:
                    test_encoded = self._message_to_binary(test_msg, 'auto')
                    if test_encoded and test_encoded[0] == binary[0]:
                        return test_msg
                return str(binary[0])
            
            # 多段消息直接返回编码表示
            return f"Multi-segment: {binary}"
                
        except Exception as e:
            logging.error(f"Failed to convert binary to message: {e}")
            return str(binary) if binary else "unknown"
    
    def _match_decoded_with_candidates(self, decoded_messages: List[int], 
                                     candidates_messages: List[str]) -> Tuple[str, float]:
        """
        将解码结果与候选消息进行智能匹配
        
        Args:
            decoded_messages: CredID解码出的消息段列表
            candidates_messages: 候选消息列表
            
        Returns:
            (匹配的消息, 匹配置信度)
        """
        best_match = None
        best_confidence = 0.0

        for candidate in candidates_messages:
            candidate_encoded = self._message_to_binary(candidate, 'smart')
            confidence = self._calculate_sequence_match(decoded_messages, candidate_encoded)
            
            if confidence > best_confidence:
                best_match = candidate
                best_confidence = confidence
        
        if best_match and best_confidence >= 0.5:  # 至少50%匹配度
            return best_match, best_confidence
        else:
            # 没有好的匹配，尝试重构
            reconstructed = self._binary_to_message(decoded_messages)
            return reconstructed, 0.0
    
    
    
    def _calculate_sequence_match(self, decoded: List[int], candidate: List[int]) -> float:
        """
        计算两个序列的匹配度
        
        Args:
            decoded: 解码出的序列
            candidate: 候选序列
            
        Returns:
            匹配度 (0.0 - 1.0)
        """
        if not decoded or not candidate:
            return 0.0
        
        # 完全匹配
        if decoded == candidate:
            return 1.0
        
        # 长度匹配 + 部分匹配
        if len(decoded) == len(candidate):
            matches = sum(1 for a, b in zip(decoded, candidate) if a == b)
            return matches / len(decoded)
        
        # 前缀匹配（处理截断情况）
        min_len = min(len(decoded), len(candidate))
        if min_len > 0:
            prefix_matches = sum(1 for a, b in zip(decoded[:min_len], candidate[:min_len]) if a == b)
            max_len = max(len(decoded), len(candidate))
            
            # 🔧 修复：前缀匹配得分 = 实际匹配数 / 最大长度 * 前缀权重
            return (prefix_matches / max_len) * 0.8
        
        return 0.0
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    def get_mode(self) -> str:
        """获取当前模式"""
        return self.mode
    
    def get_last_encoded_binary(self) -> Optional[List[int]]:
        """
        获取最后一次嵌入时编码的binary数据
        
        Returns:
            最后一次编码的binary列表，如果没有则返回None
        """
        return self.last_encoded_binary
 
    def reset(self):
        """重置处理器，清除缓存"""
        self.message_model = None
        self.wm_processor = None
        logging.info("CredIDWatermark reset completed") 