# AIGC内容标识系统设计文档

## 🎯 项目目标

构建一个功能完整的AIGC（人工智能生成内容）标识系统，为AI生成内容提供多层次的标识解决方案，满足版权保护、合规监管和内容溯源等需求：

### 📋 标识技术体系
- **隐式水印技术**：基于深度学习的不可见水印，支持版权保护和内容追踪
  - **文本隐式水印**：CredID算法的LLM文本水印（仅AI生成模式）
  - **图像隐式水印**：双后端支持（VideoSeal默认，PRC-Watermark可选）
  - **音频隐式水印**：AudioSeal算法，完整集成Bark文本转语音支持
  - **视频隐式水印**：HunyuanVideo生成 + VideoSeal水印技术栈

- **显式标识技术**：可见的合规标记，满足监管要求和用户知情权
  - **文本显式标识**：在文本中插入标准合规标识文案
  - **图像显式标识**：图像叠加可见文字标记，支持多种位置和样式
  - **音频显式标识**：语音标识插入（基于Bark TTS）
  - **视频显式标识**：视频画面可见文字标记（基于FFmpeg）

- **隐式元数据标识**：（规划中）结构化元数据嵌入，支持生成模型、时间戳、参数等信息标识

### 🌟 核心价值
- **全链路内容标识**：从生成到发布的完整标识解决方案
- **多维度合规支持**：同时满足技术防护和监管合规的双重需求
- **双模式支持**：每个模态都支持"AI生成内容"和"上传现有文件"两种处理模式
- **统一接口**：提供一致的`embed()`和`extract()`API接口，支持`operation='watermark|visible_mark'`
- **对比显示**：Web界面自动保存并显示原文件vs标识文件的并排对比
- **产业级部署**：离线优先、浏览器兼容、配置驱动，满足生产环境需求

## 📁 目录结构与层级关系（当前实现）

```
unified_watermark_tool/
├── config/
│   ├── default_config.yaml             # 统一配置文件（所有模态）
│   ├── text_config.yaml                # 文本水印专用配置
│   └── visible_mark_config.yaml        # 可见标识功能配置
├── src/
│   ├── __init__.py
│   ├── unified/                        # 统一引擎与高层门面
│   │   ├── unified_engine.py           # UnifiedWatermarkEngine（支持双模式）
│   │   └── watermark_tool.py           # 高层封装：推荐入口
│   ├── text_watermark/                 # 文本水印（CredID，仅AI生成）
│   │   ├── __init__.py
│   │   ├── credid_watermark.py         # CredID算法高级封装
│   │   └── credid/                     # CredID框架（watermarking/attacks/...）
│   ├── image_watermark/                # 图像水印（双后端+双模式）
│   │   ├── __init__.py
│   │   ├── image_watermark.py          # 统一图像接口（懒加载后端选择）
│   │   ├── prc_watermark.py            # PRC-Watermark后端
│   │   ├── PRC-Watermark/              # PRC算法实现
│   │   └── videoseal_image_watermark.py# VideoSeal图像后端（默认）
│   ├── audio_watermark/                # 音频水印（双模式支持）
│   │   ├── __init__.py
│   │   ├── audio_watermark.py          # 统一音频接口（支持TTS+上传）
│   │   ├── audioseal_wrapper.py        # AudioSeal核心封装
│   │   ├── bark_generator.py           # Bark TTS集成
│   │   ├── utils.py                    # 音频I/O、质量评估、可视化
│   │   └── audioseal/                  # AudioSeal算法实现
│   ├── video_watermark/                # 视频水印（双模式支持）
│   │   ├── __init__.py
│   │   ├── video_watermark.py          # 统一视频接口（生成+上传）
│   │   ├── hunyuan_video_generator.py  # HunyuanVideo文生视频
│   │   ├── model_manager.py            # 模型管理（离线优先）
│   │   ├── videoseal_wrapper.py        # VideoSeal水印算法
│   │   └── utils.py                    # 视频I/O、转码、性能监控
│   └── utils/                          # 通用工具和模型管理
│       ├── visible_mark.py             # 可见标识模块（合规标识添加）
├── templates/                          # Web界面模板
│   └── index.html                      # 统一Web演示界面
├── demo_outputs/                       # 演示输出目录
├── demo_uploads/                       # 演示上传目录
├── tests/                              # 测试套件
│   ├── test_unified_engine.py          # 统一引擎测试
│   ├── test_video_watermark_demo.py    # 视频端到端测试
│   └── test_audio_watermark.py         # 音频水印测试
├── app.py                              # Flask Web应用后端
├── run_tests.py                        # 统一测试运行器
├── audio_watermark_demo.py             # 音频端到端演示
└── models/                             # 本地模型缓存目录
```

层级关系（自顶向下）：
- **Web应用层**：`app.py`（Flask Web后端）+ `templates/index.html`（前端界面）
- **应用层**：`WatermarkTool`（推荐API入口，支持双模式双操作）
- **引擎层**：`UnifiedWatermarkEngine`（统一路由 text/image/audio/video，支持watermark/visible_mark操作）
- **算法层**：各模态标识技术实现（支持AI生成+文件上传，支持隐式水印+显式标识）
- **工具层**：`utils`、各模态内部的I/O、模型管理、文件转码、可见标识模块

## 🏗️ 核心架构设计

### AIGC内容标识系统架构

本系统采用**分层模块化架构**，为AIGC内容提供全方位标识解决方案：

1. **Web应用层**：
   - Flask Web应用提供REST API和文件服务
   - 统一前端界面支持双模式（AI生成/文件上传）和双操作（隐式水印/显式标识）切换
   - 实时状态反馈和多媒体播放支持，标识效果对比展示

2. **用户接口层**：
   - `WatermarkTool`提供统一的AIGC内容标识API接口
   - 支持AI生成和文件上传两种内容处理模式
   - 支持隐式水印和显式标识两种标识操作类型
   - 自动保存原文件和标识文件用于效果对比

3. **核心引擎层**：
   - `UnifiedWatermarkEngine`统一管理所有标识操作
   - 智能路由：根据模态（text/image/audio/video）和操作（watermark/visible_mark）选择合适技术
   - 懒加载和离线优先策略，双模式处理逻辑和错误恢复

4. **标识技术层**：
   - 各模态的隐式水印算法：CredID、VideoSeal、AudioSeal等深度学习技术
   - 各模态的显式标识算法：文本插入、图像叠加、语音标识、视频标记
   - 后端技术选择和参数配置管理，批处理和性能优化

5. **基础设施层**：
   - YAML配置文件管理和模型缓存离线加载
   - 多媒体文件I/O、格式转码、质量评估
   - 可见标识模块（合规标识生成）和元数据处理

### 1. 统一内容标识引擎（UnifiedWatermarkEngine）- 支持双模式双操作

位置：`src/unified/unified_engine.py`（高层封装请使用 `src/unified/watermark_tool.py`）

核心特性：
- **双模式支持**：每个模态都支持"AI生成内容"和"上传现有文件"两种处理模式
- **双操作支持**：支持隐式水印（`operation='watermark'`）和显式标识（`operation='visible_mark'`）两种标识操作
- **统一接口**：`embed(content, message, modality, operation='watermark|visible_mark', **kwargs)`和`extract(content, modality, operation='watermark|visible_mark', **kwargs)`四模态统一接口
- **智能路由**：根据模态和操作类型自动选择最适合的标识技术
- **原文件保存**：AI生成模式和文件上传模式都自动保存原文件和标识文件用于Web对比显示
- **技术选择**：`text=credid`（仅AI生成），`image=videoseal`，`audio=audioseal`，`video=hunyuan+videoseal`
- **离线优先**：文本/图像/视频模型优先从本地缓存加载，避免网络依赖
- **配置驱动**：读取`config/default_config.yaml`和`config/text_config.yaml`，支持运行时参数调整

### 🎯 AIGC内容标识接口使用示例（推荐通过 `WatermarkTool`）:

```python
from src.unified.watermark_tool import WatermarkTool

tool = WatermarkTool()

# ===== AIGC内容隐式水印标识（operation='watermark'）=====
# 适用于版权保护、内容追踪、技术防护等场景

# 文本水印标识（仅支持AI生成模式）
watermarked_text = tool.embed("这是测试文本", "wm_msg", 'text')
text_result = tool.extract(watermarked_text, 'text')
# 返回: {'detected': True, 'message': 'wm_msg', 'confidence': 0.95}

# 图像AI生成 + 隐式水印标识（VideoSeal默认后端）
img = tool.embed("a cat under the sun", "hello_vs", 'image')
img_res = tool.extract(img, 'image', replicate=16, chunk_size=16)
# 后端自动保存original_image.png和watermarked_image.png用于效果对比

# 音频AI生成（Bark TTS）+ 隐式水印标识
audio_out = tool.embed("Hello world", "hello_audio", 'audio',
                      output_path="outputs/audio/generated.wav")
audio_res = tool.extract(audio_out, 'audio')
# 后端自动保存original_audio.wav和watermarked_audio.wav

# 视频AI生成（HunyuanVideo）+ 隐式水印标识
video_path = tool.embed("阳光洒在海面上", "video_wm", 'video',
                       num_frames=49, height=720, width=1280)
video_res = tool.extract(video_path, 'video')
# 后端自动保存original_video.mp4和watermarked_video.mp4

# ===== 上传文件隐式水印标识模式 =====
# 对已有AIGC内容进行后处理标识

# 图像文件隐式水印标识
img_wm = tool.embed("watermark message", "file_msg", 'image',
                    image_input="/path/to/image.jpg")
img_file_res = tool.extract(img_wm, 'image')

# 音频文件隐式水印标识
audio_wm = tool.embed("audio watermark", "audio_msg", 'audio',
                     audio_input="/path/to/audio.wav",
                     output_path="outputs/watermarked_audio.wav")
audio_file_res = tool.extract(audio_wm, 'audio')

# 视频文件隐式水印标识（自动转码为浏览器兼容格式）
video_wm = tool.embed("video watermark", "video_msg", 'video',
                     video_input="/path/to/video.mp4")
video_file_res = tool.extract(video_wm, 'video')

# ===== AIGC内容显式标识操作（operation='visible_mark'）=====
# 适用于监管合规、用户告知、内容标识等场景

# 文本显式标识
original_text = "这是一段原始文本内容。"
marked_text = tool.embed(original_text, "本内容由人工智能生成/合成", 'text',
                        operation='visible_mark', position='start')
text_mark_res = tool.extract(marked_text, 'text', operation='visible_mark')

# 图像显式标识（传入图像文件路径）
marked_img = tool.embed("/path/to/image.jpg", "测试标识", 'image',
                       operation='visible_mark',
                       position='bottom_right', font_percent=5.0, font_color='#FF0000')
img_mark_res = tool.extract(marked_img, 'image', operation='visible_mark')

# 音频显式标识（需要Bark TTS）
marked_audio = tool.embed("/path/to/audio.wav", "本内容由人工智能生成", 'audio',
                         operation='visible_mark',
                         position='start', voice_preset='v2/zh_speaker_6')
audio_mark_res = tool.extract(marked_audio, 'audio', operation='visible_mark')

# 视频显式标识
marked_video = tool.embed("/path/to/video.mp4", "本内容由人工智能生成", 'video',
                         operation='visible_mark',
                         position='bottom_right', font_percent=4.0, duration_seconds=3.0)
video_mark_res = tool.extract(marked_video, 'video', operation='visible_mark')

# ===== AIGC内容标识便捷接口 =====

# 一键添加显式标识（合规标记）
marked_content = tool.add_visible_mark(
    content="原始内容",  # 文本字符串或文件路径
    message="本内容由人工智能生成/合成",  # 标准合规文案
    modality='text',  # 或 'image', 'audio', 'video'
    position='end'
)

# 检测显式标识
detection_result = tool.detect_visible_mark(
    content=marked_content,
    modality='text'
)
# 返回: {'detected': True, 'message': '本内容由人工智能生成/合成', 'confidence': 1.0}

# ===== 信息查询接口 =====

# 查询支持的操作类型
supported_ops = tool.get_supported_operations()
# 返回: ['watermark', 'visible_mark']

# 查询操作详细信息
op_info = tool.get_operation_info()
# 返回操作类型的详细描述和支持的模态
```

### 📋 AIGC内容标识接口参数与返回值：

#### 文本内容标识（隐式水印+显式标识）
- **隐式水印**: `embed(prompt, message, 'text')` → `str` （仅AI生成模式，适用于版权保护）
- **显式标识**: `embed(original_text, mark_text, 'text', operation='visible_mark', position='start|end')` → `str`
- **技术原理**: 隐式水印基于LLM的统计特征嵌入；显式标识通过文本插入实现合规标记
- **返回内容**: 带标识的文本字符串，支持后续检测和验证
- **提取接口**: `extract(text, 'text', operation='watermark|visible_mark')` → `{detected: bool, message: str, confidence: float}`

#### 图像内容标识（隐式水印+显式标识）
- **隐式水印**: `embed(prompt, message, 'image', **kwargs)` → `PIL.Image` （AI生成模式，适用于版权保护）或 `embed("content", message, 'image', image_input='/path', **kwargs)` （上传模式，后处理标识）
- **显式标识**: `embed('/path/to/image.jpg', mark_text, 'image', operation='visible_mark', position='bottom_right', font_percent=5.0, font_color='#FFFFFF')` → `PIL.Image`
- **技术架构**: VideoSeal（默认，基于深度学习）或PRC-Watermark（可选，基于扩散模型）
- **标识展示**: 自动保存原图和标识图文件，Web界面提供before/after效果对比
- **检测增强**: 支持`replicate`和`chunk_size`参数提升检测置信度

#### 音频内容标识（隐式水印+显式标识）
- **隐式水印**: `embed(tts_prompt, message, 'audio', output_path=None)` → `torch.Tensor | str` （AI生成模式，适用于版权保护）或 `embed("content", message, 'audio', audio_input='/path', **kwargs)` （上传模式，后处理标识）
- **显式标识**: `embed('/path/to/audio.wav', mark_text, 'audio', operation='visible_mark', position='start', voice_preset='v2/zh_speaker_6')` → `torch.Tensor | str`
- **技术集成**: Bark TTS多语言语音生成 + AudioSeal深度学习音频水印
- **格式兼容**: 支持WAV, MP3, FLAC, AAC, M4A等主流音频格式
- **标识展示**: 自动保存原音频和标识音频，Web界面支持并排播放对比

#### 视频内容标识（隐式水印+显式标识）
- **隐式水印**: `embed(prompt, message, 'video', num_frames=49, height=720, width=1280)` → `str` （AI生成模式，适用于版权保护）或 `embed("content", message, 'video', video_input='/path', **kwargs)` （上传模式，后处理标识）
- **显式标识**: `embed('/path/to/video.mp4', mark_text, 'video', operation='visible_mark', position='bottom_right', font_percent=4.0, duration_seconds=2.0)` → `str`
- **技术架构**: HunyuanVideo文生视频技术 + VideoSeal深度学习水印算法
- **浏览器优化**: 自动转码为H.264+AAC+faststart格式，确保跨平台Web播放兼容性
- **标识展示**: 自动保存原视频和标识视频，Web界面支持并排播放效果对比

#### 统一提取接口
- **所有模态**: `extract(content, modality, operation='watermark|visible_mark', **kwargs)` → `{detected: bool, message: str, confidence: float, metadata: dict}`
- **增强参数**: 图像/视频支持`replicate`和`chunk_size`，视频支持`max_frames`限制
- **操作类型**: `operation='watermark'`检测隐式水印，`operation='visible_mark'`检测显式标识

### 🔧 离线/缓存配置建议：
- **环境变量**: 设置`TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`强制离线加载
- **模型缓存**: 将模型放在`models/`目录或通过`HF_HOME`/`HF_HUB_CACHE`指向本地缓存
- **文本模型**: 默认读取`config/text_config.yaml`的`model_name`，缓存未命中时回退`sshleifer/tiny-gpt2`
- **视频模型**: HunyuanVideo使用本地快照，避免网络下载不确定性

### 2. 文本水印模块 (CredID Algorithm) ✅ **已实现**

**CredID算法原理**：
- **多位水印**：支持嵌入多段信息（如用户ID、时间戳、版本号等）
- **logits处理**：在语言模型的logits输出上进行修改，影响token选择概率
- **双模式支持**：LM模式（高质量
- **候选优化**：支持候选消息列表的限制搜索，提升检测效率
- **智能分割**：自动处理复杂消息格式（如"log20250725143000"）

**实际实现的核心架构**：

```python
# src/text_watermark/credid_watermark.py
import torch
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer, LogitsProcessorList

class CredIDWatermark:
    """
    CredID文本水印算法统一封装
    
    ✨ 核心功能特点:
    1. 支持多种消息格式 (字符串、整数列表、字符串列表)
    2. 双模式运行: LM模式(高质量)
    3. 智能多段消息处理和自动分割
    4. 候选消息优化搜索机制
    5. 完整的错误处理和置信度评估
    6. 简化的代码结构，去除复杂的按位置分组逻辑
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化CredID水印处理器
        
        Args:
            config: 配置字典，必须包含:
                - mode: 'lm' (默认'lm')
                - model_name: 预训练模型名称
                - lm_params: LM模式参数字典
                - wm_params: 水印处理参数字典
                - 其他生成参数 (max_new_tokens, num_beams等)
        """
        self.config = config
        self.mode = config.get('mode', 'lm')  # 默认LM模式
        self.model_name = config.get('model_name', 'huggyllama/llama-7b')
        
        # 算法核心参数
        self.lm_params = config.get('lm_params', {})
        self.wm_params = config.get('wm_params', {})
        
        # 延迟初始化的组件
        self.message_model = None
        self.tokenizer_ref = None
        
        logging.info(f"CredID初始化: 模式={self.mode}, 模型={self.model_name}")
```

**🔹 核心接口 1: embed() - 水印嵌入**

```python
    def embed(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
              prompt: str, message: Union[str, List[int], List[str]], 
              segmentation_mode: str = 'auto') -> Dict[str, Any]:
        """
        🎯 核心功能: 在文本生成过程中嵌入水印
        
        📋 详细工作流程:
        1. 设置处理器 (如果还没设置)
        2. 将消息转换为CredID兼容的二进制格式 (支持多段)
        3. 创建包含水印处理器的LogitsProcessorList
        4. 使用model.generate()生成带水印文本
        5. 返回完整结果和详细元数据
        
        📥 参数说明:
            model: HuggingFace预训练语言模型 (如Llama, GPT等)
            tokenizer: 对应的分词器，必须设置pad_token
            prompt: 输入提示文本，如 "Hello, today is"
            message: 水印信息，支持多种格式:
                - str: "hello" 或复杂字符串 "log20250725143000"
                - List[int]: [123, 456, 789] 
                - List[str]: ["user", "2025", "admin"]
            segmentation_mode: 消息分割模式
                - 'auto': 自动判断最佳分割方式 (推荐)
                - 'smart': 智能分割，如 "alibaba20250725" → ["alibaba", "2025", "0725"]
                - 'whole': 整体处理
                - 'spaces': 按空格分割
                
        📤 返回值结构:
            {
                'watermarked_text': str,      # 🎯 带水印的生成文本
                'original_message': Any,      # 原始水印信息
                'binary_message': List[int],  # 转换后的二进制消息序列
                'prompt': str,                # 输入提示
                'success': bool,              # ✅/❌ 是否成功
                'metadata': {                 # 详细元数据
                    'mode': str,              # 使用的模式 ('lm')
                    'model_name': str,        # 模型名称
                    'input_length': int,      # 输入token长度
                    'output_length': int,     # 输出token长度
                    'generation_config': dict,# 生成配置参数
                    'num_message_segments': int # 消息段数
                }
            }
            
        🚨 错误情况返回:
            {
                'watermarked_text': None,
                'success': False,
                'error': str                  # 错误信息
            }
        """
```

**🔹 核心接口 2: extract() - 水印提取**

```python
    def extract(self, watermarked_text: str, 
                model: Optional[PreTrainedModel] = None,
                tokenizer: Optional[PreTrainedTokenizer] = None,
                candidates_messages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        🎯 核心功能: 从水印文本中提取水印信息
        
        📋 详细工作流程:
        1. 检查模式和参数有效性 (LM模式需要model和tokenizer)
        2. 候选消息处理: 收集所有候选消息的所有编码段 (简化策略)
        3. 使用CredID解码器进行统计检测
        4. 智能匹配: 将解码结果与候选消息进行序列匹配
        5. 置信度计算和结果验证
        
        📥 参数说明:
            watermarked_text: 可能包含水印的文本
            model: 语言模型 (LM模式必需)
            tokenizer: 分词器 (LM模式必需)
            candidates_messages: 候选消息列表，用于优化搜索
                🎯 推荐使用: 可大幅提升检测精度和效率
                例如: ["log20250725143000", "user987654321", "admin2025"]
                
        📤 返回值结构:
            {
                'extracted_message': str,           # 🎯 提取的消息
                'binary_message': List[int],        # 解码的二进制消息序列
                'confidence': float,                # 🎚️ 置信度 (0.0-1.0)
                'success': bool,                    # ✅/❌ 是否成功提取
                'detailed_confidence': List,       # 详细置信度信息
                'metadata': {
                    'mode': str,                    # 检测模式
                    'text_length': int,             # 文本长度
                    'num_decoded_segments': int,    # 解码段数
                    'detection_method': 'CredID',   # 检测方法
                    'confidence_threshold': float,  # 置信度阈值
                    'search_space': int,            # 搜索空间大小
                    'candidates_provided': bool     # 是否提供候选消息
                }
            }
            
        🚨 失败情况返回:
            {
                'extracted_message': None,
                'confidence': 0.0,
                'success': False,
                'error': str                        # 错误或"No watermark detected"
            }
        """
```

**🔧 核心内部方法**

```python
    # === 消息处理方法 ===
    def _message_to_binary(self, message: Union[str, List[int], List[str]], 
                          segmentation_mode: str = 'auto') -> List[int]:
        """将多种格式的消息转换为CredID兼容的整数序列"""
        
    def _binary_to_message(self, binary: List[int]) -> Union[str, List[str]]:
        """将解码的整数序列转换回原始消息格式"""
        
    # === 智能匹配方法 ===  
    def _match_decoded_with_candidates(self, decoded_messages: List[int], 
                                     candidates_messages: List[str]) -> Tuple[str, float]:
        """将解码结果与候选消息进行智能匹配 (简化版本)"""
        
    def _calculate_sequence_match(self, decoded: List[int], candidate: List[int]) -> float:
        """计算两个序列的匹配度分数"""
        
    # === 字符串分割方法 ===
    def _smart_segment_string(self, text: str) -> List[str]:
        """智能分割字符串，支持复杂格式如'log20250725143000'"""
```

**⚙️ 配置参数详解**

```yaml
# config/text_config.yaml - 完整配置示例
method: "CredID"
model_name: "huggyllama/llama-7b"          
mode: "lm"                                 # 'lm'(高质量) /
device: "auto"                             

# === 生成参数 ===
max_new_tokens: 110                        
num_beams: 4                               
do_sample: true                            
temperature: 0.7                           
top_p: 0.9                                
top_k: 50                                 

# === CredID LM模式核心参数 ===
lm_params:
  delta: 1.5                              # logits修改强度 (关键参数)
  prefix_len: 10                          # 前缀保护长度
  message_len: 10                         # 每段消息的二进制长度
  seed: 42                                # 随机种子
  topk: -1                               # LM top-k限制
  permutation_num: 50                     # 随机排列数
  hash_prefix_len: 1                      # 哈希前缀长度
  shifts: [21, 24, 3, 8, 14,2, 4, 28, 31, 3, 8, 14, 2, 4, 28, 16, 7, 19, 25, 11, 33, 1, 0, 8, 34]

# === 水印处理参数 ===
wm_params:
  encode_ratio: 4                         # 编码比率 (每消息位对应的token数)
  seed: 42                                
  strategy: "vanilla"                     # 'vanilla'/'max_confidence'
  max_confidence: 0.5                     
  top_k: 1000                            

# === 解码配置 ===
decode_batch_size: 16                      
disable_tqdm: false                        
confidence_threshold: 0.6                  # 成功检测的置信度阈值
```

**🚀 实际使用示例和最佳实践**

```python
# === 完整使用示例 ===
from src.text_watermark.credid_watermark import CredIDWatermark
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

# 1. 初始化系统
with open('config/text_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

watermark = CredIDWatermark(config)

# 2. 🎯 单一消息处理
result = watermark.embed(model, tokenizer, "Hello, today is", "tech")
if result['success']:
    print(f"✅ 生成文本: {result['watermarked_text']}")
    
    # 基础提取
    extracted = watermark.extract(result['watermarked_text'], model, tokenizer)
    print(f"📤 提取结果: {extracted['extracted_message']} (置信度: {extracted['confidence']:.3f})")

# 3. 🎯 复杂消息处理
complex_messages = [
    ("系统日志", "log20250725143000"),
    ("用户信息", "alibaba20250725"),
    ("管理账户", ["admin", "2025", "secure"])
]

for desc, message in complex_messages:
    result = watermark.embed(model, tokenizer, f"Entry: ", message)
    if result['success']:
        print(f"\n=== {desc} ===")
        print(f"消息: {message}")
        print(f"生成: {result['watermarked_text']}")
        
        # 🎯 候选优化提取
        candidates = ["log20250725143000", "alibaba20250725", "admin2025secure", "tech", "hello"]
        extracted = watermark.extract(
            result['watermarked_text'], 
            model, tokenizer, 
            candidates_messages=candidates
        )
        
        success_icon = "✅" if extracted['success'] else "❌"
        print(f"{success_icon} 提取: {extracted['extracted_message']} (置信度: {extracted['confidence']:.3f})")

# 4. 🎯 批量处理性能测试
import time

test_messages = ["hello", "tech2025", "user123", "log20250725143000"]
batch_start = time.time()

batch_results = []
for i, msg in enumerate(test_messages):
    embed_result = watermark.embed(model, tokenizer, f"Test {i}: ", msg)
    if embed_result['success']:
        extract_result = watermark.extract(embed_result['watermarked_text'], model, tokenizer)
        batch_results.append({
            'original': msg,
            'extracted': extract_result['extracted_message'],
            'confidence': extract_result['confidence'],
            'success': extract_result['success']
        })

batch_time = time.time() - batch_start
print(f"\n⏱️ 批量处理({len(test_messages)}条): {batch_time:.2f}秒")

# 5. 🎯 错误处理示例
try:
    # 模拟错误情况
    error_result = watermark.extract("This text has no watermark", model, tokenizer)
    if not error_result['success']:
        print(f"❌ 检测失败: {error_result.get('error', 'No watermark detected')}")
except Exception as e:
    print(f"🚨 异常处理: {e}")
```

**📊 性能和特点总结**

| 特性 | 描述 | 优势 |
|------|------|------|
| **多消息格式** | 支持字符串、列表、复杂格式 | 灵活性高，适应不同场景 |
| **候选优化** | 限制搜索空间提升效率 | 大幅提升检测精度 |
| **智能分割** | 自动处理复杂消息格式 | 无需手动预处理 |
| **简化架构** | 去除复杂的按位置分组逻辑 | 代码更清晰，维护性好 |
| **错误处理** | 完整的异常处理机制 | 生产环境可靠性高 |
| **性能监控** | 内置时间和资源使用统计 | 便于性能调优 |


## 🆕 2025-08 更新摘要（diffusers==0.34 兼容 + VideoSeal 图像后端）

### 动机
- 为兼容新的视频模型（Hunyuan），环境升级至 `diffusers==0.34`。该版本对自定义管线/模块注册有变更，旧 PRC 路径易受影响。因此新增 VideoSeal 作为图像水印的第二后端，并将相关加载改造为"懒加载 + 离线优先"。

### 主要改动
- 图像水印新增后端：`videoseal`
  - 新文件 `src/image_watermark/videoseal_image_watermark.py`：将单图当作单帧视频，复用 `src/video_watermark/videoseal_wrapper.py` 的 `embed/detect`，对图像提供无 Diffusers 依赖的稳健嵌入/提取。
  - `src/image_watermark/image_watermark.py`：
    - 懒加载具体算法处理器，避免在构造阶段加载无关依赖。
    - 支持 `algorithm: videoseal`，并在无图像输入时，先用 Stable Diffusion 生成，再调用 VideoSeal 嵌入。
  - `src/unified/watermark_tool.py`：`get_supported_algorithms()['image']` 增加 `videoseal`。
  - 检测增强：`extract(..., replicate=N, chunk_size=N)` 支持将单帧复制为多帧做均值，显著提升读出稳定性与置信度。

- 离线加载（Stable Diffusion）
  - `src/utils/model_manager.py`：
    - 强制 `TRANSFORMERS_OFFLINE/DIFFUSERS_OFFLINE/HF_HUB_OFFLINE`。
    - 解析/优先返回 HF Hub 本地缓存目录 `.../hub/models--stabilityai--stable-diffusion-2-1-base`，与 PRC 路径一致；`from_pretrained(local_files_only=True)` 离线解析 refs。

- 文本水印（CredID）离线加载
  - `test_complex_messages_real.py`：
    - 强制离线变量。
    - `AutoTokenizer/AutoModelForCausalLM.from_pretrained(..., local_files_only=True, cache_dir=...)`。
    - 自动探测缓存目录或通过配置 `hf_cache_dir` 指定。

- 导入与测试
  - 统一 `src.*` 绝对导入风格，脚本从项目根运行稳定。
  - `tests/conftest.py` 将 `src/` 注入 `sys.path`，测试时 `unified.*` 可导入。
  - 新增：
    - `tests/test_image_videoseal.py`（最小验证）
    - 根级 `test_image_videoseal_root.py`：可直接 `python` 演示
      - `--mode pil`：现有图像嵌入/提取
      - `--mode gen`：生成→嵌入→提取（完全离线，需本地 SD 权重）

### 使用与调参建议（VideoSeal 图像水印）
- 配置（示例）：
```yaml
image_watermark:
  algorithm: videoseal
  model_name: stabilityai/stable-diffusion-2-1-base
  resolution: 512
  num_inference_steps: 30
  lowres_attenuation: true
  device: cuda
```
- 生成 → 嵌入 → 提取：
```python
from src.unified.watermark_tool import WatermarkTool
tool = WatermarkTool()
tool.set_algorithm('image', 'videoseal')
img = tool.generate_image_with_watermark(prompt='a cat', message='hello_videoseal')
res = tool.extract_image_watermark(img, replicate=16, chunk_size=16)
```
- CLI 演示：
```bash
python test_image_videoseal_root.py --mode pil  --device cuda
python test_image_videoseal_root.py --mode gen  --device cuda --resolution 512 --steps 30
```

### 提升检测置信度
- 生成侧：提高 `resolution`/`num_inference_steps`；简化 prompt；使用 GPU。
- 检测侧：`replicate` 设为 8~32，并与 `chunk_size` 对齐，使用多帧均值；对单图尤其有效。

### 4. 音频水印模块 (AudioSeal Algorithm) ✅ **已完成实现**

**AudioSeal算法原理与实现状态**：
- **Meta AudioSeal算法**：基于深度学习的鲁棒音频水印技术，完整Python封装，生产环境就绪
- **16位消息编码系统**：使用SHA256哈希确保编码一致性，支持字符串到二进制的可靠转换  
- **高保真嵌入**：SNR>40dB（实测44.45dB），听觉质量几乎无损失，100%检测成功率
- **设备自适应优化**：支持CPU/CUDA自动切换和设备张量管理，修复设备不匹配问题
- **高效批处理**：3个音频2.8秒，并行处理优化，支持大规模应用

**已实现的核心架构与性能**：

```python
# src/audio_watermark/audio_watermark.py - 完整实现
import torch
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

class AudioWatermark:
    """
    AudioSeal音频水印算法统一封装 - 生产环境就绪
    
    ✅ 已完成核心功能:
    1. Meta AudioSeal完整集成 - 100%检测成功率，SNR 44.45dB
    2. Bark TTS端到端流程 - 支持多语言（中英文）高质量语音生成
    3. 多格式音频支持 - WAV/MP3/FLAC等，完整I/O处理
    4. 设备自适应优化 - CPU/CUDA自动切换，内存优化，设备一致性修复  
    5. 高效批处理 - 3个音频2.8秒，并行处理优化
    6. 完整质量评估 - SNR/MSE/相关性指标，噪声鲁棒性测试
    7. 技术问题修复 - 3D张量维度处理，设备匹配，Bark导入检测
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化AudioSeal音频水印处理器
        
        Args:
            config: 配置字典，包含:
                - algorithm: 'audioseal' (默认)
                - device: 'cuda', 'cpu', 或 'auto'
                - nbits: 消息位数 (默认16)
                - sample_rate: 采样率 (默认16000)
                - bark_config: Bark TTS配置
        """
        self.config = config
        self.algorithm = config.get('algorithm', 'audioseal')
        self.device = config.get('device', 'auto')
        self.nbits = config.get('nbits', 16)
        self.sample_rate = config.get('sample_rate', 16000)
        
        # 延迟初始化的组件
        self.audioseal_wrapper = None
        self.bark_generator = None
        
        logging.info(f"AudioWatermark初始化: 算法={self.algorithm}, 设备={self.device}")
```

**🔹 核心接口 1: embed_watermark() - 音频水印嵌入**

```python
    def embed_watermark(self, 
                       audio: Union[str, torch.Tensor, Path], 
                       message: str,
                       input_sample_rate: Optional[int] = None,
                       alpha: float = 1.0,
                       output_path: Optional[str] = None) -> Union[torch.Tensor, str]:
        """
        🎯 核心功能: 在音频中嵌入AudioSeal水印
        
        📋 详细工作流程:
        1. 音频加载和预处理 (重采样到16kHz，格式转换)
        2. 消息编码为16位二进制序列 (SHA256哈希)
        3. 使用AudioSeal生成器进行水印嵌入
        4. 后处理和输出 (保存文件或返回张量)
        
        📥 参数说明:
            audio: 输入音频，支持多种格式:
                - str/Path: 音频文件路径 (WAV, MP3, FLAC等)
                - torch.Tensor: 音频张量 (1, samples) 或 (samples,)
            message: 要嵌入的字符串消息，如 "user123", "2025_watermark"
            input_sample_rate: 输入音频采样率 (从文件推断或手动指定)
            alpha: 水印强度 (0.0-2.0，默认1.0，越高水印越强但失真越大)
            output_path: 输出文件路径 (可选，提供则保存文件)
            
        📤 返回值:
            - 如果提供output_path: 返回保存的文件路径(str)
            - 否则: 返回带水印的音频张量(torch.Tensor)
            
        🚨 错误情况:
            抛出RuntimeError异常，包含详细错误信息
        """
        self._ensure_audioseal()
        
        # 处理不同输入格式
        if isinstance(audio, (str, Path)):
            from .utils import AudioIOUtils
            audio_tensor, sr = AudioIOUtils.load_audio(
                str(audio), 
                target_sample_rate=self.sample_rate
            )
        else:
            audio_tensor = audio
            sr = input_sample_rate or self.sample_rate
        
        # 嵌入水印
        watermarked = self.audioseal_wrapper.embed(
            audio_tensor, message, sr, alpha
        )
        
        if output_path:
            from .utils import AudioIOUtils
            AudioIOUtils.save_audio(watermarked, output_path, self.sample_rate)
            return output_path
        else:
            return watermarked
```

**🔹 核心接口 2: extract_watermark() - 音频水印提取**

```python
    def extract_watermark(self, 
                         watermarked_audio: Union[str, torch.Tensor, Path],
                         input_sample_rate: Optional[int] = None,
                         detection_threshold: float = 0.5,
                         message_threshold: float = 0.5) -> Dict[str, Any]:
        """
        🎯 核心功能: 从音频中提取AudioSeal水印信息
        
        📋 详细工作流程:
        1. 音频加载和预处理
        2. 使用AudioSeal检测器进行水印检测
        3. 消息解码和匹配 (与历史消息库匹配)
        4. 置信度计算和结果验证
        
        📥 参数说明:
            watermarked_audio: 可能包含水印的音频
            input_sample_rate: 输入音频采样率
            detection_threshold: 检测阈值 (0.0-1.0，默认0.5)
            message_threshold: 消息解码阈值 (0.0-1.0，默认0.5)
            
        📤 返回值结构:
            {
                'detected': bool,               # 🎯 是否检测到水印
                'message': str,                 # 📤 解码的消息 (检测成功时)
                'confidence': float,            # 🎚️ 检测置信度 (0.0-1.0)
                'raw_bits': torch.Tensor,      # 原始二进制解码结果
                'processing_time': float,       # 处理耗时 (秒)
                'metadata': {                   # 详细元数据
                    'algorithm': 'audioseal',   # 算法名称
                    'sample_rate': int,         # 采样率
                    'audio_length': float,      # 音频时长
                    'detection_threshold': float,
                    'message_threshold': float
                }
            }
            
        🚨 失败情况返回:
            {
                'detected': False,
                'message': '',
                'confidence': 0.0,
                'error': str                    # 错误信息
            }
        """
        self._ensure_audioseal()
        
        # 处理输入音频
        if isinstance(watermarked_audio, (str, Path)):
            from .utils import AudioIOUtils
            audio_tensor, sr = AudioIOUtils.load_audio(
                str(watermarked_audio), 
                target_sample_rate=self.sample_rate
            )
        else:
            audio_tensor = watermarked_audio
            sr = input_sample_rate or self.sample_rate
        
        # 提取水印
        result = self.audioseal_wrapper.extract(
            audio_tensor, sr, detection_threshold, message_threshold
        )
        
        return result
```

**🔹 高级接口: generate_audio_with_watermark() - 文本转语音+水印**

```python
    def generate_audio_with_watermark(self,
                                     prompt: str,
                                     message: str,
                                     voice_preset: Optional[str] = None,
                                     temperature: float = 0.8,
                                     seed: Optional[int] = None,
                                     alpha: float = 1.0,
                                     output_path: Optional[str] = None) -> Union[torch.Tensor, str]:
        """
        🎯 高级功能: 文本转语音并嵌入水印 (需要Bark)
        
        📋 详细工作流程:
        1. 使用Bark TTS生成高质量语音
        2. 自动嵌入AudioSeal水印
        3. 返回带水印的语音音频
        
        📥 参数说明:
            prompt: 要转换的文本，如 "Hello, this is a test message"
            message: 要嵌入的水印信息
            voice_preset: 语音预设，如 "v2/en_speaker_6", "v2/zh_speaker_0"
            temperature: 生成温度 (0.0-1.0，控制随机性)
            seed: 随机种子 (可重现生成)
            alpha: 水印强度
            output_path: 输出文件路径 (可选)
            
        📤 返回值:
            - 如果提供output_path: 返回保存的文件路径
            - 否则: 返回带水印的音频张量
            
        🚨 依赖要求:
            需要安装Bark: pip install git+https://github.com/suno-ai/bark.git
        """
        self._ensure_bark()
        
        # 使用Bark生成语音
        generated_audio = self.bark_generator.generate_audio(
            prompt, voice_preset, temperature, seed
        )
        
        # 嵌入水印
        watermarked_audio = self.audioseal_wrapper.embed(
            generated_audio, message, self.sample_rate, alpha
        )
        
        if output_path:
            from .utils import AudioIOUtils
            AudioIOUtils.save_audio(watermarked_audio, output_path, self.sample_rate)
            return output_path
        else:
            return watermarked_audio
```

**🔧 核心内部方法**

```python
    # === 质量评估方法 ===
    def evaluate_quality(self, original: torch.Tensor, 
                        watermarked: torch.Tensor) -> Dict[str, float]:
        """计算音频质量指标 (SNR, MSE, 相关性)"""
        
    def batch_embed(self, audios: List, messages: List[str]) -> List:
        """批量音频水印嵌入"""
        
    def batch_extract(self, watermarked_audios: List) -> List[Dict]:
        """批量音频水印提取"""
        
    # === 组件初始化方法 ===
    def _ensure_audioseal(self):
        """确保AudioSeal封装器已初始化"""
        
    def _ensure_bark(self):
        """确保Bark生成器已初始化 (如果需要TTS功能)"""
```

**⚙️ 配置参数详解**

```yaml
# config/audio_config.yaml - 完整配置示例
algorithm: "audioseal"
device: "auto"                          # 'cuda', 'cpu', 'auto'
nbits: 16                              # 消息位数
sample_rate: 16000                     # 采样率 (AudioSeal要求16kHz)

# === AudioSeal参数 ===
audioseal_params:
  detection_threshold: 0.5             # 检测阈值
  message_threshold: 0.5               # 消息解码阈值
  alpha: 1.0                          # 默认水印强度

# === Bark TTS配置 ===
bark_config:
  model_size: "large"                  # 'small', 'large'
  use_gpu: true                        # 是否使用GPU
  temperature: 0.8                     # 生成温度
  default_voice: "v2/en_speaker_6"     # 默认语音预设
  target_sample_rate: 16000            # 目标采样率

# === 音频处理参数 ===
audio_params:
  supported_formats: [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
  normalize_audio: true                # 是否归一化音频
  quality_check: true                  # 是否进行质量检查
```

**🚀 实际使用示例和最佳实践**

```python
# === 完整使用示例 ===
from src.audio_watermark import create_audio_watermark
import torch
import time

# 1. 初始化系统
watermark_tool = create_audio_watermark()

# 2. 🎯 基础音频水印流程
print("=== 基础音频水印测试 ===")

# 创建测试音频 (1秒正弦波)
sample_rate = 16000
test_audio = 0.5 * torch.sin(2 * 3.14159 * 440 * torch.linspace(0, 1, sample_rate))
test_audio = test_audio.unsqueeze(0)  # 添加通道维度
test_message = "hello_audioseal_2025"

print(f"测试音频形状: {test_audio.shape}")
print(f"测试消息: '{test_message}'")

# 嵌入水印
start_time = time.time()
watermarked_audio = watermark_tool.embed_watermark(test_audio, test_message)
embed_time = time.time() - start_time

print(f"✅ 嵌入完成: {embed_time:.3f}秒")
print(f"水印音频形状: {watermarked_audio.shape}")

# 提取水印
start_time = time.time()
result = watermark_tool.extract_watermark(watermarked_audio)
extract_time = time.time() - start_time

print(f"✅ 提取完成: {extract_time:.3f}秒")
print(f"检测结果: {result['detected']}")
print(f"解码消息: '{result['message']}'")
print(f"置信度: {result['confidence']:.3f}")

# 质量评估
quality = watermark_tool.evaluate_quality(test_audio, watermarked_audio)
print(f"🎵 音频质量:")
print(f"  SNR: {quality['snr_db']:.2f} dB")
print(f"  相关性: {quality['correlation']:.3f}")

# 3. 🎯 文件I/O处理
print("\n=== 文件I/O测试 ===")

# 保存原始音频
from src.audio_watermark.utils import AudioIOUtils
AudioIOUtils.save_audio(test_audio, "test_original.wav", sample_rate)

# 从文件嵌入水印
watermarked_path = watermark_tool.embed_watermark(
    "test_original.wav", 
    test_message,
    output_path="test_watermarked.wav"
)
print(f"💾 水印音频已保存: {watermarked_path}")

# 从文件提取水印
file_result = watermark_tool.extract_watermark("test_watermarked.wav")
print(f"📁 文件检测: {'✅' if file_result['detected'] else '❌'}")
print(f"📁 文件消息: '{file_result['message']}'")

# 4. 🎯 Bark TTS + 水印 (需要安装Bark)
print("\n=== 文本转语音+水印测试 ===")
try:
    tts_text = "Hello, this is a test of text to speech with watermark."
    tts_message = "bark_tts_demo"
    
    # 生成带水印的语音
    generated_audio = watermark_tool.generate_audio_with_watermark(
        prompt=tts_text,
        message=tts_message,
        voice_preset="v2/en_speaker_6",
        temperature=0.7,
        seed=42,
        output_path="test_tts_watermarked.wav"
    )
    
    print(f"🎤 TTS音频已生成: {generated_audio}")
    
    # 验证TTS音频中的水印
    tts_result = watermark_tool.extract_watermark(generated_audio)
    print(f"🎤 TTS检测: {'✅' if tts_result['detected'] else '❌'}")
    print(f"🎤 TTS消息: '{tts_result['message']}'")
    
except Exception as e:
    print(f"⚠️ TTS功能不可用: {e}")
    print("请安装Bark: pip install git+https://github.com/suno-ai/bark.git")

# 5. 🎯 批量处理测试
print("\n=== 批量处理测试 ===")
test_messages = ["batch_01", "batch_02", "batch_03"]
test_audios = []

# 生成测试音频
for i, msg in enumerate(test_messages):
    # 不同频率的正弦波
    freq = 440 + i * 100  # 440Hz, 540Hz, 640Hz
    audio = 0.5 * torch.sin(2 * 3.14159 * freq * torch.linspace(0, 1, sample_rate))
    test_audios.append(audio.unsqueeze(0))

batch_start = time.time()

# 批量嵌入
watermarked_audios = watermark_tool.batch_embed(test_audios, test_messages)
print(f"📦 批量嵌入完成: {len([a for a in watermarked_audios if a is not None])}/{len(test_messages)}")

# 批量提取
batch_results = watermark_tool.batch_extract(watermarked_audios)
batch_time = time.time() - batch_start

print(f"⏱️ 批量处理总时间: {batch_time:.3f}秒")
success_count = sum(1 for r in batch_results if r.get('detected', False))
print(f"🎯 批量成功率: {success_count}/{len(batch_results)} ({success_count/len(batch_results):.1%})")

for i, result in enumerate(batch_results):
    status = "✅" if result.get('detected', False) else "❌"
    msg = result.get('message', 'None')
    conf = result.get('confidence', 0.0)
    print(f"  {i+1}. {status} {test_messages[i]} → {msg} (置信度: {conf:.3f})")

# 6. 🎯 性能统计
print("\n=== 性能统计 ===")
model_info = watermark_tool.get_model_info()
print(f"算法: {model_info['algorithm']}")
print(f"设备: {model_info.get('device', 'Unknown')}")
print(f"采样率: {model_info.get('sample_rate', 'Unknown')} Hz")
print(f"消息位数: {model_info.get('nbits', 'Unknown')}")
```

**📊 性能基准和实测数据**

| 功能指标 | 实测性能 | 技术特点 | 状态 |
|----------|----------|----------|------|
| **基础嵌入** | 0.93秒/1秒音频 | 高效GPU加速，内存优化 | ✅ 生产就绪 |
| **基础提取** | 0.04秒/1秒音频 | 实时检测能力 | ✅ 生产就绪 |
| **音频质量** | SNR: 44.45dB | 几乎无听觉差异，超过40dB标准 | ✅ 高质量 |
| **检测成功率** | 100% | 稳定可靠的算法，无误检 | ✅ 生产就绪 |
| **TTS生成** | 3-8秒/句 | 多语言高质量语音，智能缓存 | ✅ 可用 |
| **批处理** | 2.8秒/3个音频 | 高效并行处理，扩展性好 | ✅ 生产就绪 |
| **噪声鲁棒性** | SNR≥10dB可靠检测 | 抗各种音频攻击 | ✅ 验证通过 |

**🔧 技术实现亮点与解决的问题**：

| 特性 | 实现描述 | 解决的关键问题 | 价值 |
|------|---------|-------------|-----|
| **Meta AudioSeal完整集成** | 深度学习音频水印技术，Python完整封装 | 鲁棒性、抗攻击能力、API稳定性 | 生产环境可靠性 |
| **16位消息编码系统** | SHA256哈希确保消息一致性，字符串↔二进制 | 消息编码一致性、可验证性 | 数据可靠性保证 |
| **设备自适应与优化** | 自动CPU/CUDA检测，张量设备一致性管理 | 设备不匹配、内存优化、兼容性 | 部署灵活性 |
| **3D张量维度处理** | 解决AudioSeal对(batch,channels,time)严格要求 | 模型接口稳定性、维度匹配错误 | 算法集成成功 |
| **Bark TTS智能集成** | 本地优先缓存、符号链接、多语言支持 | 网络依赖、存储空间、语音质量 | 端到端可用性 |
| **高效批处理架构** | 并行音频处理、内存优化、错误容错 | 大规模处理性能、资源利用率 | 生产扩展性 |
| **多格式音频兼容** | WAV/MP3/FLAC等格式无缝支持 | 格式转换、编码兼容性 | 使用便利性 |
| **完整质量评估体系** | SNR/MSE/相关性/鲁棒性全面测试 | 质量监控、性能验证 | 质量保证 |

**🎯 多模态水印统一接口设计（已实现）**：

| 接口要素 | 文本水印(CredID) | 图像水印(PRC) | 音频水印(AudioSeal) | 统一设计理念 |
|----------|----------|----------|----------|--------------|
| **输入格式** | `(model, tokenizer, prompt, message)` | `(prompt, message, key_id)` | `(audio, message)` | 简化参数，专注核心功能 |
| **输出格式** | `{watermarked_text, success, metadata}` | `PIL.Image` | `torch.Tensor 或 file_path` | 直接返回结果对象 |
| **检测输入** | `(text, model, tokenizer, candidates)` | `(image, key_id, mode)` | `(audio, thresholds)` | 支持多种输入格式 |
| **检测输出** | `{extracted_message, confidence, success}` | `{detected, message, confidence}` | `{detected, message, confidence}` | 统一的结果结构 |
| **性能表现** | 候选消息优化搜索，多段处理 | 100%检测率，三种精度模式 | 100%检测率，44dB音质，批处理 | 生产环境就绪 |
| **高级功能** | 智能分割，错误处理 | 多精度检测，离线模式 | TTS集成，鲁棒性测试 | 每个模态的专门优化 |
| **配置管理** | YAML配置文件驱动 | YAML配置文件驱动 | YAML配置文件驱动 | 一致的配置方式 |
| **错误处理** | 详细异常信息和状态 | 详细异常信息和状态 | 详细异常信息和状态 | 统一错误处理机制 |
| **部署状态** | ✅ 生产就绪 | ✅ 生产就绪 | ✅ 生产就绪 | 完整的多模态解决方案 |

### 🚀 音频水印模块使用指南（生产环境）

**基础依赖安装**：
```bash
# 基础功能（必需）
pip install torch torchaudio julius soundfile librosa scipy matplotlib

# 高级功能：文本转语音（可选）
pip install git+https://github.com/suno-ai/bark.git
```

**快速开始示例**：
```python
from src.audio_watermark import create_audio_watermark

# 1. 初始化（自动设备检测）
watermark_tool = create_audio_watermark()

# 2. 基础水印流程
import torch
audio = torch.randn(1, 16000)  # 1秒测试音频
message = "production_watermark_2025"

# 嵌入水印（0.93秒，SNR 44.45dB）
watermarked = watermark_tool.embed_watermark(audio, message)

# 提取水印（0.04秒，100%成功率）
result = watermark_tool.extract_watermark(watermarked)
print(f"检测: {result['detected']}, 消息: {result['message']}")

# 3. 文本转语音+水印（需要Bark）
tts_audio = watermark_tool.generate_audio_with_watermark(
    prompt="Hello, this is a watermarked speech",
    message="tts_demo",
    voice_preset="v2/en_speaker_6"
)
```

**生产环境配置示例**：
```yaml
# config/audio_config.yaml
algorithm: "audioseal"
device: "auto"              # 自动选择最佳设备
nbits: 16                   # 16位消息编码
sample_rate: 16000          # AudioSeal标准采样率

audioseal_params:
  detection_threshold: 0.5  # 检测阈值
  alpha: 1.0               # 水印强度

bark_config:
  model_size: "large"       # 高质量模式
  use_gpu: true             # 启用GPU加速
  temperature: 0.8          # 生成温度
  default_voice: "v2/en_speaker_6"
```

## 🎬 视频水印模块（HunyuanVideo + VideoSeal）

本模块将 Diffusers 的 HunyuanVideo 文生视频与 VideoSeal 水印整合为统一工作流，默认离线使用本地快照，避免联网不确定性。

- 模型卡参考（Diffusers 示例）：[HunyuanVideo 模型卡](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)

### 代码结构
- `src/video_watermark/model_manager.py`
  - 负责定位/确保本地 HunyuanVideo 快照可用；优先本地，必要时可开启下载。
- `src/video_watermark/hunyuan_video_generator.py`
  - 按工作脚本方式从本地快照加载：
    - `HunyuanVideoTransformer3DModel.from_pretrained(local_path, subfolder="transformer", torch_dtype, local_files_only=True)`
    - `HunyuanVideoPipeline.from_pretrained(local_path, transformer=transformer, torch_dtype, local_files_only=True)`
  - CUDA 下启用 `vae.enable_tiling()` 与 `enable_model_cpu_offload()`，降低显存与黑屏风险。
  - 提供：`generate_video(...)` 与 `generate_video_tensor(...)`（返回 `(frames, C, H, W)`）
- `src/video_watermark/videoseal_wrapper.py`
  - 嵌入与提取水印；字符串⇄bits 转换；分块检测聚合。
- `src/video_watermark/utils.py`
  - 视频 I/O（OpenCV）、保存/读取、计时、GPU 内存监控。
- `src/video_watermark/video_watermark.py`
  - 对上层提供统一接口：
    - `generate_video_with_watermark(prompt, message, ...) -> str`
    - `embed_watermark(video_path, message, ...) -> str`
    - `extract_watermark(video_path, max_frames=None, chunk_size=None) -> Dict`
    - `batch_process_videos(...) -> list`

### 主要接口（输入/输出）
- `HunyuanVideoGenerator.generate_video(prompt, negative_prompt=None, num_frames=49, height=720, width=1280, num_inference_steps=30, guidance_scale=6.0, seed=None, output_path=None)`
  - 输入：提示词、帧数（建议 4*k+1，如 13/49/75）、分辨率、步数等
  - 输出：帧序列/数组或保存的文件路径
- `HunyuanVideoGenerator.generate_video_tensor(...) -> torch.Tensor`
  - 输出：`(frames, channels, height, width)`，值域 `[0, 1]`
- `VideoWatermark.generate_video_with_watermark(prompt, message, ..., lowres_attenuation=True) -> str`
  - 输出：带水印视频文件路径
- `VideoWatermark.embed_watermark(video_path, message, ..., max_frames=None) -> str`
  - 输出：带水印视频文件路径
- `VideoWatermark.extract_watermark(video_path, max_frames=None, chunk_size=None) -> Dict[str, Any]`
  - 输出：`{"detected": bool, "message": str, "confidence": float, ...}`

### 使用示例（统一接口）
```python
from src.video_watermark.video_watermark import create_video_watermark

wm = create_video_watermark()

# 文生视频 + 水印（5秒@15fps → 75帧）
out_path = wm.generate_video_with_watermark(
    prompt="阳光洒在海面上",
    message="demo_msg",
    num_frames=75,
    height=320,
    width=512,
    num_inference_steps=30,
    seed=42
)

# 提取水印
result = wm.extract_watermark(out_path, max_frames=50)
```

### 测试与运行
- 回归测试：`tests/test_video_watermark_demo.py`
  - 用例1：纯文生视频（包含非黑屏像素检查与保存）
  - 用例2：文生视频 + 水印嵌入 + 提取验证
- 运行：
```bash
conda activate mmwt
python -u unified_watermark_tool/tests/test_video_watermark_demo.py
```

### 重要约定与建议
- 仅离线加载本地 HunyuanVideo 快照（`local_files_only=True`）。
- CUDA 环境下启用 `vae.enable_tiling()` 与 `enable_model_cpu_offload()`；避免与 `device_map` 并用。
- 5秒@15fps 推荐 `num_frames=75` 与 `320x512` 分辨率；如 OOM，生成器会自适应降参重试。

## 🏷️ AIGC显式标识模块（Visible Marking for Compliance）

本模块为AIGC内容提供显式可见标识功能，是AIGC内容标识系统的重要组成部分。支持对AI生成和用户上传的多媒体内容添加标准化的可见合规标记，满足监管要求并保障用户知情权。

### 核心特性
- **全模态覆盖**：支持文本、图像、音频、视频所有AIGC内容的显式标识
- **合规导向**：内置标准合规文案（"本内容由人工智能生成/合成"），满足监管要求
- **灵活配置**：支持标识位置、样式、时长等多维度自定义配置
- **技术兼容**：自动处理格式转码和浏览器兼容性，确保跨平台展示
- **效果对比**：保留原文件与标识文件，Web界面提供before/after对比展示
- **用户友好**：清晰标识AI生成内容，保障用户知情权和选择权

### 架构实现
位置：`src/utils/visible_mark.py`

核心功能模块：
- `add_text_mark_to_text()`: 文本内容标识添加
- `add_overlay_to_image()`: 图像可见标识叠加
- `add_overlay_to_video_ffmpeg()`: 视频可见标识叠加（基于FFmpeg）
- `add_voice_mark_to_audio()`: 音频语音标识添加（基于Bark TTS）

### 🔹 图像可见标识接口

```python
def add_overlay_to_image(image: Image.Image, 
                        text: str, 
                        position: str = 'bottom_right',
                        font_percent: float = 5.0,
                        font_color: str = '#FFFFFF',
                        bg_rgba: Optional[tuple] = None) -> Image.Image:
    """
    🎯 核心功能: 在图像上添加可见文字标识
    
    📋 详细工作流程:
    1. 根据图像尺寸计算字体大小和位置
    2. 创建透明图层进行文字绘制
    3. 应用抗锯齿和阴影效果提升可读性
    4. 合成最终带标识的图像
    
    📥 参数说明:
        image: PIL图像对象
        text: 标识文字，如 "本内容由人工智能生成/合成"
        position: 标识位置
            - 'top_left': 左上角
            - 'top_right': 右上角  
            - 'bottom_left': 左下角
            - 'bottom_right': 右下角（默认）
            - 'center': 居中
        font_percent: 字体大小占图像宽度的百分比 (1.0-15.0，默认5.0)
        font_color: 字体颜色，支持十六进制 '#FFFFFF' 或颜色名 'white'
        bg_rgba: 背景颜色 (R,G,B,A)，None表示无背景
        
    📤 返回值:
        PIL.Image: 带有可见标识的图像
        
    🚨 使用示例:
        from PIL import Image
        from src.utils.visible_mark import add_overlay_to_image
        
        img = Image.open("input.jpg")
        marked_img = add_overlay_to_image(
            img, 
            "本内容由人工智能生成",
            position="bottom_right",
            font_percent=4.0,
            font_color="#FFFF00"
        )
        marked_img.save("output.jpg")
    """
```

### 🔹 视频可见标识接口

```python
def add_overlay_to_video_ffmpeg(input_path: str,
                               output_path: str,
                               text: str,
                               position: str = 'bottom_right',
                               font_percent: float = 5.0,
                               duration_seconds: float = 2.0,
                               font_color: str = 'white',
                               box_color: str = 'transparent') -> str:
    """
    🎯 核心功能: 使用FFmpeg在视频上添加可见文字标识
    
    📋 详细工作流程:
    1. 检测视频分辨率和帧率信息
    2. 计算字体大小和标识显示位置
    3. 使用FFmpeg drawtext滤镜叠加文字
    4. 输出浏览器兼容格式的标识视频
    
    📥 参数说明:
        input_path: 输入视频文件路径
        output_path: 输出视频文件路径
        text: 标识文字内容
        position: 标识位置，支持与图像相同的位置选项
        font_percent: 字体大小占视频宽度的百分比 (1.0-10.0，默认5.0)
        duration_seconds: 标识显示时长（秒，默认2.0秒）
        font_color: 字体颜色，支持FFmpeg颜色名称
        box_color: 文字背景框颜色，'transparent'表示透明
        
    📤 返回值:
        str: 输出视频文件路径
        
    🚨 使用示例:
        from src.utils.visible_mark import add_overlay_to_video_ffmpeg
        
        output_path = add_overlay_to_video_ffmpeg(
            "input.mp4",
            "output.mp4", 
            "本内容由人工智能生成",
            position="bottom_right",
            font_percent=3.0,
            duration_seconds=3.0,
            font_color="yellow"
        )
        print(f"标识视频已保存: {output_path}")
    """
```

### 🔹 音频可见标识接口

```python
def add_voice_mark_to_audio(input_path: str,
                           output_path: str, 
                           mark_text: str,
                           position: str = 'start',
                           voice_preset: str = 'v2/zh_speaker_6') -> str:
    """
    🎯 核心功能: 在音频文件中添加语音标识（需要Bark TTS）
    
    📋 详细工作流程:
    1. 使用Bark TTS生成标识语音片段
    2. 加载原始音频文件
    3. 根据位置参数混合语音标识和原始音频
    4. 输出带有语音标识的最终音频文件
    
    📥 参数说明:
        input_path: 输入音频文件路径（支持WAV, MP3等）
        output_path: 输出音频文件路径
        mark_text: 标识语音内容，如 "本内容由人工智能生成" 
        position: 标识位置
            - 'start': 音频开头（默认）
            - 'end': 音频结尾
        voice_preset: Bark语音预设
            - 'v2/zh_speaker_6': 中文女声（默认）
            - 'v2/en_speaker_6': 英文女声
            - 其他Bark支持的预设
            
    📤 返回值:
        str: 输出音频文件路径
        
    🚨 依赖要求:
        需要安装Bark TTS: pip install git+https://github.com/suno-ai/bark.git
        
    🚨 使用示例:
        from src.utils.visible_mark import add_voice_mark_to_audio
        
        output_path = add_voice_mark_to_audio(
            "input.wav",
            "output.wav",
            "本内容由人工智能生成", 
            position="start",
            voice_preset="v2/zh_speaker_6"
        )
        print(f"标识音频已保存: {output_path}")
    """
```

### 🔹 文本可见标识接口

```python
def add_text_mark_to_text(text: str, 
                         mark: str = "本内容由人工智能生成/合成",
                         position: str = 'start') -> str:
    """
    🎯 核心功能: 在文本内容中插入可见标识文案
    
    📋 详细工作流程:
    1. 根据位置参数确定插入点
    2. 处理标识文案的格式化（换行、分隔符等）
    3. 将标识文案与原始文本合并
    4. 返回带标识的完整文本
    
    📥 参数说明:
        text: 原始文本内容
        mark: 标识文案，默认合规文案
        position: 插入位置
            - 'start': 文本开头（默认）
            - 'end': 文本结尾
            
    📤 返回值:
        str: 带有可见标识的文本
        
    🚨 使用示例:
        from src.utils.visible_mark import add_text_mark_to_text
        
        original_text = "这是一段示例文本内容。"
        marked_text = add_text_mark_to_text(
            original_text,
            mark="本内容由AI生成",
            position="start" 
        )
        print(marked_text)
        # 输出: 本内容由AI生成\n\n这是一段示例文本内容。
    """
```

### Web API集成

可见标识功能已完全集成到Flask Web应用中，提供RESTful API接口：

#### API端点
- **路径**: `/api/visible_mark`
- **方法**: `POST`
- **功能**: 为上传的文件添加可见标识

#### 请求参数
```javascript
// 表单数据格式
{
    "modality": "image|audio|video|text",    // 模态类型
    "mark_text": "标识内容",                   // 自定义标识文字
    "file": File,                            // 上传的文件（文本模态除外）
    "text": "文本内容",                       // 文本模态专用
    
    // 图像专用参数
    "position": "bottom_right",              // 标识位置
    "font_percent": 5.0,                     // 字体大小百分比
    "font_color": "#FFFFFF",                 // 字体颜色
    
    // 视频专用参数  
    "duration_seconds": 2.0,                 // 显示时长
    
    // 音频专用参数
    "voice_preset": "v2/zh_speaker_6"        // 语音预设
}
```

#### 响应格式
```json
{
    "task_id": "task_1757324404_71c759dc",
    "status": "completed",
    "output_path": "/demo_outputs/task_1757324404_marked_image.png",
    "timestamp": "2025-01-08T12:34:56"
}
```

### 使用示例：Web界面集成

前端JavaScript调用示例：
```javascript
// 图像可见标识
function addVisibleMarkToImage(file, markText, position, fontSize, fontColor) {
    const formData = new FormData();
    formData.append('modality', 'image');
    formData.append('file', file);
    formData.append('mark_text', markText || '本内容由人工智能生成/合成');
    formData.append('position', position || 'bottom_right');
    formData.append('font_percent', fontSize || 5.0);
    formData.append('font_color', fontColor || '#FFFFFF');
    
    return fetch('/api/visible_mark', {
        method: 'POST',
        body: formData
    }).then(response => response.json());
}

// 视频可见标识
function addVisibleMarkToVideo(file, markText, position, duration) {
    const formData = new FormData();
    formData.append('modality', 'video'); 
    formData.append('file', file);
    formData.append('mark_text', markText || '本内容由人工智能生成/合成');
    formData.append('position', position || 'bottom_right');
    formData.append('font_percent', 4.0);
    formData.append('duration_seconds', duration || 2.0);
    formData.append('font_color', 'white');
    
    return fetch('/api/visible_mark', {
        method: 'POST',
        body: formData
    }).then(response => response.json());
}

// 音频可见标识
function addVisibleMarkToAudio(file, markText, position, voicePreset) {
    const formData = new FormData();
    formData.append('modality', 'audio');
    formData.append('file', file);
    formData.append('mark_text', markText || '本内容由人工智能生成');
    formData.append('position', position || 'start'); 
    formData.append('voice_preset', voicePreset || 'v2/zh_speaker_6');
    
    return fetch('/api/visible_mark', {
        method: 'POST',
        body: formData
    }).then(response => response.json());
}
```

### 配置参数

可见标识功能的参数可通过配置文件控制：

```yaml
# config/visible_mark_config.yaml
visible_marking:
  # 默认标识文案
  default_text: "本内容由人工智能生成/合成"
  
  # 图像标识配置
  image_config:
    default_position: "bottom_right"
    default_font_percent: 5.0
    default_font_color: "#FFFFFF"
    supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    
  # 视频标识配置  
  video_config:
    default_position: "bottom_right"
    default_font_percent: 4.0
    default_duration: 2.0
    default_font_color: "white"
    supported_formats: [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
  # 音频标识配置
  audio_config:
    default_position: "start"
    default_voice_preset: "v2/zh_speaker_6"
    supported_formats: [".wav", ".mp3", ".flac", ".m4a", ".aac"]
    
  # 文本标识配置
  text_config:
    default_position: "start" 
    separator: "\n\n"
```

### 实际应用场景

1. **内容合规标识**：为AI生成的图像、视频、音频内容添加标准合规标识
2. **版权声明**：在媒体文件中添加版权或来源信息
3. **品牌标识**：为企业内容添加品牌watermark或logo文字
4. **法律合规**：满足不同地区对AI生成内容标识的法律要求
5. **内容溯源**：为内容添加生成时间、模型版本等元信息

### 技术特点总结

| 特性 | 描述 | 优势 |
|------|------|------|
| **多模态统一** | 支持文本、图像、音频、视频四种模态 | 一致的API接口，便于集成 |
| **灵活配置** | 支持位置、样式、时长等多维度配置 | 适应不同应用场景需求 |
| **高质量输出** | 抗锯齿、阴影效果、格式优化 | 专业级视觉效果 |
| **浏览器兼容** | 自动格式转码，确保Web播放 | 无缝Web集成体验 |
| **合规导向** | 内置标准合规文案和位置 | 满足监管要求 |
| **批量处理** | 支持API批量调用和处理 | 高效的生产环境应用 |