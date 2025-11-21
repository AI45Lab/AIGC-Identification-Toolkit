# 统一水印工具 - 系统整体架构

## 系统整体架构图

```mermaid
graph TB
    %% Web界面层
    subgraph "Web界面层 (Presentation Layer)"
        UI[Web界面 - index.html]
        UI --> |用户交互| JS[JavaScript前端逻辑]
        JS --> |API调用| REST[REST API接口]
    end

    %% API接口层  
    subgraph "API接口层 (API Gateway Layer)"
        REST --> |Flask路由| APP[app.py - Flask应用]
        APP --> |任务管理| TASK[任务状态管理器]
        APP --> |文件管理| FILE[文件服务器]
        APP --> |候选消息| CAND[候选消息管理器]
    end

    %% 统一接口层
    subgraph "统一接口层 (Unified Interface Layer)"
        APP --> |调用| TOOL[WatermarkTool - 高级API]
        TOOL --> |依赖| ENGINE[UnifiedWatermarkEngine - 核心引擎]
        ENGINE --> |配置管理| CONFIG[配置管理器]
    end

    %% 模态处理层
    subgraph "模态处理层 (Modality Processing Layer)"
        ENGINE --> |文本| TEXT[TextWatermark]
        ENGINE --> |图像| IMAGE[ImageWatermark]  
        ENGINE --> |音频| AUDIO[AudioWatermark]
        ENGINE --> |视频| VIDEO[VideoWatermark]
    end

    %% 算法实现层
    subgraph "算法实现层 (Algorithm Implementation Layer)"
        %% 文本算法
        TEXT --> CREDID[CredID算法框架]
        CREDID --> |多方水印| CREDID_CORE[CredID核心算法]
        
        %% 图像算法
        IMAGE --> |默认后端| VIDEOSEAL_IMG[VideoSeal图像水印]
        IMAGE --> |可选后端| PRC[PRC-Watermark]
        VIDEOSEAL_IMG --> |复用| VIDEOSEAL_CORE[VideoSeal核心算法]
        
        %% 音频算法
        AUDIO --> AUDIOSEAL[AudioSeal算法]
        AUDIO --> |TTS生成| BARK[Bark文本转语音]
        
        %% 视频算法
        VIDEO --> |生成模型| HUNYUAN[HunyuanVideo生成]
        VIDEO --> |水印嵌入| VIDEOSEAL_CORE
    end

    %% 底层支持层
    subgraph "底层支持层 (Infrastructure Layer)"
        %% 模型管理
        MODEL_MGR[模型管理器]
        CACHE[模型缓存]
        
        %% 设备管理
        DEVICE[设备管理 - CPU/CUDA]
        
        %% 文件系统
        STORAGE[文件存储系统]
        
        %% 配置系统
        CONFIG_FILES[配置文件系统]
    end

    %% 连接底层支持
    CREDID_CORE --> MODEL_MGR
    VIDEOSEAL_CORE --> MODEL_MGR
    AUDIOSEAL --> MODEL_MGR
    HUNYUAN --> MODEL_MGR
    BARK --> MODEL_MGR
    
    MODEL_MGR --> CACHE
    MODEL_MGR --> DEVICE
    
    FILE --> STORAGE
    CONFIG --> CONFIG_FILES

    %% 样式定义
    classDef webLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef apiLayer fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef unifiedLayer fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef modalityLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef algorithmLayer fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef infrastructureLayer fill:#f1f8e9,stroke:#558b2f,stroke-width:2px

    class UI,JS,REST webLayer
    class APP,TASK,FILE,CAND apiLayer
    class TOOL,ENGINE,CONFIG unifiedLayer
    class TEXT,IMAGE,AUDIO,VIDEO modalityLayer
    class CREDID,CREDID_CORE,VIDEOSEAL_IMG,VIDEOSEAL_CORE,PRC,AUDIOSEAL,BARK,HUNYUAN algorithmLayer
    class MODEL_MGR,CACHE,DEVICE,STORAGE,CONFIG_FILES infrastructureLayer
```

## 架构层次说明

### 1. Web界面层 (Presentation Layer)
- **Web界面 (index.html)**: 统一的多模态水印操作界面
- **JavaScript前端逻辑**: 处理用户交互、模态切换、实时状态更新
- **REST API接口**: 前后端通信桥梁

### 2. API接口层 (API Gateway Layer)  
- **Flask应用 (app.py)**: Web服务器和API路由管理
- **任务状态管理器**: 跟踪水印处理任务的实时状态
- **文件服务器**: 处理文件上传、下载和存储
- **候选消息管理器**: 管理水印消息用于提取时的智能匹配

### 3. 统一接口层 (Unified Interface Layer)
- **WatermarkTool**: 高级API封装，向后兼容的用户接口
- **UnifiedWatermarkEngine**: 核心引擎，协调所有模态的水印操作
- **配置管理器**: 统一管理各模态的配置参数

### 4. 模态处理层 (Modality Processing Layer)
- **TextWatermark**: 文本水印处理器（仅AI生成模式）
- **ImageWatermark**: 图像水印处理器（双模式：AI生成+文件上传）
- **AudioWatermark**: 音频水印处理器（双模式：TTS生成+文件上传）
- **VideoWatermark**: 视频水印处理器（双模式：AI生成+文件上传）

### 5. 算法实现层 (Algorithm Implementation Layer)
- **CredID**: 多方文本水印算法框架
- **VideoSeal**: 通用视频水印算法（支持图像和视频）
- **AudioSeal**: 深度学习音频水印算法
- **PRC-Watermark**: 可选的图像水印算法
- **HunyuanVideo**: AI视频生成模型
- **Bark TTS**: 文本转语音生成模型

### 6. 底层支持层 (Infrastructure Layer)
- **模型管理器**: 统一的深度学习模型加载和管理
- **模型缓存**: 本地模型存储和缓存策略
- **设备管理**: CPU/CUDA设备自适应分配
- **文件存储系统**: 多媒体文件的存储和访问
- **配置文件系统**: YAML配置文件的解析和管理

## 核心设计原则

1. **模块化设计**: 每个模态独立实现，便于维护和扩展
2. **双模式支持**: AI生成模式和文件上传模式无缝切换
3. **离线优先**: 优先使用本地缓存模型，减少网络依赖
4. **懒加载机制**: 按需初始化组件，优化内存使用
5. **统一配置**: 集中化配置管理，支持参数优化
6. **原文件保存**: 自动保存原文件和水印文件用于对比显示