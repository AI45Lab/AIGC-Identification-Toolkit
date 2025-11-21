# 文本水印双算法系统使用指南

## 📋 概述

本系统集成了两种文本水印算法：
- **PostMark**（默认）：后处理水印，支持黑盒LLM
- **CredID**：生成时水印，需要访问模型logits


## 🚀 快速开始

### 方式1: 使用默认PostMark算法

```python
from src.unified.watermark_tool import WatermarkTool

tool = WatermarkTool()

# PostMark是后处理，所以需要先有生成的文本
generated_text = """
Your LLM generated text here...
This could be from GPT-4, Claude, or any other LLM.
"""

# 嵌入水印
watermarked_text = tool.embed(generated_text, "my_watermark", 'text')

# 提取水印（需要提供原始水印词列表以提高准确率）
result = tool.extract(watermarked_text, 'text',
                     original_words=["detected", "words", "list"])
print(f"检测到水印: {result['detected']}")
print(f"置信度: {result['confidence']:.2%}")
```

### 方式2: 切换到CredID算法

#### 修改配置文件

编辑 `config/text_config.yaml`:

```yaml
# 将algorithm从postmark改为credid
algorithm: "credid"
```

#### 使用CredID

```python
from src.unified.watermark_tool import WatermarkTool
from transformers import AutoModelForCausalLM, AutoTokenizer

tool = WatermarkTool()

# 加载模型（CredID需要）
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# CredID是生成时嵌入，输入的是prompt
prompt = "Tell me about AI:"

# 嵌入水印（生成时）
watermarked_text = tool.embed(prompt, "my_watermark", 'text',
                              model=model, tokenizer=tokenizer)

# 提取水印
result = tool.extract(watermarked_text, 'text',
                     model=model, tokenizer=tokenizer,
                     candidates_messages=["my_watermark"])
print(f"检测到水印: {result['detected']}")
print(f"提取的消息: {result['message']}")
```

## 📖 详细用法

### PostMark算法详解

#### 依赖模型

PostMark需要以下本地模型（已下载到您的环境）:
- `nomic-ai/nomic-embed-text-v1` (嵌入模型)
- `mistralai/Mistral-7B-Instruct-v0.2` (插入LLM)
- `paragram_xxl.pkl` (相似度计算)
- `filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl` (预计算嵌入)

#### 配置参数

```yaml
postmark:
  embedder: "nomic"              # 嵌入模型
  inserter: "mistral-7b-inst"    # 插入LLM
  ratio: 0.12                    # 水印词比例（12%）
  iterate: "v2"                  # 迭代版本
  threshold: 0.7                 # 检测阈值
```

#### 完整示例

```python
from src.text_watermark.postmark_watermark import PostMarkWatermark

# 初始化
config = {
    'embedder': 'nomic',
    'inserter': 'mistral-7b-inst',
    'ratio': 0.12,
    'iterate': 'v2',
    'threshold': 0.7
}
watermark = PostMarkWatermark(config)

# 已生成的文本（来自任何LLM）
text = "Your generated text here..."

# 嵌入水印
result = watermark.embed(text, message="watermark_id")

# 提取水印（使用原始水印词列表）
detection = watermark.extract(
    result['watermarked_text'],
    original_words=result['watermark_words']  # 关键！
)

print(f"水印存在率: {detection['presence_score']:.2%}")
```

### CredID算法详解

#### 配置参数

```yaml
credid:
  mode: "lm"                     # 模式：lm或random
  model_name: "huggyllama/llama-7b"
  lm_params:
    delta: 2.0                   # 水印强度
    message_len: 10              # 消息长度（位）
  wm_params:
    encode_ratio: 4              # 编码比率
```

#### 完整示例

```python
from src.text_watermark.credid_watermark import CredIDWatermark
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化
config = {
    'mode': 'lm',
    'model_name': 'gpt2',
    'lm_params': {'delta': 1.5, 'message_len': 10},
    'wm_params': {'encode_ratio': 8}
}
watermark = CredIDWatermark(config)

# 加载模型
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# 嵌入（生成时）
result = watermark.embed(model, tokenizer,
                        prompt="AI is",
                        message="my_watermark")

# 提取
detection = watermark.extract(
    result['watermarked_text'],
    model=model,
    tokenizer=tokenizer,
    candidates_messages=["my_watermark", "wrong_msg"]
)

print(f"提取的消息: {detection['extracted_message']}")
```

## 🔄 算法切换方法

### 方法1: 修改配置文件

编辑 `config/text_config.yaml`:

```yaml
# 切换到PostMark
algorithm: "postmark"

# 或切换到CredID
# algorithm: "credid"
```

### 方法2: 代码动态切换

```python
from src.text_watermark.text_watermark import TextWatermark

watermark = TextWatermark()

# 查看当前算法
print(f"当前算法: {watermark.get_algorithm()}")

# 切换到CredID
watermark.set_algorithm('credid')

# 切换到PostMark
watermark.set_algorithm('postmark')
```

### 方法3: UnifiedEngine切换

```python
from src.unified.unified_engine import UnifiedWatermarkEngine

engine = UnifiedWatermarkEngine()

# 查看默认算法
print(engine.get_default_algorithms())
# 输出: {'text': 'postmark', 'image': 'videoseal', ...}

# 内部会自动根据配置使用相应算法
```

## 🧪 运行测试

验证双算法系统是否正常工作：

```bash
# 从项目根目录运行完整测试套件
python tests/test_text_watermark_dual_algorithm.py
```

测试内容包括：
1. PostMark基础功能测试
2. CredID基础功能测试
3. 统一接口算法切换测试
4. 配置文件算法切换测试
5. UnifiedEngine集成测试

## 📌 重要注意事项

### PostMark特性
1. **后处理**：必须对已生成的文本进行处理
2. **水印词列表**：检测时最好提供原始水印词列表
3. **本地模型**：需要Mistral-7B等大模型，确保GPU内存充足
4. **处理时间**：嵌入过程需要1-5秒（取决于文本长度）

### CredID特性
1. **生成时嵌入**：输入是prompt，输出是生成+水印的文本
2. **模型要求**：需要完整的模型和分词器访问权限
3. **候选消息**：提取时提供候选消息列表可提高准确率
4. **内存需求**：需要加载完整LLM到内存

### 性能对比
- **PostMark**：适合生产环境，支持任何LLM
- **CredID**：适合研究场景，检测率略高但限制多

## 🔧 故障排除

### PostMark问题

**问题1: 找不到PostMark模块**
```bash
# 检查PostMark目录
ls src/text_watermark/PostMark/

# 应该包含：postmark/, paragram_xxl.pkl, 等文件
```

**问题2: 模型文件缺失**
```bash
# 检查关键文件
ls src/text_watermark/PostMark/*.pkl

# 应该有：
# - paragram_xxl.pkl
# - filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl
# - valid_wtmk_words_in_wiki_base-only-f1000.pkl
```

**问题3: Mistral模型未下载**
```bash
# 检查HuggingFace缓存
ls ~/.cache/huggingface/models--mistralai--Mistral-7B-Instruct-v0.2/
```

### CredID问题

**问题1: 模型加载失败**
- 确保模型名称正确
- 检查HuggingFace缓存目录
- 尝试使用更小的测试模型（如`sshleifer/tiny-gpt2`）

**问题2: 内存不足**
- 减少`max_new_tokens`
- 使用CPU模式（`device: 'cpu'`）
- 使用更小的模型

## 📚 相关文件

- 配置文件: `config/text_config.yaml`
- PostMark封装: `src/text_watermark/postmark_watermark.py`
- CredID封装: `src/text_watermark/credid_watermark.py`
- 统一接口: `src/text_watermark/text_watermark.py`
- 测试脚本: `tests/test_text_watermark_dual_algorithm.py`
- 引擎集成: `src/unified/unified_engine.py`

## 💡 最佳实践

### 使用PostMark的场景
- ✅ 使用第三方API（OpenAI、Anthropic等）
- ✅ 无法访问模型内部
- ✅ 需要处理已生成的文本
- ✅ 生产环境部署

### 使用CredID的场景
- ✅ 自己部署的开源模型
- ✅ 完全控制生成过程
- ✅ 研究和实验
- ✅ 需要最高检测准确率

## 🎓 进一步学习

- [PostMark论文](https://arxiv.org/abs/2403.07344)
- [CredID文档](src/text_watermark/credid/README.md)
- [统一接口文档](src/unified/README.md)
