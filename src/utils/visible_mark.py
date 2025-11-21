#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态可见标识工具模块
为文本、图像、音频、视频内容添加符合规定的AI生成/合成显示标识
"""

import os
import re
import math
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 默认提示语，覆盖"人工智能/AI + 生成/合成"要求
DEFAULT_LABEL = "本内容由人工智能生成/合成"

def ensure_label_text(mark_text: Optional[str]) -> str:
    """
    确保标识文本符合规定要求：
    - 同时包含"人工智能"或"AI" 
    - 同时包含"生成"和/或"合成"
    """
    text = (mark_text or DEFAULT_LABEL).strip()
    
    # 检查是否包含必要关键词
    ai_ok = ("人工智能" in text) or ("AI" in text.upper())
    gen_ok = ("生成" in text) or ("合成" in text)
    
    if not ai_ok or not gen_ok:
        logger.warning(f"标识文本不符合要求，使用默认文本: {text}")
        text = DEFAULT_LABEL
    
    return text

def find_system_font() -> str:
    """查找可渲染中文的字体，优先项目内字体，返回绝对路径"""
    # 获取项目根目录绝对路径
    base = Path(__file__).resolve().parents[2]  # 指向 unified_watermark_tool 根目录
    
    candidates = [
        # 项目内字体（绝对路径）
        base / "templates" / "fonts" / "NotoSansSC-Regular.otf",
        base / "templates" / "fonts" / "NotoSansCJK-Regular.ttc", 
        base / "fonts" / "NotoSansSC-Regular.otf",
        base / "fonts" / "NotoSansCJK-Regular.ttc",
        
        # Linux 系统中文字体（按优先级排序）
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        
        # macOS 中文字体
        Path("/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"), 
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        
        # Windows 中文字体
        Path("C:/Windows/Fonts/msyh.ttc"),     # 微软雅黑
        Path("C:/Windows/Fonts/simhei.ttf"),   # 黑体
        Path("C:/Windows/Fonts/simsun.ttc"),   # 宋体
        Path("C:/Windows/Fonts/simkai.ttf"),   # 楷体
        
        # 最后兜底的非中文字体（可能显示为方块）
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    
    for font_path in candidates:
        if font_path.exists():
            logger.info(f"使用字体: {font_path}")
            return str(font_path)
    
    logger.warning("未找到合适的中文字体，将使用默认字体（可能显示为方块）")
    return ""

def add_text_mark_to_text(text: str, mark_text: Optional[str] = None, 
                         position: str = "start") -> str:
    """
    为文本添加可见标识
    
    Args:
        text: 原始文本内容
        mark_text: 自定义标识文本
        position: 标识位置 start|middle|end
    
    Returns:
        添加标识后的文本
    """
    label = ensure_label_text(mark_text)
    formatted_label = f"【提示】{label}"
    
    if position == "end":
        # 文本末尾
        return text + ("\n\n" if not text.endswith("\n") else "") + formatted_label + "\n"
    elif position == "middle":
        # 文本中间适当位置
        lines = text.split('\n')
        mid = len(lines) // 2
        lines.insert(mid, f"\n{formatted_label}\n")
        return '\n'.join(lines)
    else:
        # 文本起始（默认）
        return f"{formatted_label}\n\n{text}"

def add_overlay_to_image(img_in: Image.Image, mark_text: Optional[str] = None,
                        position: str = "bottom_right", font_percent: float = 5.0,
                        font_color: str = "#FFFFFF", bg_rgba: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 153)) -> Image.Image:
    """
    为图像添加可见文字标识
    
    Args:
        img_in: 输入图像
        mark_text: 自定义标识文本
        position: 标识位置 top_left|top_right|bottom_left|bottom_right
        font_percent: 字体大小占最短边的百分比（不低于5%）
        font_color: 字体颜色
        bg_rgba: 背景色（RGBA），为None则无背景框
    
    Returns:
        添加标识后的图像
    """
    label = ensure_label_text(mark_text)
    img = img_in.convert("RGBA")
    W, H = img.size
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img, "RGBA")
    
    # 计算字体大小：不低于最短边的5%
    min_side = min(W, H)
    font_size = max(1, int(min_side * max(font_percent, 5.0) / 100.0))
    
    # 加载字体
    try:
        font_path = find_system_font()
        if font_path:
            try:
                # 首先尝试正常加载
                font = ImageFont.truetype(font_path, font_size)
                # 验证字体是否支持中文（测试渲染一个中文字符）
                test_bbox = draw.textbbox((0, 0), "中", font=font)
                if test_bbox[2] - test_bbox[0] > 0:  # 有实际宽度，说明字体支持中文
                    logger.info(f"字体验证成功: {font_path}")
                else:
                    raise ValueError("字体不支持中文字符")
            except Exception as font_error:
                logger.warning(f"字体文件 {font_path} 加载失败: {font_error}")
                # 对于.ttc文件，尝试指定索引0
                if font_path.endswith('.ttc'):
                    try:
                        font = ImageFont.truetype(font_path, font_size, index=0)
                        logger.info(f"使用索引0成功加载TTC字体: {font_path}")
                    except Exception as ttc_error:
                        logger.warning(f"TTC字体索引加载失败: {ttc_error}")
                        raise ttc_error
                else:
                    raise font_error
        else:
            # 没有找到字体路径，使用默认字体
            font = ImageFont.load_default()
            logger.warning("使用默认字体，中文可能显示为方块")
    except Exception as e:
        logger.warning(f"字体加载失败: {e}，使用默认字体")
        font = ImageFont.load_default()
        
    # 记录实际使用的字体信息用于调试
    try:
        if hasattr(font, 'getname'):
            font_name = font.getname()
            logger.info(f"最终使用字体: {font_name}")
        elif hasattr(font, 'font') and hasattr(font.font, 'family'):
            logger.info(f"最终使用字体族: {font.font.family}")
    except:
        logger.info("无法获取字体名称信息")
    
    # 获取文本尺寸
    bbox = draw.textbbox((0, 0), label, font=font, stroke_width=2)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 计算文本位置
    padding = max(8, font_size // 4)
    
    if position == "top_left":
        x, y = padding, padding
    elif position == "top_right":
        x, y = W - text_width - padding, padding
    elif position == "bottom_left":
        x, y = padding, H - text_height - padding
    else:  # bottom_right (默认)
        x, y = W - text_width - padding, H - text_height - padding
    
    # 绘制半透明背景框（可选）
    if bg_rgba is not None:
        background_padding = padding // 2
        draw.rectangle([
            x - background_padding, 
            y - background_padding,
            x + text_width + background_padding, 
            y + text_height + background_padding
        ], fill=bg_rgba)
    
    # 绘制文字（带描边以增强可辨识度）
    stroke_width = max(2, font_size // 15)  # 增强描边宽度
    stroke_color = (0, 0, 0, 255) if font_color.upper() in ['#FFFFFF', '#FFF', 'WHITE'] else (255, 255, 255, 255)
    
    draw.text((x, y), label, font=font, fill=font_color, 
              stroke_width=stroke_width, stroke_fill=stroke_color)
    
    return img.convert("RGB")

def add_overlay_to_video_ffmpeg(input_path: str, output_path: str,
                               mark_text: Optional[str] = None, position: str = "bottom_right",
                               font_percent: float = 5.0, duration_seconds: float = 2.0,
                               font_color: str = "white", box_color: str = "transparent") -> str:
    """
    使用FFmpeg为视频添加可见文字标识
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        mark_text: 自定义标识文本
        position: 标识位置
        font_percent: 字体大小百分比（不低于5%）
        duration_seconds: 显示持续时间（不少于2秒）
        font_color: 字体颜色
        box_color: 背景框颜色
    
    Returns:
        输出视频路径
    """
    label = ensure_label_text(mark_text)
    
    # 确保持续时间不少于2秒
    duration = max(duration_seconds, 2.0)
    
    # 位置映射
    position_map = {
        "top_left": ("10", "10"),
        "top_right": ("w-tw-10", "10"),
        "bottom_left": ("10", "h-th-10"),
        "bottom_right": ("w-tw-10", "h-th-10"),
    }
    
    x_expr, y_expr = position_map.get(position, position_map["bottom_right"])
    
    # 字体大小：最短边的百分比
    fontsize = f"h*{max(font_percent, 5.0)/100.0:.3f}"
    
    # 查找字体文件
    fontfile = find_system_font()
    fontopt = f":fontfile='{fontfile}'" if fontfile else ""
    
    if fontfile:
        logger.info(f"FFmpeg将使用字体: {fontfile}")
    else:
        logger.warning("FFmpeg未找到中文字体，可能显示为方块")
    
    # 转义文本中的特殊字符
    text_escaped = label.replace(":", r"\:").replace("'", r"\'").replace("%", r"\%")
    
    # 构建drawtext滤镜参数
    if box_color == "transparent":
        # 无背景框版本，但添加阴影效果增强可读性
        drawtext_filter = (
            f"drawtext=text='{text_escaped}'{fontopt}:fontsize={fontsize}:"
            f"fontcolor={font_color}:box=0:"
            f"borderw=3:bordercolor=black@0.8:"  # 添加黑色边框增强可读性
            f"x={x_expr}:y={y_expr}:enable='between(t,0,{duration:.2f})'"
        )
    else:
        # 带背景框版本
        drawtext_filter = (
            f"drawtext=text='{text_escaped}'{fontopt}:fontsize={fontsize}:"
            f"fontcolor={font_color}:box=1:boxcolor={box_color}:boxborderw=6:"
            f"x={x_expr}:y={y_expr}:enable='between(t,0,{duration:.2f})'"
        )
    
    # 构建FFmpeg命令
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", drawtext_filter,
        "-c:v", "libx264",
        "-profile:v", "main",
        "-level", "4.0", 
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]
    
    try:
        logger.info(f"执行FFmpeg命令: {' '.join(cmd[:10])}...")
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, text=True)
        logger.info(f"视频标识添加完成: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        logger.error(f"FFmpeg处理失败: {error_msg}")
        raise RuntimeError(f"视频处理失败: {error_msg[:200]}...")

def add_voice_mark_to_audio(input_path: str, output_path: str,
                           mark_text: Optional[str] = None, position: str = "start",
                           voice_preset: str = "v2/zh_speaker_6") -> str:
    """
    为音频添加语音标识
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        mark_text: 自定义标识文本
        position: 标识位置 start|middle|end
        voice_preset: 语音预设
    
    Returns:
        输出音频路径
    """
    label = ensure_label_text(mark_text)
    prompt = f"提示：{label}。"
    
    try:
        # 导入音频工具
        from src.audio_watermark.utils import AudioIOUtils
        
        # 1) 读取原始音频
        original_audio, sample_rate = AudioIOUtils.load_audio(
            input_path, target_sample_rate=16000, mono=True
        )
        
        # 2) 生成提示语音（使用Bark TTS）
        try:
            from src.audio_watermark.bark_generator import create_bark_generator
            
            tts_generator = create_bark_generator(target_sample_rate=16000)
            voice_mark = tts_generator.generate_audio(
                prompt=prompt,
                voice_preset=voice_preset,
                temperature=0.7  # 正常语速
            )
            
            logger.info(f"使用Bark TTS生成语音标识: {prompt}")
            
        except Exception as bark_error:
            logger.warning(f"Bark TTS不可用: {bark_error}，使用备用提示音")
            
            # 备用方案：生成2秒提示音（440Hz正弦波）
            import torch
            duration_samples = int(2.0 * 16000)  # 2秒
            t = torch.arange(duration_samples, dtype=torch.float32) / 16000.0
            
            # 生成渐变的提示音
            voice_mark = 0.3 * torch.sin(2 * math.pi * 440 * t)  # 440Hz
            voice_mark *= torch.exp(-t * 2.0)  # 渐弱效果
            voice_mark = voice_mark.unsqueeze(0)  # 添加通道维度
        
        # 3) 按位置拼接音频
        import torch
        
        if position == "end":
            # 音频末尾
            combined_audio = torch.cat([original_audio, voice_mark], dim=-1)
        elif position == "middle":
            # 音频中间适当位置
            mid_point = original_audio.shape[-1] // 2
            combined_audio = torch.cat([
                original_audio[..., :mid_point], 
                voice_mark,
                original_audio[..., mid_point:]
            ], dim=-1)
        else:
            # 音频起始（默认）
            combined_audio = torch.cat([voice_mark, original_audio], dim=-1)
        
        # 4) 保存输出音频
        AudioIOUtils.save_audio(combined_audio, output_path, sample_rate=16000)
        
        logger.info(f"音频标识添加完成: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"音频标识添加失败: {e}")
        raise RuntimeError(f"音频处理失败: {str(e)}")

def validate_mark_requirements(mark_text: Optional[str]) -> bool:
    """
    验证标识文本是否符合规定要求
    
    Returns:
        是否符合要求
    """
    if not mark_text:
        return True  # 使用默认文本
    
    text = mark_text.strip()
    ai_ok = ("人工智能" in text) or ("AI" in text.upper())
    gen_ok = ("生成" in text) or ("合成" in text)
    
    return ai_ok and gen_ok

# 便捷函数：批量处理
def batch_add_visible_marks(file_paths: list, output_dir: str, 
                          mark_text: Optional[str] = None,
                          **kwargs) -> list:
    """
    批量为文件添加可见标识
    
    Args:
        file_paths: 文件路径列表
        output_dir: 输出目录
        mark_text: 标识文本
        **kwargs: 其他参数
    
    Returns:
        输出文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for file_path in file_paths:
        try:
            file_path = Path(file_path)
            
            # 判断文件类型
            ext = file_path.suffix.lower()
            
            if ext in ['.txt', '.md']:
                # 文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                marked_content = add_text_mark_to_text(content, mark_text, **kwargs)
                output_path = output_dir / f"{file_path.stem}_marked{ext}"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(marked_content)
                    
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                # 图像文件
                img = Image.open(file_path)
                marked_img = add_overlay_to_image(img, mark_text, **kwargs)
                output_path = output_dir / f"{file_path.stem}_marked.png"
                marked_img.save(output_path)
                
            elif ext in ['.wav', '.mp3', '.flac', '.m4a']:
                # 音频文件
                output_path = output_dir / f"{file_path.stem}_marked.wav"
                add_voice_mark_to_audio(str(file_path), str(output_path), mark_text, **kwargs)
                
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # 视频文件
                output_path = output_dir / f"{file_path.stem}_marked.mp4"
                add_overlay_to_video_ffmpeg(str(file_path), str(output_path), mark_text, **kwargs)
                
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                continue
                
            results.append(str(output_path))
            logger.info(f"处理完成: {file_path} -> {output_path}")
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            continue
    
    return results