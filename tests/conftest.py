import sys
import os
from pathlib import Path


def setup_project_path():
    """设置项目路径，确保可以导入 src 模块"""
    
    # 方法1：从当前文件位置推断项目根目录
    current_file = Path(__file__).resolve()
    
    # 向上查找项目根目录的标志文件
    project_indicators = ['src', 'CLAUDE.md', 'requirements.txt', '.git']
    
    for parent in [current_file.parent.parent, current_file.parent, current_file.parent.parent.parent]:
        if any((parent / indicator).exists() for indicator in project_indicators):
            project_root = parent
            break
    else:
        # 如果找不到，使用默认位置
        project_root = current_file.parent.parent
    
    src_dir = project_root / "src"
    
    if src_dir.exists():
        # 确保 src 目录在 Python 路径的最前面
        src_path = str(src_dir)
        if src_path in sys.path:
            sys.path.remove(src_path)
        sys.path.insert(0, src_path)
        
        print(f"已添加 src 目录到 Python 路径: {src_path}")
        return True
    else:
        print(f"警告: 找不到 src 目录在 {project_root}")
        return False


# 设置项目路径
setup_project_path()


