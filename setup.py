from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="AIGC-Identification-Toolkit",
    version="0.1.0",
    author="码农团队",
    author_email="",
    description="AIGC内容标识开发套件 - 支持文本、图像、音频和视频的水印和显式标识",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
        ],
        "audio": [
            "bark @ git+https://github.com/suno-ai/bark.git",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "watermark-tool=src.unified.watermark_tool:main",
        ],
    },
) 