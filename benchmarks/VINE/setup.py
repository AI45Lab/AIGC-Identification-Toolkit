from setuptools import setup, find_packages
import os

setup(
    name="vine",                       # Package name, consistent with the project name
    version="0.1.0",                   # Initial version number
    description="A Python package for vine project",  # Short description
    long_description=open("README.md").read() if os.path.exists("README.md") else "",  # Optional, long description from README
    long_description_content_type="text/markdown",  # If README is in Markdown format
    author="Shilin Lu",               
    author_email="shilin002@e.ntu.edu.sg",  
    url="https://github.com/Shilin-LU/VINE",  
    packages=find_packages(),          # Automatically discover packages (assumes code is in vine/ directory)
    install_requires=[
        # Dependencies extracted from environment.yaml
        "einops==0.8.0",
        "numpy==1.26.4",
        "open-clip-torch==2.26.1",
        "opencv-python==4.6.0.66",
        "pillow==10.4.0",
        "scipy==1.11.1",
        "timm>=0.9.2",
        "tokenizers==0.19.1",
        "torch==2.0.1",
        "torchdata==0.6.1",
        "torchmetrics==0.2.0",
        "torchvision==0.15.2",
        "tqdm>=4.65.0",
        "triton==2.0.0",
        "urllib3==1.26.19",
        "xformers==0.0.20",
        "streamlit-keyup==0.2.0",
        "lpips",
        "clean-fid",
        "peft==0.11.1",
        "dominate",
        "gradio==3.43.1",
        "transformers==4.43.3",
        "accelerate==0.30.0",
        "datasets==2.20.0",
        "pilgram",
        "kornia",
        "wandb",
        "scikit-image",
        "scikit-learn",
        "ipykernel",
        "sentencepiece",
        "pytorch-lightning==1.2.9",
        "bchlib==0.7",
        "easydict",
        "clip @ git+https://github.com/openai/CLIP.git",
        # Install diffusers from the VINE repository's diffusers subdirectory
        "diffusers @ git+https://github.com/Shilin-LU/VINE.git#subdirectory=diffusers",
    ],
    python_requires=">=3.10",          # Specify Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,         # Include non-code files (e.g., data files) if any
)