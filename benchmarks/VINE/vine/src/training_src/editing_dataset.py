import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import vine.src.training_src.extra_utils as extra_utils
import torch.nn.functional as F
from PIL import ImageFile
from datasets import load_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

class EditData(Dataset):
    def __init__(self, data_path, secret_size=100, size=(512, 512)):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.files_list = load_dataset("clip-filtered-dataset", cache_dir="./dataset/clip-filtered-dataset")
        self.t_256 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.ToTensor(),
        ])
        self.to_tensor = transforms.ToTensor()
        
    def __getitem__(self, idx):
        img_cover = self.files_list[idx]['image']
        img_cover_256 = self.t_256(img_cover)
        
        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover)
        
        img_cover = 2.0 * img_cover - 1.0
        img_cover_256 = 2.0 * img_cover_256 - 1.0
        
        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()

        prompt = self.files_list[idx]['prompt']
        return {"cover_img": img_cover, "cover_img_256": img_cover_256, "secret": secret, "prompt": prompt}

    def __len__(self):
        return self.files_list.num_rows


def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params