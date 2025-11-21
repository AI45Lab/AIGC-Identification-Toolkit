import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import vine.src.training_src.extra_utils as extra_utils
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100, size=(512, 512)):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.files_list = []
        self.files_list += glob(os.path.join(self.data_path, '*.jpg'))
        self.files_list += glob(os.path.join('/export/home2/shilin/Datasets/OpenImage/test', '*.jpg'))
        self.files_list += glob(os.path.join('/export/home2/shilin/Datasets/OpenImage/validation', '*.jpg'))
        self.files_list += glob(os.path.join('/export/home2/shilin/Datasets/OpenImage/train', '*.jpg'))
        
        self.transform_list = []
        
        self.transform1 = transforms.Compose([
            transforms.Resize(self.size[0], interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(self.size[0]),
            transforms.ToTensor(),
        ])
        self.transform_list.append(self.transform1)
        
        self.transform2 = transforms.Compose([
            # transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(self.size[0]),
            transforms.ToTensor(),
        ])
        self.transform_list.append(self.transform2)
        self.to_tensor = transforms.ToTensor()
        
    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]

        img_cover = Image.open(img_cover_path).convert('RGB')
        h, w = img_cover.size
        if min(h, w) < self.size[0]:
            img_cover = self.transform1(img_cover)
        else:        
            weights = [0.3, 0.7]
            sampled_trans = random.choices(self.transform_list, weights)[0]
            img_cover = sampled_trans(img_cover)
        # img_cover = ImageOps.fit(img_cover, self.size)
        # img_cover = self.to_tensor(img_cover)
        
        img_cover = 2.0 * img_cover - 1.0
        # img_cover = Ft.normalize(img_cover, mean=[0.5], std=[0.5]) ### TODO:check this
        
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.

        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()

        return {"cover_img": img_cover, "secret": secret,}

    def __len__(self):
        return len(self.files_list)


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