import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for dataset and dataloader options
    # parser.add_argument("--dataset_folder", default="data/my_horse2zebra", type=str)
    parser.add_argument("--dataset_folder", default="/export/home2/shilin/Datasets/mirflickr", type=str)
    parser.add_argument("--val_folder", default="/export/home2/shilin/Datasets/val_set", type=str)
    parser.add_argument("--train_img_prep", default="resize_286_randomcrop_256x256_hflip", type=str)
    parser.add_argument("--val_img_prep", default="no_resize", type=str)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=14000000)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=1, type=int)
    parser.add_argument("--lora_rank_vae", default=1, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=1e+9)
    parser.add_argument("--output_dir", type=str, default="output/test", required=False)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="test")
    parser.add_argument("--key_change", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default='logs')
    parser.add_argument("--validation_steps", type=int, default=1e+9,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`"
        ),
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=10, help=("Max number of checkpoints to store."),)
    
    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=1e-4,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_step_rules", type=str, default="1:10000,0.1", help="")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--secret_size", type=int, default=100,)
    parser.add_argument("--fixed_input", action="store_true",)
    parser.add_argument('--l2_loss_scale', type=float, default=2.0)
    parser.add_argument('--l2_loss_ramp', type=int, default=100000)
    parser.add_argument('--l2_edge_gain', type=float, default=10.0)
    parser.add_argument('--l2_edge_ramp', type=int, default=100000)
    parser.add_argument('--l2_edge_delay', type=int, default=20000000000)
    parser.add_argument('--lpips_loss_scale', type=float, default=1.5)
    parser.add_argument('--lpips_loss_ramp', type=int, default=100000)
    parser.add_argument('--secret_loss_scale', type=float, default=1.5)
    parser.add_argument('--G_loss_scale', type=float, default=0.5)
    parser.add_argument('--G_loss_ramp', type=int, default=100000)
    parser.add_argument('--borders', type=str, choices=['black','random','randomrgb','image','white'], default='black')
    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=100.0)
    parser.add_argument('--v_scale', type=float, default=100.0)
    parser.add_argument('--no_gan', action='store_true')
    parser.add_argument('--rnd_trans', type=float, default=.1)
    parser.add_argument('--rnd_bri', type=float, default=.3)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--rnd_sat', type=float, default=1.0)
    parser.add_argument('--rnd_hue', type=float, default=.1)
    parser.add_argument('--contrast_low', type=float, default=.5)
    parser.add_argument('--contrast_high', type=float, default=1.5)
    parser.add_argument('--jpeg_quality', type=float, default=50)
    parser.add_argument('--no_jpeg', action='store_true')
    parser.add_argument('--rnd_trans_ramp', type=int, default=10000)
    parser.add_argument('--rnd_bri_ramp', type=int, default=10000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=10000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=10000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=10000)
    parser.add_argument('--contrast_ramp', type=int, default=10000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=10000)
    parser.add_argument('--no_im_loss_steps', help="Train without image loss for first x steps", type=int, default=1000)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--no_perspective', action='store_true')
    parser.add_argument('--no_motionBlur', action='store_true')
    parser.add_argument('--N_blur', type=int, default=7)
    parser.add_argument('--no_gaussianNoise', action='store_true')
    parser.add_argument('--no_contrast', action='store_true')
    parser.add_argument('--no_bright', action='store_true')
    parser.add_argument('--no_saturation', action='store_true')
    parser.add_argument('--no_falloff_im', action='store_true')  #不使用图片边缘权重调整
    parser.add_argument('--use_rgb_Imageloss', action='store_true')
    parser.add_argument('--ig_filter_step', type=int, default=50000000)
    parser.add_argument('--crop_resize_step', type=int, default=500000000)
    parser.add_argument('--imagenetc_step', type=int, default=5000)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--cuda', default= True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ic_up_level_interval', type=int, default=10000)
    
    args = parser.parse_args()
    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
