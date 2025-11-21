import torch.nn as nn
import numpy as np
import torch
import os
import torch.nn.functional as F
from saicinpainting.evaluation.masks.mask import propose_random_square_crop
from saicinpainting.evaluation.utils import load_yaml
from saicinpainting.training.data.masks import MixedMaskGenerator
import PIL.Image as Image
from torchvision.utils import save_image


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]
    

def get_random_irregular_mask(image, max_tamper_area, outdir='./output_img/', save=False):
    
    black_image = Image.new('RGB', (image.shape[2], image.shape[3]), (0, 0, 0))
    batch_mask = []
    if image.shape[2] == 512:
        config_path = ['./config_mask/random_medium_512.yaml',
                        './config_mask/random_thick_512.yaml',
                        './config_mask/random_thin_512.yaml',]
    elif image.shape[2] == 256:
        config_path = ['./config_mask/random_medium_256.yaml',
                        './config_mask/random_thick_256.yaml',
                        './config_mask/random_thin_256.yaml',]
    else:
        raise ValueError('Invalid Image Size') 
    
    for i in range(image.shape[0]):
        random_config = np.random.randint(0, len(config_path))
        config = load_yaml(config_path[random_config])
        
        variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
        mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
                                            variants_n=variants_n)

        while True:
            src_masks = mask_generator.get_masks(black_image)

            filtered_image_mask_pairs = []
            # max_tamper_area = config.get('max_tamper_area', 1)
            for cur_mask in src_masks:
                if config.cropping.out_square_crop:
                    (crop_left,
                        crop_top,
                        crop_right,
                        crop_bottom) = propose_random_square_crop(cur_mask,
                                                                min_overlap=config.cropping.crop_min_overlap)
                    cur_mask = cur_mask[crop_top:crop_bottom, crop_left:crop_right]
                    # cur_image = black_image.copy().crop((crop_left, crop_top, crop_right, crop_bottom))
                # else:
                    # cur_image = black_image

                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue

                filtered_image_mask_pairs.append(cur_mask)
            
            if len(filtered_image_mask_pairs) > 0:
                break

        mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                        size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                        replace=False)

        # crop masks; save masks together with input image
        if save:
            mask_basename = outdir
            for i, idx in enumerate(mask_indices):
                cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename + f'_crop{i:03d}'
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + f'_mask{i:03d}.png')
        
        batch_mask.append(torch.tensor(filtered_image_mask_pairs[0]))
    
    stacked_batch_mask = torch.stack(batch_mask)
    stacked_batch_mask = stacked_batch_mask.unsqueeze(1)
    
    return stacked_batch_mask
                

def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[1]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)

        noised_and_cover[0] = noised_image[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()

        return noised_and_cover
    
    
class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range=(0.8, 0.9), width_ratio_range=(0.8, 0.9)):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, noised_and_cover, max_tamper_area=0.5, height_ratio_range=(0.8, 0.9), width_ratio_range=(0.8, 0.9)):
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        a = torch.rand(1)[0]
        if a <= 0.3:
            cropout_mask = torch.zeros_like(noised_image)
            h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                        height_ratio_range=self.height_ratio_range,
                                                                        width_ratio_range=self.width_ratio_range)
            cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
            cropout_mask = 1 - cropout_mask
            cropout_mask = cropout_mask.to(noised_image.device)
        else:
            cropout_mask = get_random_irregular_mask(noised_image, max_tamper_area=max_tamper_area)
            cropout_mask = cropout_mask.to(noised_image.device)
                
        # noised_and_cover[0] = noised_image * cropout_mask + cover_image * (1-cropout_mask)
        # noised_and_cover[0] = noised_image * (1 - cropout_mask)
        noised_and_cover[0] = noised_image * (1 - cropout_mask) + cover_image * cropout_mask
                
        # save_image(noised_and_cover[0], 'output.png')

        return  noised_and_cover
    

class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self, keep_ratio_range=[0.6, 0.9]):
        super(Dropout, self).__init__()
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]


    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        mask_percent = np.random.uniform(self.keep_min, self.keep_max)

        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        # mask_tensor.unsqueeze_(0)
        # mask_tensor.unsqueeze_(0)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + cover_image * (1-mask_tensor)
        return [noised_image, cover_image]


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_range=[0.5, 2.0], interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = ['nearest', 'bilinear', 'bicubic']


    def forward(self, noised_image, resize_ratio_min=None, resize_ratio_max=None):
        
        if resize_ratio_min is not None:
            self.resize_ratio_min = resize_ratio_min
            self.resize_ratio_max = resize_ratio_max
        
        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)

        origal_size = noised_image.shape[2:]
        random_number1 = np.random.randint(0, len(self.interpolation_method))
        random_number2 = np.random.randint(0, len(self.interpolation_method))
        resized_img = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method[random_number1],
                                )   

        recovered_img = F.interpolate(
                                    resized_img,
                                    size=(origal_size[0], origal_size[1]),
                                    mode=self.interpolation_method[random_number2],
                                )
                
        return recovered_img