from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch, io, os, argparse, random
from vine.w_bench_utils.distortion.utils import set_random_seed, to_tensor, to_pil
from tqdm import tqdm


distortion_strength_paras = dict(
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(90, 10),
)


def relative_strength_to_absolute(strength, distortion_type):
    assert 0 <= strength <= 1
    strength = (
        strength
        * (
            distortion_strength_paras[distortion_type][1]
            - distortion_strength_paras[distortion_type][0]
        )
        + distortion_strength_paras[distortion_type][0]
    )
    strength = max(strength, min(*distortion_strength_paras[distortion_type]))
    strength = min(strength, max(*distortion_strength_paras[distortion_type]))
    return strength


def apply_distortion(
    images,
    distortion_type,
    strength=None,
    distortion_seed=0,
    same_operation=False,
    relative_strength=True,
    return_image=True,
):
    # Convert images to PIL images if they are tensors
    if not isinstance(images[0], Image.Image):
        images = to_pil(images)
    # Check if strength is relative and convert if needed
    if relative_strength:
        strength = relative_strength_to_absolute(strength, distortion_type)
    # Apply distortions
    distorted_images = []
    seed = distortion_seed
    for image in images:
        distorted_images.append(
            apply_single_distortion(
                image, distortion_type, strength, distortion_seed=seed
            )
        )
        # If not applying the same distortion, increment the seed
        if not same_operation:
            seed += 1
    # Convert to tensors if needed
    if not return_image:
        distorted_images = to_tensor(distorted_images)
    return distorted_images


def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0):
    # Accept a single image
    assert isinstance(image, Image.Image)
    # Set the random seed for the distortion if given
    set_random_seed(distortion_seed)
    # Assert distortion type is valid
    assert distortion_type in distortion_strength_paras.keys()
    # Assert strength is in the correct range
    if strength is not None:
        assert (
            min(*distortion_strength_paras[distortion_type])
            <= strength
            <= max(*distortion_strength_paras[distortion_type])
        )

    if distortion_type == "brightness":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["brightness"])
        )
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "contrast":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["contrast"])
        )
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = (
            int(strength)
            if strength is not None
            else random.uniform(*distortion_strength_paras["blurring"])
        )
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["noise"])
        )
        image = to_tensor([image], norm_type=None)
        noise = torch.randn(image.size()) * std
        distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["compression"])
        )
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)

    else:
        assert False

    return distorted_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_images_folder', type=str, default="./vine_encoded_wbench")
    parser.add_argument('--edited_output_folder', type=str, default="./output/distorted_wmed_wbench")
    args = parser.parse_args() 
    
    steps = {
        'brightness': (1.2, 1.4, 1.6, 1.8, 2.0),
        'contrast': (1.2, 1.4, 1.6, 1.8, 2.0),
        'blurring': (1, 3, 5, 7, 9),
        'noise': (0.02, 0.04, 0.06, 0.08, 0.1),
        'compression': (10, 25, 40, 55, 70)
    }

    for c in ['DISTORTION_1K']:
        for i in tqdm(os.listdir(os.path.join(args.wm_images_folder, '512', c))):
            path = os.path.join(args.wm_images_folder, '512', c, i)
            image = Image.open(path)
            #for distortion_type in distortion_strength_paras:
            for distortion_type in ['brightness', 'contrast', 'blurring', 'noise', 'compression']:
                for step in steps[distortion_type]:
                    save_path = os.path.join(args.edited_output_folder, '512', c, distortion_type, str(step))
                    os.makedirs(save_path, exist_ok=True)
                    image_edited = apply_single_distortion(image, distortion_type, strength=step)
                    image_edited = to_pil(to_tensor([image_edited], norm_type=None).clamp(0, 1), norm_type=None)[0]
                    assert np.min(np.array(image_edited)) >= 0 and np.max(np.array(image_edited)) <= 255
                    assert image_edited.size == (512, 512)
                    save_path = os.path.join(save_path, i)
                    print(save_path)
                    image_edited.save(save_path)