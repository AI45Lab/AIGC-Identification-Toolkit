import os, torch, PIL.Image, argparse
import numpy as np
import pandas as pd
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from tqdm import tqdm


def edit_by_cnInpaint(device, inputPath_img, inputPath_msk, inputPath_prmt, outputPath):
    # Model Preparation
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",
                                                 torch_dtype=torch.float16,
                                                 safety_checker=None,
                                                 requires_safety_checker=False)
    controlnet.to(device)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                    controlnet=controlnet,
                                                                    torch_dtype=torch.float16,
                                                                    safety_checker=None,
                                                                    requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Acquire Data and Process Editing:
    os.makedirs(outputPath, exist_ok=True)
    ID = pd.read_csv(inputPath_prmt).iloc[:, 1].tolist()
    for idx, prompt in tqdm(enumerate(pd.read_csv(inputPath_prmt).iloc[:, 2].tolist())):
        image = load_image(os.path.join(inputPath_img, f"{str(idx)}_{str(ID[idx])}_wm.png")).resize((512, 512))
        if inputPath_msk is not None:
            mask = load_image(os.path.join(inputPath_msk, f"{str(idx)}_{str(ID[idx])}.png")).resize(image.size)
        else:
            mask = PIL.Image.new("RGB", image.size, (255, 255, 255))

        image = pipe(
            prompt,
            num_inference_steps=50,
            generator=torch.Generator(device=device).manual_seed(1),
            eta=1.0,
            image=image,
            mask_image=mask,
            control_image=_make_inpaint_condition(image, mask).to(device),
        ).images[0]

        # Edited images are saved to `samples/{SAMPLE_NUM}/edited/`
        path = outputPath + f"{str(idx)}_{str(ID[idx])}.png"
        print(f"\t> Edited image {str(idx)}_{str(ID[idx])} is saved at: `{path}`")
        image.save(path)


def _make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_images_folder", type=str, default='./vine_encoded_wbench/512/LOCAL_EDITING_5K')
    parser.add_argument("--wbench_path", type=str, default='./W-Bench/LOCAL_EDITING_5K')
    parser.add_argument("--edited_output_folder", type=str, default='./output/edited_wmed_wbench')
    args = parser.parse_args()
    
    MODE = "REGION"
    SPEC = "_cnInpaint"

# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    DEVICE = "cuda:0"
    CHOICES = ['10-20', '20-30', '30-40', '40-50', '50-60']

    for CHOICE in CHOICES:
        print(f"\n\n>> Currently processing the choice of {CHOICE}...\n")
        INPUT_PATH_IMAGE = os.path.join(args.wm_images_folder, f"{CHOICE}")   
        INPUT_PATH_MASK = os.path.join(args.wbench_path, f"{CHOICE}/mask")   
        INPUT_PATH_PROMPT = os.path.join(args.wbench_path, f"{CHOICE}/prompts.csv")   
        OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{MODE}{SPEC}/{CHOICE}/")  
        os.makedirs(OUTPUT_PATH, exist_ok=True)
# TODO ---------------------------------------- DASHBOARD ENDS ------------------------------------------------------------

        print(f"\n>> Processing edited images for [{MODE}{SPEC}] on DEVICE={DEVICE}...")
        edit_by_cnInpaint(
            device=DEVICE,
            inputPath_img=INPUT_PATH_IMAGE,
            inputPath_msk=INPUT_PATH_MASK,
            inputPath_prmt=INPUT_PATH_PROMPT,
            outputPath=OUTPUT_PATH
        )