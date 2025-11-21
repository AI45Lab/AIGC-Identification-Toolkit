import os, torch, PIL.Image, argparse
import pandas as pd
from diffusers import StableDiffusion3InstructPix2PixPipeline, DDIMScheduler
from diffusers.utils import load_image
from tqdm import tqdm


def edit_by_UltraEdit(device, guidance, inputPath_img, inputPath_msk, inputPath_prmt, outputPath):
    # Model Preparation
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask",
                                                                   torch_dtype=torch.float16,
                                                                   safety_checker=None,
                                                                   requires_safety_checker=False)
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
            prompt=prompt,
            image=image,
            mask_img=mask,
            negative_prompt="",
            num_inference_steps=50,     
            image_guidance_scale=1.5,
            guidance_scale=guidance,
        ).images[0]

        path = outputPath + f"{str(idx)}_{str(ID[idx])}.png"
        print(f"\t> Edited image {str(idx)}_{str(ID[idx])} is saved at: `{path}`")
        image.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_images_folder", type=str, default='./vine_encoded_wbench/512/LOCAL_EDITING_5K')
    parser.add_argument("--wbench_path", type=str, default='./W-Bench/LOCAL_EDITING_5K')
    parser.add_argument("--edited_output_folder", type=str, default='./output/edited_wmed_wbench')
    args = parser.parse_args()
    
    MODE = "REGION"
    SPEC = "_UltraEdit"

# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    DEVICE = 'cuda:0'  
    GUIDANCE = 7.5
    CHOICES = ['10-20', '20-30', '30-40', '40-50', '50-60']

    for CHOICE in CHOICES:
        print(f"\n\n>> Currently processing the choice of {CHOICE}...\n")
        INPUT_PATH_IMAGE = os.path.join(args.wm_images_folder, f"{CHOICE}")   
        INPUT_PATH_MASK = os.path.join(args.wbench_path, f"{CHOICE}/mask")  
        INPUT_PATH_PROMPT = os.path.join(args.wbench_path, f"{CHOICE}/prompts.csv")   
        OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{MODE}{SPEC}/{CHOICE}/")  
        os.makedirs(OUTPUT_PATH, exist_ok=True)
# TODO ---------------------------------------- DASHBOARD ENDS ------------------------------------------------------------

        print(f"\n>> Processing edited images for [{MODE}], with CHOICE={CHOICE}, on DEVICE={DEVICE}...")
        edit_by_UltraEdit(
            device=DEVICE,
            guidance=GUIDANCE,
            inputPath_img=INPUT_PATH_IMAGE,
            inputPath_msk=INPUT_PATH_MASK,
            inputPath_prmt=INPUT_PATH_PROMPT,
            outputPath=OUTPUT_PATH
        )