import os, argparse, torch
import pandas as pd
import PIL.Image
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
    parser.add_argument("--wm_images_folder", type=str, default='./vine_encoded_wbench/512/INSTRUCT_1K')
    parser.add_argument("--editing_prompt_path", type=str, default='./W-Bench/INSTRUCT_1K/prompts.csv')
    parser.add_argument("--edited_output_folder", type=str, default='./output/edited_wmed_wbench')
    args = parser.parse_args()
        
    MODE = "INSTRUCT"
    SPEC = "_UltraEdit"

# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    for CHOICE in [5, 6, 7, 8, 9]:
        DEVICE = 'cuda:0' 

        print(f"\n\n>> Currently processing the CHOICE of {CHOICE}...\n")
        OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{MODE}{SPEC}/{CHOICE}/")  
        os.makedirs(OUTPUT_PATH, exist_ok=True)
# TODO ---------------------------------------- DASHBOARD ENDS ------------------------------------------------------------

        print(f"\n>> Processing edited images for [{MODE}], with CHOICE={CHOICE}, on DEVICE={DEVICE}...")
        edit_by_UltraEdit(
            device=DEVICE,
            guidance=CHOICE,
            inputPath_img=args.wm_images_folder,
            inputPath_msk=None,
            inputPath_prmt=args.editing_prompt_path,
            outputPath=OUTPUT_PATH
        )
