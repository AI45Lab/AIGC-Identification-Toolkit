import os, argparse, torch
import pandas as pd
from tqdm import tqdm
from diffusers.utils import load_image
from diffusers import StableDiffusionInstructPix2PixPipeline, DDIMScheduler


def edit_by_InstructPix2Pix(device, guidance, inputPath_img, inputPath_prmt, outputPath):
    # Model Preparation
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                  torch_dtype=torch.float16,
                                                                  safety_checker=None,
                                                                  requires_safety_checker=False)
    pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Acquire Data and Process Editing:
    os.makedirs(outputPath, exist_ok=True)
    ID = pd.read_csv(inputPath_prmt).iloc[:, 1].tolist()
    for idx, prompt in tqdm(enumerate(pd.read_csv(inputPath_prmt).iloc[:, 2].tolist())):
        image = load_image(os.path.join(inputPath_img, f"{str(idx)}_{str(ID[idx])}_wm.png")).resize((512, 512))

        image = pipe(
            prompt=prompt,
            image=image,
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
    SPEC = "_Pix2Pix"

# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    GUIDANCE_RANGE = [5, 6, 7, 8, 9]
    DEVICE = 'cuda'

    for GUIDANCE in GUIDANCE_RANGE:
        print(f"\n\n>> Currently processing the CHOICE of {GUIDANCE}...\n")
        OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{MODE}{SPEC}/{GUIDANCE}/") 
        os.makedirs(OUTPUT_PATH, exist_ok=True)
# TODO ---------------------------------------- DASHBOARD ENDS ------------------------------------------------------------

        print(f"\n>> Processing edited images for [{MODE}], with GUIDANCE={GUIDANCE}, on DEVICE={DEVICE}...")
        edit_by_InstructPix2Pix(
            device=DEVICE,
            guidance=GUIDANCE,
            inputPath_img=args.wm_images_folder,
            inputPath_prmt=args.editing_prompt_path,
            outputPath=OUTPUT_PATH
        )
        """ PROCESS END """


