from utils_det_inversion import container_inversion, load_image
from diffusers import StableDiffusionPipeline
import torch, argparse, os
from PIL import Image
from tqdm import tqdm
import numpy as np
repo = 'stabilityai/stable-diffusion-2-1-base'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_images_folder', type=str, default="./vine_encoded_wbench/512/DET_INVERSION_1K")
    parser.add_argument('--edited_output_folder', type=str, default="./output/edited_wmed_wbench/DET_INVERSION_1K")
    parser.add_argument('--inv_type', type=str, default='dpm')
    parser.add_argument('--dpm_order', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args() 

    for STEPS in [15, 25, 35, 45]:
        DEVICE = 'cuda'      
        OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{STEPS}")
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        inv_type = args.inv_type
        dpm_order = args.dpm_order
        num_steps = STEPS
        batch_size = args.batch_size

        dtype = torch.float32
        device = DEVICE
        ldm_pipe = StableDiffusionPipeline.from_pretrained(repo, safety_checker=None, torch_dtype=dtype)
        ldm_pipe.to(device)

        files = [file for file in os.listdir(args.wm_images_folder) if file.endswith(".png")]
        
        with torch.no_grad():
            for file in tqdm(files):
                img_tensor = load_image(os.path.join(args.wm_images_folder, file))
                img_tensors = img_tensor.to(device=device, dtype=dtype)

                output_image_tensors = container_inversion(img_tensors, ldm_pipe, inv_type=inv_type, dpm_order=dpm_order, num_steps=num_steps, batch_size=1)
                # ***
                output_image_tensors = output_image_tensors.clamp(0,1)

                numpy_image = output_image_tensors.cpu().permute(0, 2, 3, 1).detach().numpy()
                numpy_image = (numpy_image * 255).astype(np.uint8)

                img = Image.fromarray(numpy_image[0], 'RGB')
                img.save(os.path.join(OUTPUT_PATH, file))
