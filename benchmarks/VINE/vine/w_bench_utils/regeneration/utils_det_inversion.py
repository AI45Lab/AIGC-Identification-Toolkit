from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
import torch
import os
from typing import Union, Tuple, Optional
from PIL import Image
from torchvision import transforms
import numpy as np
repo = 'stabilityai/stable-diffusion-2-1-base'


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None: 
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return transforms.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.   # x.shape: torch.Size([8,3,512,512])
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215  # latents: torch.Size([8,4,64,64])
    return latents

def latent2image(latents: torch.Tensor, vae: AutoencoderKL):
    latents = 1 / 0.18215 * latents 
    image = vae.decode(latents)['sample']    
    image_tensor = image/2.0 + 0.5  
    return image_tensor


def container_inversion(input_img, pipe, inv_type = 'ddim', dpm_order = 2, num_steps=50, batch_size=8):  
    # input_img.requires_grad: True
    if inv_type == 'dpm':
        image_to_latent_schedule = DPMSolverMultistepInverseScheduler.from_pretrained(repo, subfolder='scheduler', solver_order=dpm_order)
        latent_to_image_schedule = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder='scheduler', solver_order=dpm_order)

    elif inv_type == 'ddim':
        image_to_latent_schedule = DDIMInverseScheduler.from_pretrained(repo, subfolder='scheduler')
        latent_to_image_schedule = DDIMScheduler.from_pretrained(repo, subfolder='scheduler')
    else:
        raise ValueError(f'Unknown inversion type {inv_type}')

    # image to latents
    pipe.scheduler = image_to_latent_schedule
    vae = pipe.vae
    latents = img_to_latents(input_img, vae)  # latents.requires_grad: True

    inv_latents, _ = pipe(prompt=[""]*batch_size, negative_prompt=[""]*batch_size, guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)
    

    print(type(inv_latents))

    # latents to image
    pipe.scheduler = latent_to_image_schedule
    # image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
    #              num_inference_steps=num_steps, latents=inv_latents)
    output_latents = pipe(prompt=[""]*batch_size, negative_prompt=[""]*batch_size, guidance_scale=1., output_type='latent',
            num_inference_steps=num_steps, latents=inv_latents)
    
    output_image_tensor = latent2image(output_latents[0], vae) 
    # cv2.imwrite(f"lala.png", cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
    return output_image_tensor 



if __name__ == "__main__":
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--images_folder_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--inv_type', type=str, default='ddim')
    parser.add_argument('--dpm_order', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    folder_path=args.images_folder_path
    output_folder=args.output_path
    inv_type=args.inv_type
    dpm_order=args.dpm_order
    num_steps=args.num_steps
    batch_size=args.batch_size

    dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ldm_pipe = StableDiffusionPipeline.from_pretrained(repo, safety_checker=None, torch_dtype=dtype)
    ldm_pipe.to(device)


    for parameter in ldm_pipe.unet.parameters():
        parameter.requires_grad=False
    for parameter in ldm_pipe.vae.parameters():
        parameter.requires_grad=False
    for parameter in ldm_pipe.text_encoder.parameters():
        parameter.requires_grad=False

    images = []
    for num in range(batch_size):
        img_lists=sorted(os.listdir(folder_path))[:batch_size]
        # first_element = img_lists.pop(0)
        # img_lists.append(first_element)

        img_name=img_lists[num]
        img_path= folder_path+'/'+img_name
        print(img_path)
        img_tensor = load_image(img_path)  #[1,3,512,512] (0-1)
        images.append(img_tensor)

    image_tensors = torch.cat(images, dim=0)  # image_tensors: torch.Size([8,3,512,512])
    image_tensors = image_tensors.to(device=device, dtype=dtype)


    #  ***
    image_tensors.requires_grad=True
    image_tensors=image_tensors.half()
    output_image_tensors = container_inversion(image_tensors, ldm_pipe, inv_type = inv_type, dpm_order = dpm_order, num_steps= num_steps, batch_size=batch_size)
    print(output_image_tensors.requires_grad)


    # ***
    output_image_tensors=output_image_tensors.clamp(0,1)
   
    numpy_images = output_image_tensors.cpu().permute(0, 2, 3, 1).detach().numpy()
    numpy_images  = (numpy_images  * 255).astype(np.uint8)

    for i, image_array in enumerate(numpy_images):
        img = Image.fromarray(image_array, 'RGB')
        img.save(output_folder + '/' + f'image_{i}.png')