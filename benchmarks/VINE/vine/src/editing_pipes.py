import torch
from typing import Union, Tuple, Optional
import torch
from PIL import Image
from diffusers import DDIMInverseScheduler, AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
from torchvision import transforms as tvt
from PIL import Image 


def edit_by_UltraEdit(pipe, image, prompt, text_guidance, num_inference_steps):
    
    mask = Image.new("RGB", (512, 512), (255, 255, 255))
    image = pipe(
        prompt=prompt,
        image=image,
        mask_img=mask,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        image_guidance_scale=1.5,
        guidance_scale=text_guidance,
        generator=torch.manual_seed(42),
    ).images[0]

    return image


def edit_by_InstructPix2Pix(pipe, image, prompt, guidance, num_inference_steps=50):
    
    image = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=1.5,
        guidance_scale=guidance,
        generator=torch.manual_seed(42),
    ).images[0]

    return image
        

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size)
        # pil_img.save('temp1.png')
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.no_grad()
def ddim_inversion(pipe, imgname, repo, device, inv_type = 'ddim', dpm_order = 2, num_steps: int = 50, verify: Optional[bool] = False,
                   prompt1 = None, prompt2 = None) -> torch.Tensor:

    if isinstance(imgname, str):
        input_img = load_image(imgname, target_size=512).to(device=device)
    else:
        input_img = imgname.to(device=device)
    
    if inv_type == 'dpm':
        image_to_latent_schedule = DPMSolverMultistepInverseScheduler.from_pretrained(repo, subfolder='scheduler', solver_order=dpm_order)
        latent_to_image_schedule = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder='scheduler', solver_order=dpm_order)
        # print(image_to_latent_schedule)
    elif inv_type == 'ddim':
        image_to_latent_schedule = DDIMInverseScheduler.from_pretrained(repo, subfolder='scheduler')
        latent_to_image_schedule = DDIMScheduler.from_pretrained(repo, subfolder='scheduler')
    else:
        raise ValueError(f'Unknown inversion type {inv_type}')
    
    # image to latents
    pipe.scheduler = image_to_latent_schedule
    vae = pipe.vae
    latents = img_to_latents(input_img, vae)

    inv_latents, _ = pipe(prompt=prompt1, negative_prompt="", guidance_scale=1.0,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)
                
    # latents to image
    if verify:
        # image_edited = pipe.vae.decode(inv_latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        # image_edited = pipe.image_processor.postprocess(image_edited)
        pipe.scheduler = latent_to_image_schedule
        image = pipe(prompt=prompt2, negative_prompt="", guidance_scale=1.0,
                     num_inference_steps=num_steps, latents=inv_latents).images[0]
        
        return image