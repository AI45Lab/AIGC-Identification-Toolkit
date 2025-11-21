from PIL import Image, ImageEnhance
import numpy as np
import cv2, os, torch
from skimage.util import random_noise
from tqdm import tqdm


class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError


class GaussianBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = cv2.imread(img_path)
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
            cv2.imwrite(out_path, img)


class GaussianNoiseAttacker(WMAttacker):
    def __init__(self, std):
        self.std = std

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            image = cv2.imread(img_path)
            image = image / 255.0
            # Add Gaussian noise to the image
            noise_sigma = self.std  # Vary this to change the amount of noise
            noisy_image = random_noise(image, mode='gaussian', var=noise_sigma ** 2)
            # Clip the values to [0, 1] range after adding the noise
            noisy_image = np.clip(noisy_image, 0, 1)
            noisy_image = np.array(255 * noisy_image, dtype='uint8')
            cv2.imwrite(out_path, noisy_image)


class JPEGAttacker(WMAttacker):
    def __init__(self, quality=80):
        self.quality = quality

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            img.save(out_path, "JPEG", quality=self.quality)


class BrightnessAttacker(WMAttacker):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)
            img.save(out_path)


class ContrastAttacker(WMAttacker):
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)
            img.save(out_path)


class RotateAttacker(WMAttacker):
    def __init__(self, degree=30):
        self.degree = degree

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            img = img.rotate(self.degree)
            img.save(out_path)


class ScaleAttacker(WMAttacker):
    def __init__(self, scale=0.5):
        self.scale = scale

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            w, h = img.size
            img = img.resize((int(w * self.scale), int(h * self.scale)))
            img.save(out_path)


class CropAttacker(WMAttacker):
    def __init__(self, crop_size=0.5):
        self.crop_size = crop_size

    def attack(self, image_paths, out_paths):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            img = Image.open(img_path)
            w, h = img.size
            img = img.crop((int(w * self.crop_size), int(h * self.crop_size), w, h))
            img.save(out_path)


class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)  #latents: torch.size([5,4,64,64])
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents,
                                   head_start_step=50 - max(self.noise_step // 20, 1), 
                                   guidance_scale=7.5,
                                   generator=generator, )

                images = images[0]
                for img, out in zip(images, outs_buf):
                    img.save(out)

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                img = Image.open(img_path)
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2 
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor    # latents: torch.Size([1, 4, 64, 64])
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)  #noise: torch.Size([1, 4, 64, 64])
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)  #timestep: tensor([60])
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents



from typing import Callable, List, Optional, Union
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ReSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            prompt1_steps: Optional[int] = None,
            prompt2: Optional[str] = None,
            head_start_latents: Optional[Union[torch.FloatTensor, list]] = None,
            head_start_step: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # import pdb; pdb.set_trace()
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        if prompt2 is not None:
            text_embeddings2 = self._encode_prompt(
                prompt2, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # print(timesteps)

        # 5. Prepare latent variables
        if head_start_latents is None:
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
        else:
            if type(head_start_latents) == list:
                latents = head_start_latents[-1]
                assert len(head_start_latents) == self.scheduler.config.solver_order

            else:
                latents = head_start_latents  # if there is a head start
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # print((i, t))
                if not head_start_step or i >= head_start_step:  # if there is no head start or we reached the hs step
                    # expand the latents if we are doing classifier free guidance
                    # print(latents.shape)
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    if prompt1_steps is None or i < prompt1_steps:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    else:
                        # print(f'i = {i}, atteding to prompt2')
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings2).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        has_nsfw_concept = False
        # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)