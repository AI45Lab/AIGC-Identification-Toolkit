from __future__ import annotations
import math
import os
import random
import einops
import k_diffusion as K
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "./instruct-pix2pix/"))
sys.path.append(os.path.join(BASE_DIR, "./instruct-pix2pix/stable_diffusion/"))
from tqdm import tqdm
from argparse import ArgumentParser
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--config", default="./w_bench_utils/global_editing/instruct-pix2pix/configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default='./w_bench_utils/global_editing/instruct-pix2pix/checkpoints/MagicBrush-epoch-52-step-4999.ckpt', type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", type=str)            # todo: input_path_image: path4img
    parser.add_argument("--output", type=str)           # todo: output_path_image: path4img
    parser.add_argument("--edit", type=str)             # todo: prompt: str
    parser.add_argument("--cfg-text", type=float)       # todo: txt_g: int, {5, 6, 7, 8, 9}
    parser.add_argument("--cfg-image", type=float)      # todo: img_g: int, 1.5
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wm_images_folder", type=str, default='./vine_encoded_wbench/512/INSTRUCT_1K')
    parser.add_argument("--editing_prompt_path", type=str, default='./W-Bench/INSTRUCT_1K/prompts.csv')
    parser.add_argument("--edited_output_folder", type=str, default='./output/edited_wmed_wbench')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    seed = random.randint(0, 100000) if args.seed is None else args.seed

    """ DASHBOARD START """
    # todo - conda activate ip2p
    # todo - python .py

    cfg_text_range = [5, 6, 7, 8, 9]

    MODE = "INSTRUCT"
    SPEC = "_MagicBrush"

    for i in cfg_text_range:
        args.cfg_text = i
        args.cfg_image = 1.5
        OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{MODE}{SPEC}/{cfg_text_range[cfg_text_range.index(args.cfg_text)]}")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        print(f"\n>> Processing edited images for [{MODE}{SPEC}], with GUIDANCE of [{args.cfg_text}][{args.cfg_image}]...")

        IDs = pd.read_csv(args.editing_prompt_path).iloc[:, 1].tolist()
        for idx, prompt in tqdm(enumerate(pd.read_csv(args.editing_prompt_path).iloc[:, 2].tolist())):
            args.edit = prompt
            args.input = os.path.join(args.wm_images_folder, f"{str(idx)}_{str(IDs[idx])}_wm.png")
            args.output = os.path.join(OUTPUT_PATH, f"{str(idx)}_{str(IDs[idx])}.png")

            """ DASHBOARD END """

            input_image = Image.open(args.input).convert("RGB")
            width, height = input_image.size
            factor = args.resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

            if args.edit == "":
                input_image.save(args.output)
                return

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                }
                torch.manual_seed(seed)
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image.save(args.output)


if __name__ == "__main__":
    main()
