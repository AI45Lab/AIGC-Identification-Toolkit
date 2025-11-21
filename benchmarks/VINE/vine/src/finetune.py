import os
import gc
import math
import lpips
import torch
import wandb
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import shutil
from diffusers.optimization import get_scheduler
from packaging import version
from transformers import AutoTokenizer, CLIPTextModel
import accelerate
from diffusers.utils.torch_utils import is_compiled_module
from vine.src.vine_turbo import VINE_Turbo, VAE_encode, VAE_decode
from vine.src.training_src.training_utils import parse_args
from vine.src.training_src.stega_utils import StegaData, get_secret_acc, count_parameters
import vine.src.training_src.extra_utils as extra_utils
from kornia import color
from vine.src.stega_encoder_decoder import CustomConvNeXt
from vine.src.training_src.transformations import TransformNet
from vine.src.training_src.wm_modules import Discriminator
from diffusers import StableDiffusionInstructPix2PixPipeline, DDIMScheduler
from vine.src.training_src.editing_dataset import EditData
import bchlib
BCH_POLYNOMIAL = 137
BCH_BITS = 5

IMAGE_SIZE = 256
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
   
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)
    
    watermark_encoder = VINE_Turbo.from_pretrained("Shilin-LU/VINE-B-Enc")
    
    decoder = CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec")
    decoder.to(accelerator.device)
    
    transform_net = TransformNet(device=accelerator.device)
        
    watermark_encoder.sec_encoder.requires_grad_(False)
    total_params, trainable_params, frozen_params = count_parameters(watermark_encoder.sec_encoder)
    print(f"sec_encoder total parameters: {total_params}")
    print(f"sec_encoder trainable parameters: {trainable_params}")
    print(f"sec_encoder frozen parameters: {frozen_params}")

    watermark_encoder.unet.requires_grad_(False)
    total_params, trainable_params, frozen_params = count_parameters(watermark_encoder.unet)
    print(f"UNET total parameters: {total_params}")
    print(f"UNET trainable parameters: {trainable_params}")
    print(f"UNET frozen parameters: {frozen_params}")
    
    watermark_encoder.vae_a2b.requires_grad_(False)
    total_params, trainable_params, frozen_params = count_parameters(watermark_encoder.vae_a2b)
    print(f"VAE total parameters: {total_params}")
    print(f"VAE trainable parameters: {trainable_params}")
    print(f"VAE frozen parameters: {frozen_params}")
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    watermark_encoder.vae_a2b.to(accelerator.device, dtype=weight_dtype)
    net_disc_a = Discriminator()

    gc.collect()
    torch.cuda.empty_cache()
    
    # crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()
    if args.enable_xformers_memory_efficient_attention:
        watermark_encoder.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        watermark_encoder.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    params_sec = list(decoder.parameters())
    params_gen = list(decoder.parameters())
    
    optimizer_gen = torch.optim.Adam(params_gen, lr=args.learning_rate)
    optimizer_sec = torch.optim.Adam(params_sec, lr=args.learning_rate)

    params_disc = list(net_disc_a.parameters())
    optimizer_disc = torch.optim.RMSprop(params_disc, lr=0.00001)

    dataset_train = EditData(args.dataset_folder, args.secret_size, size=(512, 512))
    print(f"==================Training dataset size: {len(dataset_train)}==================")
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True)  

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    class_name = model.__class__.__name__
                    sub_dir = f"{class_name}"
          
                    if isinstance(model, type(unwrap_model(watermark_encoder.sec_encoder))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(decoder))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(net_disc_a))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(watermark_encoder.unet))):
                        torch.save(model.state_dict(), os.path.join(output_dir, f'{sub_dir}.pth'))
                    elif isinstance(model, type(unwrap_model(watermark_encoder.vae_enc))):
                        torch.save(model.vae.state_dict(), os.path.join(output_dir, f'vae.pth'))
                    else:
                        pass

                    i -= 1

        def load_model_hook(models, input_dir):
                # pop models so that they are not loaded again
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(model, type(unwrap_model(watermark_encoder.sec_encoder))):
                        model.load_state_dict(torch.load(os.path.join(input_dir, 'StegaStampEncoder.pth')))
                    elif isinstance(model, type(unwrap_model(decoder))):
                        model.load_state_dict(torch.load(os.path.join(input_dir, 'CustomConvNeXt.pth')))
                    elif isinstance(model, type(unwrap_model(net_disc_a))):
                        model.load_state_dict(torch.load(os.path.join(input_dir, 'Discriminator.pth')))
                    elif isinstance(model, type(unwrap_model(watermark_encoder.unet))):
                        # model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                        model.load_state_dict(torch.load(os.path.join(input_dir, 'UNet2DConditionModel.pth')))
                    elif isinstance(model, VAE_encode):
                        model.vae.load_state_dict(torch.load(os.path.join(input_dir, 'vae.pth')))
                    elif isinstance(model, VAE_decode):
                        model.vae.load_state_dict(torch.load(os.path.join(input_dir, 'vae.pth')))
                    else:
                        pass

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
    
    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    
    lr_scheduler_sec = get_scheduler(args.lr_scheduler, optimizer=optimizer_sec,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
        
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.to(accelerator.device)
    net_lpips.requires_grad_(False)
    cross_entropy = torch.nn.BCELoss().to(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=args.revision, use_fast=False,)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)
    
    fixed_a2b_tokens = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    watermark_encoder.fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.unsqueeze(0).to(accelerator.device))[0].detach()
    del text_encoder, tokenizer, fixed_a2b_tokens  # free up some memory
    
    watermark_encoder.unet, watermark_encoder.vae_enc, watermark_encoder.vae_dec, net_disc_a, decoder, watermark_encoder.sec_encoder, transform_net, = accelerator.prepare(
        watermark_encoder.unet, watermark_encoder.vae_enc, watermark_encoder.vae_dec, net_disc_a, decoder, watermark_encoder.sec_encoder, transform_net, 
    )
    net_lpips, optimizer_gen, optimizer_sec, optimizer_disc, train_dataloader, lr_scheduler_sec, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_sec, optimizer_disc, train_dataloader, lr_scheduler_sec, lr_scheduler_gen, lr_scheduler_disc
    )

    # Move al networksr to device and cast to weight_dtype
    watermark_encoder.to(accelerator.device, dtype=weight_dtype)
    net_disc_a.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    decoder.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name, config=dict(vars(args)),
            init_kwargs={"wandb": {"name": 
                f"""
                {args.key_change}
                """
            }}          
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    first_epoch = 0
    global_step = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            gc.collect()
            torch.cuda.empty_cache()
    else:
        initial_global_step = 0
        
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    
    t_val = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    t_val1 = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC), 
    ])
        
    val_img = glob(os.path.join(args.val_folder, "*.jpg"))
    mask_img = Image.new("RGB", (512, 512), (255, 255, 255))
    pipe_pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                            torch_dtype=torch.float16,
                                                                            safety_checker=None,
                                                                            requires_safety_checker=False)
    pipe_pix2pix = pipe_pix2pix.to(accelerator.device)
    pipe_pix2pix.scheduler = DDIMScheduler.from_config(pipe_pix2pix.scheduler.config)
    gc.collect()
    torch.cuda.empty_cache()
    watermark_encoder.fixed_a2b_emb_base = watermark_encoder.fixed_a2b_emb_base.repeat(args.train_batch_size, 1, 1).to(dtype=weight_dtype)
    timesteps = torch.tensor([watermark_encoder.sched.config.num_train_timesteps - 1] * args.train_batch_size, device=accelerator.device).long()
    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [net_disc_a, watermark_encoder.sec_encoder, decoder]
            with accelerator.accumulate(*l_acc):
                img_a_256 = batch["cover_img_256"].to(dtype=weight_dtype) 
                img_a_512 = batch["cover_img"].to(dtype=weight_dtype)
                secret = batch["secret"].to(dtype=weight_dtype)
                
                #############
                no_im_loss = global_step < args.no_im_loss_steps
                l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
                secret_loss_scale = args.secret_loss_scale

                ##
                lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
                G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
                l2_edge_gain=0
                if global_step > args.l2_edge_delay:
                    l2_edge_gain = min(args.l2_edge_gain * (global_step-args.l2_edge_delay) / args.l2_edge_ramp, args.l2_edge_gain)

                loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
                yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
    
                #############
                encoded_image_256 = watermark_encoder(img_a_256, timesteps, secret)
                
                if torch.rand(1)[0] >= 0.8:
                    transformed_image = transform_net(encoded_image_256, img_a_256, global_step, args) # [-1, 1]
                    avg_psnr = extra_utils.computePsnr(0.5 * (encoded_image_256 + 1), 0.5 * (img_a_256 + 1))
                else:
                    residual_256 = encoded_image_256 - img_a_256
                    residual_512 = t_val1(residual_256)
                    encoded_image_512 = residual_512 + img_a_512
                    encoded_image_512 = 0.5 * (encoded_image_512 + 1)
                    encoded_image_512 = torch.clamp(encoded_image_512, min=0.0, max=1.0)
                    
                    level = min(int((global_step - 1) / 1000) + 1, 5)
                    if global_step < 6 * 1000:
                        guidance_scale = level + 4
                    else:
                        guidance_scale = np.random.randint(level + 1, level + 5) # 6-9

                    print("pix2pix guidance scale:" + str(guidance_scale))
                    image = pipe_pix2pix(
                        batch["prompt"],
                        image=encoded_image_512.to(torch.float16),
                        num_inference_steps=25,
                        image_guidance_scale=1.5,
                        guidance_scale=guidance_scale,
                        output_type="latent",
                    )
                    
                    transformed_image = transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC)(image[0])
                    avg_psnr = extra_utils.computePsnr(encoded_image_512, 0.5 * (img_a_512 + 1))
                
                transformed_image = transformed_image.to(device=accelerator.device, dtype=weight_dtype)
                transformed_image = (transformed_image / 2 + 0.5).clamp(0, 1)
                decoded_secret = decoder(transformed_image)
                decoded_secret = decoded_secret.to(dtype=weight_dtype)
                bit_acc, str_acc = get_secret_acc(secret, decoded_secret)
                
                lpips_loss = torch.mean(net_lpips(img_a_256, encoded_image_256))
                secret_loss = cross_entropy(decoded_secret, secret)

                size = (int(img_a_256.shape[2]), int(img_a_256.shape[3]))
                falloff_speed = 4
                falloff_im = np.ones(size)
                for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
                    falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
                    falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
                for j in range(int(falloff_im.shape[1] / falloff_speed)):
                    falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
                    falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
                falloff_im = 1 - falloff_im
                falloff_im = torch.from_numpy(falloff_im).float()
                falloff_im = falloff_im.to(accelerator.device)
                falloff_im *= l2_edge_gain   #[400,400]

                if args.use_rgb_Imageloss:
                    im_diff = encoded_image_256 - img_a_256
                    if not args.no_falloff_im:
                        im_diff += im_diff * (falloff_im.unsqueeze_(0))
                    image_loss = torch.mean((im_diff) ** 2)
                else:
                    encoded_image_yuv = color.rgb_to_yuv(encoded_image_256)
                    image_input_yuv = color.rgb_to_yuv(img_a_256)
                    im_diff = encoded_image_yuv - image_input_yuv
                    if not args.no_falloff_im:
                        im_diff += im_diff * (falloff_im.unsqueeze_(0))
                    yuv_loss = torch.mean((im_diff) ** 2, axis=[0, 2, 3])
                    yuv_scales = torch.Tensor(yuv_scales).to(device=img_a_256.device, dtype=weight_dtype)
                    image_loss = torch.dot(yuv_loss, yuv_scales)
                    
                D_output_fake_forG, _ = net_disc_a(encoded_image_256.detach())
                G_loss = D_output_fake_forG
            
                if no_im_loss:
                    loss = secret_loss
                    accelerator.backward(secret_loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
        
                    optimizer_sec.step()
                    lr_scheduler_sec.step()
                    optimizer_sec.zero_grad()
                else:
                    loss = loss_scales[0] * image_loss + loss_scales[1] * lpips_loss + loss_scales[2] * secret_loss
                    if not args.no_gan:
                        loss += loss_scales[3] * G_loss
                        
                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
        
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()

                    if not args.no_gan:
                        D_output_real, _ = net_disc_a(0.5* (img_a_256 + 1))
                        D_output_fake_forD, _ = net_disc_a(encoded_image_256.detach())
                        D_loss = D_output_real - D_output_fake_forD

                        accelerator.backward(D_loss, retain_graph=False)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(list(net_disc_a.parameters()), 0.25)
                        optimizer_disc.step()
                        lr_scheduler_disc.step()
                        optimizer_disc.zero_grad()
                        for p in net_disc_a.parameters():
                            p.data.clamp_(-0.01, 0.01)

            logs = {}
            logs["train_chart/loss"] = loss.detach().item()
            logs["train_chart/image_loss"] = image_loss.detach().item()
            logs["train_chart/lpips_loss"] = lpips_loss.detach().item()
            logs["train_chart/secret_loss"] = secret_loss.detach().item()
            logs["train_chart/bit_acc"] = bit_acc
            logs["train_chart/str_acc"] = str_acc
            logs["train_chart/psnr"] = avg_psnr
            if not args.no_gan:
                logs["train_chart/loss_gan"] = G_loss.detach().item()
                if not no_im_loss:
                    logs["train_chart/loss_D"] = D_loss.detach().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.viz_freq == 0:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["cover_img"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/cover_img": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(1)],
                                }
                                log_dict["train/watermarked"] = [wandb.Image(encoded_image_256[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(1)]
                                log_dict["train/transformed"] = [wandb.Image(transformed_image[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(1)]
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    # Get checkpoints:
                    if global_step % args.checkpointing_steps == 0:

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        print(f"Saving checkpoint to {save_path}")
                        accelerator.save_state(save_path)
                        
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    # save checkpoint of best image loss and secret loss
                    # if global_step > 200 * args.checkpointing_steps:
                    #     if avg_psnr < max_psnr:
                    #         save_path = os.path.join(args.output_dir, f"best-psnr-checkpoint")
                    #         accelerator.save_state(save_path)
                    #         max_psnr = avg_psnr
                    
                    if global_step % args.validation_steps == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        ### secret encoding
                        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
                        val_secret = "Hello"
                        if len(val_secret) > 7:
                            print('Error: Can only encode 56bits (7 characters) with ECC')
                            raise SystemExit

                        data = bytearray(val_secret + ' ' * (7 - len(val_secret)), 'utf-8')
                        ecc = bch.encode(data)
                        packet = data + ecc

                        packet_binary = ''.join(format(x, '08b') for x in packet)
                        secret_val = [int(x) for x in packet_binary]
                        secret_val.extend([0, 0, 0, 0])
                        secret_val = torch.tensor(secret_val, dtype=torch.float).unsqueeze(0)
                        secret_val = secret_val.to(accelerator.device, dtype=weight_dtype)
                        
                        val_bit_acc_total = 0
                        val_str_acc_total = 0
                        val_avg_psnr = 0
                        val_image_loss = 0
                        val_lpips_loss = 0
                        val_secret_loss = 0
                        with torch.no_grad():
                            for i in range(len(val_img)):
                                input_image = Image.open(val_img[i]).convert('RGB')
                                input_image = t_val(input_image)
                                input_image = input_image.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
                                input_image = input_image * 2 - 1
                                timesteps_val = torch.tensor([watermark_encoder.sched.config.num_train_timesteps - 1] * 1, device=input_image.device).long()
                                # fixed_a2b_emb = fixed_a2b_emb_base.repeat(1, 1, 1).to(dtype=weight_dtype)
                                
                                encoded_image = watermark_encoder(input_image, timesteps_val, secret_val)
                                
                                transformed_image_val = transform_net(encoded_image, input_image, global_step, args)
                                transformed_image_val = 0.5 * (transformed_image_val + 1)
                                # latent_for_decode = vae_enc(transformed_image_val, direction="a2b").to(transformed_image.dtype)
                                decoded_secret_val = decoder(transformed_image_val.to(dtype=weight_dtype))
                                decoded_secret_val = decoded_secret_val.to(dtype=weight_dtype)
                                
                                bit_acc, str_acc = get_secret_acc(secret_val, decoded_secret_val)
                                val_bit_acc_total += bit_acc
                                val_str_acc_total += str_acc
                                
                                val_avg_psnr += extra_utils.computePsnr(0.5 * (encoded_image + 1), 0.5 * (input_image + 1))
                                
                                val_lpips_loss += torch.mean(net_lpips(input_image, encoded_image))
                                val_secret_loss += cross_entropy(decoded_secret_val, secret_val)

                                encoded_image_yuv = color.rgb_to_yuv(encoded_image)
                                image_input_yuv = color.rgb_to_yuv(input_image)
                                im_diff = encoded_image_yuv - image_input_yuv
                                if not args.no_falloff_im:
                                    size = (int(input_image.shape[2]), int(input_image.shape[3]))
                                    falloff_speed = 4
                                    falloff_im = np.ones(size)
                                    for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
                                        falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
                                        falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
                                    for j in range(int(falloff_im.shape[1] / falloff_speed)):
                                        falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
                                        falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
                                    falloff_im = 1 - falloff_im
                                    falloff_im = torch.from_numpy(falloff_im).float()
                                    falloff_im = falloff_im.to(accelerator.device)
                                    falloff_im *= l2_edge_gain   #[400,400]
                                    im_diff += im_diff * (falloff_im.unsqueeze_(0))
                                yuv_loss = torch.mean((im_diff) ** 2, axis=[0, 2, 3])
                                yuv_scales = torch.Tensor(yuv_scales).to(device=input_image.device)
                                val_image_loss += torch.dot(yuv_loss, yuv_scales.to(weight_dtype))

                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    val_log_dict = {
                                        "val_img/cover": [wandb.Image(input_image[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(1)],
                                    }
                                    val_log_dict["val_img/watermarked"] = [wandb.Image(encoded_image[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(1)]
                                    val_log_dict["val_img/transformed"] = [wandb.Image(transformed_image_val[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(1)]
                                    tracker.log(val_log_dict)   

                        logs["val_chart/image_loss"] = val_image_loss.detach().item() / len(val_img)
                        logs["val_chart/lpips_loss"] = val_lpips_loss.detach().item() / len(val_img)
                        logs["val_chart/secret_loss"] = val_secret_loss.detach().item() / len(val_img)
                        logs["val_chart/bit_acc"] = val_bit_acc_total / len(val_img)
                        logs["val_chart/str_acc"] = val_str_acc_total / len(val_img)
                        logs["val_chart/psnr"] = val_avg_psnr / len(val_img)
                            
                    gc.collect()
                    torch.cuda.empty_cache()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args()
    main(args)
