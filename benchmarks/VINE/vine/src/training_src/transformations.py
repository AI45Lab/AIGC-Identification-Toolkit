import vine.src.training_src.extra_utils as extra_utils
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvtF
from vine.src.training_src.augment_imagenetc import RandomImagenetC
from PIL import Image
from vine.src.training_src.other_noises import Crop, Cropout, Dropout, Resize
from vine.src.training_src.jpeg_compression import JpegCompression
from vine.src.training_src.mbrs_noise import MbrsNoise
import torchvision.transforms as transforms
import pilgram
import random


class TransformNet(nn.Module):
    def __init__(self, device, rnd_bri=0.3, rnd_hue=0.1, do_jpeg=False, jpeg_quality=50, rnd_noise=0.02, 
                 rnd_sat=1.0, rnd_trans=0.1, contrast=[0.5, 1.5], ramp=1000, imagenetc_level=5, ic_up_level_interval=10000) -> None:
        super().__init__()
        self.rnd_bri = rnd_bri
        self.rnd_hue = rnd_hue
        self.jpeg_quality = jpeg_quality
        self.rnd_noise = rnd_noise
        self.rnd_sat = rnd_sat
        self.rnd_trans = rnd_trans
        self.contrast_low, self.contrast_high = contrast
        self.do_jpeg = do_jpeg
        # self.ramp = ramp
        self.register_buffer('step0', torch.tensor(0))  # large number
        if imagenetc_level > 0:
            self.imagenetc = ImagenetCTransform(max_severity=imagenetc_level)
        # self.crop = Crop()
        self.cropout = Cropout()
        self.dropout = Dropout(keep_ratio_range=[0.7, 0.9])
        self.resize = Resize(resize_ratio_range=[0.5, 2.0])
        self.ig_filter = IG_Filter()
        self.jpeg = JpegCompression(device=device)
        self.up_level_interval = 2000
        self.ic_up_level_interval = ic_up_level_interval
        self.mbrs_noise = MbrsNoise(['Combined([JpegMask(50),Jpeg(50)])'])
        # self.imagenetc_step = 3000
        # self.crop_resize_step = 6000
        # self.ig_filter_step = 12000
        
    def activate(self, global_step):
        if self.step0 == 0:
            print(f'[TRAINING] Activating TransformNet at step {global_step}')
            self.step0 = torch.tensor(global_step)
    
    def is_activated(self):
        return self.step0 > 0
    
    def forward(self, encoded_image, cover_img, global_step, args, p=0.999):
        # encoded_image: [batch_size, 3, H, W] in range [-1, 1]
        if torch.rand(1)[0] >= p:
            return encoded_image
        
        encoded_image_type = encoded_image.dtype
        encoded_image = encoded_image.to(torch.float32)
        
        if hasattr(self, 'imagenetc') and torch.rand(1)[0] < 0.5 and global_step > args.imagenetc_step: 
            level = min(int((global_step - args.imagenetc_step) / self.ic_up_level_interval) + 1, 7)
            
            if global_step < 6 * self.ic_up_level_interval + args.imagenetc_step:
                corrupt_strength = level
            else:
                corrupt_strength = np.random.randint(1, level + 1)
            
            # print(f'[TRAINING] ImagenetC level {corrupt_strength} at step {global_step}')
            encoded_image = self.imagenetc(encoded_image, corrupt_strength=corrupt_strength)
            
            return encoded_image.to(encoded_image_type)
        
        encoded_image = encoded_image * 0.5 + 0.5  # [-1, 1] -> [0, 1]      
        cover_img = cover_img * 0.5 + 0.5  # [-1, 1] -> [0, 1] 
        
        ramp_fn = lambda ramp: np.min([(global_step-self.step0.cpu().item()) / ramp, 1.])

        ### cropout/dropout and resize
        if global_step > args.crop_resize_step:
            a = torch.rand(1)[0]
            # if a <= 0.25:
            level = min(0.05 * (int((global_step - args.crop_resize_step) / self.up_level_interval) + 1), 0.3)
            
            if global_step < 6 * self.up_level_interval + args.crop_resize_step:
                max_tamper_area = level
            else:
                max_tamper_area = np.random.uniform(0.1, 0.3)
            
            # print(f'[TRAINING] Cropout level {max_tamper_area} at step {global_step}')
            
            encoded_image = self.cropout([encoded_image, cover_img], max_tamper_area=max_tamper_area, 
                                         height_ratio_range=(max(1 - max_tamper_area, 0.8), max(1 - max_tamper_area, 0.9)), 
                                         width_ratio_range=(max(1 - max_tamper_area, 0.8), max(1 - max_tamper_area, 0.9)))[0]
            encoded_image = encoded_image * 2 - 1  # [0, 1] -> [-1, 1]
            return encoded_image
            
            # if a > 0.25 and a <= 0.5:
            #     level = min(0.1 * (int((global_step - args.crop_resize_step) / self.up_level_interval) + 1), 0.5)
                
            #     if global_step < 6 * self.up_level_interval + args.crop_resize_step:
            #         resize_level = level
            #     else:
            #         resize_level = np.random.uniform(0.1, 0.5000001)
                    
            #     # print(f'[TRAINING] Resize level {resize_level} at step {global_step}')
            #     encoded_image = self.resize(encoded_image, 
            #                                 resize_ratio_min=1.0 - resize_level, 
            #                                 resize_ratio_max=1.0 + resize_level)
            #     encoded_image = encoded_image * 2 - 1  # [0, 1] -> [-1, 1]
            #     return encoded_image
            
        ### ig filter
        if torch.rand(1)[0] < 0.5 and global_step > args.ig_filter_step:
            encoded_image = self.ig_filter(encoded_image)
            encoded_image = encoded_image * 2 - 1  # [0, 1] -> [-1, 1]
            return encoded_image
        
        rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

        contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
        contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
        contrast_params = [contrast_low, contrast_high]

        rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

        # blur
        if not args.no_motionBlur:
            N_blur = args.N_blur
            f = extra_utils.random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                                        wmin_line=3)
            f = f.to(encoded_image.device, encoded_image.dtype)
            encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

        # noise
        if not args.no_gaussianNoise:
            noise = torch.normal(mean=0, std=rnd_noise, size=encoded_image.size(), dtype=encoded_image.dtype)
            noise = noise.to(encoded_image.device)
            encoded_image = encoded_image + noise
            encoded_image = torch.clamp(encoded_image, 0, 1)

        # contrast & brightness
        rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
        rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
        rnd_brightness = extra_utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, encoded_image.shape[0])  # [batch_size, 3, 1, 1]

        contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], contrast_params[1])
        contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
        contrast_scale = contrast_scale.to(encoded_image.device, encoded_image.dtype)
        rnd_brightness = rnd_brightness.to(encoded_image.device, encoded_image.dtype)
        if not args.no_contrast:
            encoded_image = encoded_image * contrast_scale
        if not args.no_bright:
            # print("encoded_image.shape: ", encoded_image.shape)
            # print("rnd_brightness.shape: ", rnd_brightness.shape)
            encoded_image = encoded_image + rnd_brightness
        encoded_image = torch.clamp(encoded_image, 0, 1)

        # saturation
        sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
        sat_weight = sat_weight.to(encoded_image.device, encoded_image.dtype)
        if not args.no_saturation:
            encoded_image_lum = torch.sum(encoded_image * sat_weight, dim=1, keepdim=True)  #这里原版写错了，把sum写成了mean。已改正
            encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

        # jpeg
        # encoded_image = encoded_image.reshape([-1, 3, 512, 512])
        
        if global_step < 10000:
            jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
            if jpeg_quality < 50:
                jpeg_factor = 5000. / jpeg_quality
            else:
                jpeg_factor = 200. - jpeg_quality * 2
            jpeg_factor = jpeg_factor / 100. + .0001
            
            if not args.no_jpeg:
                encoded_image = extra_utils.jpeg_compress_decompress(encoded_image, rounding=extra_utils.round_only_at_0,
                                                                    factor=jpeg_factor)
            encoded_image = encoded_image.to(encoded_image_type)
        else:
            if torch.rand(1)[0] < 0.4:
                jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
                if jpeg_quality < 50:
                    jpeg_factor = 5000. / jpeg_quality
                else:
                    jpeg_factor = 200. - jpeg_quality * 2
                jpeg_factor = jpeg_factor / 100. + .0001
                
                if not args.no_jpeg:
                    encoded_image = extra_utils.jpeg_compress_decompress(encoded_image, rounding=extra_utils.round_only_at_0,
                                                                        factor=jpeg_factor)
                encoded_image = encoded_image.to(encoded_image_type)
            elif torch.rand(1)[0] > 0.4 and torch.rand(1)[0] < 0.7:
                encoded_image = self.jpeg(encoded_image)
                encoded_image = encoded_image.to(encoded_image_type)
            else:
                encoded_image_ = encoded_image * 2 - 1
                cover_img_ = cover_img * 2 - 1
                encoded_image = self.mbrs_noise([encoded_image_, cover_img_]) # [-1, 1]
                encoded_image = encoded_image.to(encoded_image_type)
            
        encoded_image = encoded_image * 2 - 1  # [0, 1] -> [-1, 1]
        return encoded_image


class ImagenetCTransform(nn.Module):
    def __init__(self, max_severity=5) -> None:
        super().__init__()
        self.max_severity = max_severity
        self.tform = RandomImagenetC(max_severity=max_severity, phase='train')
    
    def forward(self, x, corrupt_strength=None):
        # x: [batch_size, 3, H, W] in range [-1, 1]
        img0 = x.detach().cpu().numpy()
        img = img0 * 127.5 + 127.5  # [-1, 1] -> [0, 255]
        img = img.transpose(0, 2, 3, 1).astype(np.uint8)
        img = [Image.fromarray(i) for i in img]
        img = [self.tform(i, corrupt_strength=corrupt_strength) for i in img]
        img = np.array([np.array(i) for i in img], dtype=np.float32)
        img = img.transpose(0, 3, 1, 2) / 127.5 - 1.  # [0, 255] -> [-1, 1]
        residual = torch.from_numpy(img - img0).to(x.device)
        img = torch.from_numpy(img).to(x.device)
        x = x + residual
        return img
    
    
class IG_Filter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tform = [
            pilgram._1977,
            pilgram.aden, 
            pilgram.brannan,
            pilgram.brooklyn,
            pilgram.clarendon,
            pilgram.earlybird,
            pilgram.gingham,
            pilgram.hudson,
            pilgram.inkwell,
            pilgram.kelvin,
            pilgram.lark,
            pilgram.lofi,
            pilgram.maven, 
            pilgram.mayfair,
            pilgram.moon,
            pilgram.nashville,
            pilgram.perpetua,
            pilgram.reyes,
            pilgram.rise,
            pilgram.slumber,
            pilgram.stinson,
            pilgram.toaster,
            pilgram.valencia,
            pilgram.walden,
            pilgram.willow,
            pilgram.xpro2,
        ]
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        
    def forward(self, x):
        # x: [batch_size, 3, H, W] in range [0, 1]
        processed_residual = []
        for single_img in x:
            encoded_pil_image = self.to_pil(single_img)
            selected_filter = random.choice(self.tform)
            filtered_img_pil = selected_filter(encoded_pil_image)
            filtered_img_tensor = self.to_tensor(filtered_img_pil).to(x.device)
            residual = filtered_img_tensor - single_img
            processed_residual.append(residual)
    
        processed_residual = torch.stack(processed_residual)
        x = x + processed_residual
        return x
