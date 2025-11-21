import numpy as np
import cv2
import torch.nn.functional as F
import torch
import cv2
import itertools
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    coords = torch.from_numpy(np.stack(np.meshgrid(range(N_blur), range(N_blur), indexing='ij'), axis=-1)) - (0.5 * (N_blur-1)) # （7,7,2)
    manhat = torch.sum(torch.abs(coords), dim=-1)   # (7, 7)

    # nothing, default
    vals_nothing = (manhat < 0.5).float()           # (7, 7)

    # gauss
    sig_gauss = torch.rand(1)[0] * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords ** 2, dim=-1) /2. / sig_gauss ** 2)

    # line
    theta = torch.rand(1)[0] * 2.* np.pi
    v = torch.FloatTensor([torch.cos(theta), torch.sin(theta)]) # (2)
    dists = torch.sum(coords * v, dim=-1)                       # (7, 7)

    sig_line = torch.rand(1)[0] * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    w_line = torch.rand(1)[0] * (0.5 * (N_blur-1) + 0.1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists ** 2 / 2. / sig_line ** 2) * (manhat < w_line) # (7, 7)

    t = torch.rand(1)[0]
    vals = vals_nothing
    if t < (probs[0] + probs[1]):
        vals = vals_line
    else:
        vals = vals
    if t < probs[0]:
        vals = vals_gauss
    else:
        vals = vals

    v = vals / torch.sum(vals)      
    z = torch.zeros_like(v)     
    f = torch.stack([v,z,z, z,v,z, z,z,v], dim=0).reshape([3, 3, N_blur, N_blur])
    return f


def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    rnd_hue = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-rnd_hue, rnd_hue)
    rnd_brightness = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-rnd_bri, rnd_bri)
    return rnd_hue + rnd_brightness


def computePsnr(encoded_warped, image_input):
    mse = F.mse_loss(encoded_warped, image_input, reduction='none')
    mse = mse.mean([1, 2, 3])  
    psnr = 10 * torch.log10(1**2 / mse)
    average_psnr = psnr.mean().item()  

    return average_psnr


def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 8))

    for i in range(batch_size):
        tl_x = -d/2     # Top left corner, top
        tl_y = -d/2   # Top left corner, left
        bl_x = -d/2  # Bot left corner, bot
        bl_y = -d/2    # Bot left corner, left
        tr_x = d/2     # Top right corner, top
        tr_y = d/2   # Top right corner, right
        br_x = d/2   # Bot right corner, bot
        br_y = d/2   # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y +  image_size]], dtype = "float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i,0,:] = M_inv.flatten()[:8]
        Ms[i,1,:] = M.flatten()[:8]
    return Ms

def convert_tf_transform_to_kornia(M_tf, batch_size):
    M_kornia = torch.zeros((batch_size, 3, 3), dtype=M_tf.dtype, device=M_tf.device)
    M_kornia[:, 2, 2] = 1 
    M_kornia[:, 0, 0] = M_tf[:, 0]  # a0
    M_kornia[:, 0, 1] = M_tf[:, 1]  # a1
    M_kornia[:, 0, 2] = M_tf[:, 2]  # a2
    M_kornia[:, 1, 0] = M_tf[:, 3]  # a3
    M_kornia[:, 1, 1] = M_tf[:, 4]  # a4
    M_kornia[:, 1, 2] = M_tf[:, 5]  # a5
    M_kornia[:, 2, 0] = M_tf[:, 6]  # a6
    M_kornia[:, 2, 1] = M_tf[:, 7]  # a7
    return M_kornia

def round_only_at_0(x):
    cond = (torch.abs(x) < 0.5).float()
    return cond * (x ** 3) + (1 - cond) * x


# 1. RGB -> YCbCr
def rgb_to_ycbcr_jpeg(image):
    matrix = torch.tensor(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=image.dtype).T
    shift = torch.tensor([0., 128., 128.], dtype=image.dtype) 
    matrix=matrix.to(image.device)
    shift=shift.to(image.device)

    img = image.permute(0, 2, 3, 1)
    result = torch.tensordot(img, matrix, dims=1) + shift
    result = result.permute(0,3,1,2)
    return result

def ycbcr_to_rgb_jpeg(image):
    matrix = torch.tensor(
        [[1., 0., 1.402], [1., -0.344136, -0.714136], [1., 1.772, 0.]],
        dtype=image.dtype).t()
    shift = torch.tensor([0, -128, -128], dtype=image.dtype)
    matrix=matrix.to(image.device)
    shift=shift.to(image.device)

    image = image.permute(0, 2, 3, 1)
    result = torch.tensordot(image + shift, matrix, dims=1)
    result = result.permute(0,3,1,2)    
    return result


# 2. Chroma subsampling
def downsampling_420(image):
    # Assuming image is a PyTorch tensor of shape [batch, channels, height, width]

    # Split the channels
    y, cb, cr = image.split(1, dim=1)

    # Downsample Cb and Cr channels by 2 in both dimensions using average pooling
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2, padding=0)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2, padding=0)

    # Return the downsampled channels, now with cb and cr having half the width and height of y
    return (y.squeeze(axis=1), cb.squeeze(axis=1), cr.squeeze(axis=1))

# -2. Chroma upsampling
def repeat(x, k=2):
    x = x.unsqueeze(-1)
    x = x.repeat_interleave(k, dim=2).repeat_interleave(k, dim=1)
    return x.squeeze(-1)

def upsampling_420(y, cb, cr):
    cb_upsampled = repeat(cb)
    cr_upsampled = repeat(cr)
    image = torch.stack((y, cb_upsampled, cr_upsampled), dim=1)
    return image


# # 3. Block splitting
def image_to_patches(image):
    k=8
    height, width = image.shape[1:3]
    batch_size = image.shape[0]
    image_reshaped = image.view(batch_size, height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, -1, k, k)

# -3. Block joining
def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    batch_size = patches.size(0) 
    image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    return image_transposed.view(batch_size, height, width)


# 4. DCT
import itertools 
def dct_8x8(image):
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)

    tensor =torch.from_numpy(tensor).float().to(image.device, image.dtype)
    scale = torch.from_numpy(np.outer(alpha, alpha) * 0.25).float().to(image.device, image.dtype)

    result = scale * torch.tensordot(image, tensor, dims=2)
    return result

# -4. Inverse DCT
def idct_8x8(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = torch.from_numpy(np.outer(alpha, alpha)).float().to(image.device)
    image = image * alpha

    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    tensor = torch.from_numpy(tensor).float().to(image.device, image.dtype)

    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    
    result = 0.25 * torch.tensordot(image, tensor, dims=2) + 128
    return result


def y_quantize(image, rounding, factor=1):
    
    # 5. Quantizaztion
    y_table = torch.tensor(
        [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], 
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], 
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], 
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], 
        dtype=image.dtype).t()

    c_table = torch.empty((8, 8), dtype=image.dtype)
    c_table.fill_(99)
    c_table[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], 
                                    [24, 26, 56, 99], [47, 66, 99, 99]], 
                                    dtype=image.dtype).t()
    y_table=y_table.to(image.device)
    c_table=c_table.to(image.device)
    
    image = image / (y_table * factor)   
    image = rounding(image)
    return image

def c_quantize(image, rounding, factor=1):
    # 5. Quantizaztion
    y_table = torch.tensor(
        [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], 
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], 
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], 
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], 
        dtype=image.dtype).t()

    c_table = torch.empty((8, 8), dtype=image.dtype)
    c_table.fill_(99)
    c_table[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], 
                                    [24, 26, 56, 99], [47, 66, 99, 99]], 
                                    dtype=image.dtype).t()
    y_table=y_table.to(image.device)
    c_table=c_table.to(image.device)
    
    image = image / (c_table * factor)
    image = rounding(image)
    return image


# -5. Dequantization
def y_dequantize(image, factor=1):
    # 5. Quantizaztion
    y_table = torch.tensor(
        [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], 
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], 
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], 
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], 
        dtype=image.dtype).t()

    c_table = torch.empty((8, 8), dtype=image.dtype)
    c_table.fill_(99)
    c_table[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], 
                                    [24, 26, 56, 99], [47, 66, 99, 99]], 
                                    dtype=image.dtype).t()
    y_table=y_table.to(image.device)
    c_table=c_table.to(image.device)
    
    return image * (y_table * factor)

def c_dequantize(image, factor=1):
    # 5. Quantizaztion
    y_table = torch.tensor(
        [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], 
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], 
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], 
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], 
        dtype=image.dtype).t()
    
    c_table = torch.empty((8, 8), dtype=image.dtype)
    c_table.fill_(99)
    c_table[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], 
                                    [24, 26, 56, 99], [47, 66, 99, 99]], 
                                    dtype=image.dtype).t()
    y_table=y_table.to(image.device)
    c_table=c_table.to(image.device)
    
    return image * (c_table * factor)


def jpeg_compress_decompress(image,
                             downsample_c=True,
                             rounding=round_only_at_0,
                             factor=1):
    image *= 255
    height, width = image.shape[2:4]
    orig_height, orig_width = height, width

    if height % 16 != 0 or width % 16 != 0:
    # Round up to next multiple of 16
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width
        top = vpad // 2
        bottom = vpad - top
        left = wpad // 2
        right = wpad - left

        image = F.pad(image, (left, right, top, bottom), mode='reflect')

    # "Compression"
    image = rgb_to_ycbcr_jpeg(image)
    assert downsample_c==True
    y, cb, cr = downsampling_420(image)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = components[k] 
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding,
                      factor) if k in ('cb', 'cr') else y_quantize(
                          comp, rounding, factor)
        components[k] = comp

    # "Decompression"
    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(
            comp, factor)
        comp = idct_8x8(comp)
        if k in ('cb', 'cr'):
            comp = patches_to_image(comp, int(height/2), int(width/2))
        else:
            comp = patches_to_image(comp, height, width)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']    #y:[2,400,400]; cb:[2,200,200]; cr:[2,200,200]
    
    image = upsampling_420(y, cb, cr)
    image = ycbcr_to_rgb_jpeg(image)

    # crop to original size
    if orig_height != height or orig_width != width:
        image = image[:, :, top:-bottom, left:-right]
    
    image = torch.clamp(image, min=0., max=255.)
    image /= 255
    return image
