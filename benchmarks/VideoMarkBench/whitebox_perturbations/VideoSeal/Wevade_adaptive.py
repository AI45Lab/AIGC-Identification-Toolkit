import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity
import numpy as np
import torch.nn.functional as F


def removal(watermarked_image, gt_watermark, model, args):
    watermarked_image_cloned = watermarked_image.clone()
    groundtruth_watermark = gt_watermark.clone()

    # WEvade_W_II target watermark selection.
    if args.WEvade_type == 'WEvade-W-II':
        target_watermark = torch.randint(0, 2, (1, args.secret_size)).float() - 0.5
    # WEvade_W_I target watermark selection.
    elif args.WEvade_type == 'WEvade-W-I':
        r_target_watermark = model.detect(watermarked_image, is_video=False)["preds"][:, 1:]
        r_target_watermark = (r_target_watermark > 0).float()
        target_watermark = 0.5 - r_target_watermark

    # Post-process the watermarked image.
    for _ in range(20):
        watermarked_image = watermarked_image.requires_grad_(True)
        decoded_logits = model.detect(watermarked_image, is_video=False)["preds"][:, 1:]

        loss = - (target_watermark * decoded_logits).sum()
        grads = torch.autograd.grad(loss, watermarked_image)
        with torch.no_grad():
            watermarked_image = watermarked_image - 0.02 * torch.sign(grads[0])
            # watermarked_image = watermarked_image - 0.01 * grads[0]
            watermarked_image = torch.clamp(watermarked_image, 0, 1)

    # calculate metrics
    bound = torch.norm(watermarked_image - watermarked_image_cloned, float('inf')).item()
    # clamp the image to 0-1
    ssim_watermarked = watermarked_image_cloned.squeeze().detach().cpu()
    ssim_post_processed = watermarked_image.squeeze().detach().cpu()
    ssim = structural_similarity(ssim_watermarked.numpy(), ssim_post_processed.numpy(), channel_axis=0,
                                 data_range=1.0).item()
    psnr = 10 * torch.log10(4 / torch.mean((ssim_watermarked - ssim_post_processed) ** 2)).mean().item()
    decoded_logits = model.detect(watermarked_image, is_video=False)["preds"][:, 1:]
    rounded_decoded_watermark = (decoded_logits > 0).float()
    bit_acc_groundtruth = rounded_decoded_watermark.eq(groundtruth_watermark).sum().item() / args.secret_size
    return bit_acc_groundtruth, bound, ssim, psnr, decoded_logits.squeeze().detach().cpu().numpy()


def forgery(original_image, gt_watermark, model, args):
    groundtruth_watermark = gt_watermark.clone()
    target_watermark = groundtruth_watermark - 0.5
    original_image_cloned = original_image.clone()

    # Post-process the watermarked image.
    for _ in range(20):
        original_image = original_image.requires_grad_(True)
        decoded_watermark = model.detect(original_image, is_video=False)["preds"][:, 1:]

        loss = - (target_watermark * decoded_watermark).sum()
        grads = torch.autograd.grad(loss, original_image)
        with torch.no_grad():
            original_image = original_image - 0.02 * torch.sign(grads[0])
            original_image = torch.clamp(original_image, 0, 1)

    # calculate metrics
    bound = torch.norm(original_image - original_image_cloned, float('inf')).item()
    ssim_original = original_image_cloned.squeeze().detach().cpu()
    ssim_post_processed = original_image.squeeze().detach().cpu()
    ssim = structural_similarity(ssim_original.numpy(), ssim_post_processed.numpy(), channel_axis=0,
                                 data_range=1.0).item()
    psnr = 10 * torch.log10(4 / torch.mean((original_image_cloned - original_image) ** 2)).mean().item()
    decoded_logits = model.detect(original_image, is_video=False)["preds"][:, 1:]
    rounded_decoded_watermark = (decoded_logits > 0).float()
    bit_acc_groundtruth = rounded_decoded_watermark.eq(groundtruth_watermark).sum().item() / args.secret_size
    return bit_acc_groundtruth, bound, ssim, psnr, decoded_logits.squeeze().detach().cpu().numpy()


def detect(image, groundtruth_watermark, model, args):
    decoded_logits = model.detect(image, is_video=False)["preds"][:, 1:]
    rounded_decoded_watermark = (decoded_logits > 0).float()
    bit_acc_groundtruth = rounded_decoded_watermark.eq(groundtruth_watermark).sum().item() / args.secret_size
    return bit_acc_groundtruth, 0.0, 1.0, np.inf, decoded_logits.squeeze().detach().cpu().numpy()

