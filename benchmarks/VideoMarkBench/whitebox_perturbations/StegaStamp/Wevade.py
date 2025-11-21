import torch
import numpy as np
from utils import project
from skimage.metrics import structural_similarity


def get_watermark(decoder, img, gt, args):
    if args.smooth:
        gaussian_noise = torch.randn((args.num_noise, *img.shape[-3:])).to(img.device)
        nimg = img + gaussian_noise * args.sigma
        dmsg = decoder(nimg)
        dmsg_round = dmsg.round().clip(0, 1)
        n_bwacc = torch.sum(gt == dmsg_round, dim=1) / gt.size(-1)
        sorted_indices = torch.argsort(n_bwacc)
        median_idx = sorted_indices[len(sorted_indices) // 2]
        median_watermark = dmsg[median_idx]
        return median_watermark
    else:
        return decoder(img).squeeze(0)


def removal(watermarked_image, gt_watermark, Decoder, criterion, args):
    watermarked_image_cloned = watermarked_image.clone()
    groundtruth_watermark = gt_watermark.clone().squeeze()
    r = args.rb
    lr = args.alpha
    epsilon = args.epsilon
    success = False

    # WEvade_W_II target watermark selection.
    if args.WEvade_type == 'WEvade-W-II':
        target_watermark = torch.randint(0, 2, (watermarked_image.shape[0], args.secret_size)).to(device=watermarked_image.device).float()
    # WEvade_W_I target watermark selection.
    elif args.WEvade_type == 'WEvade-W-I':
        chosen_watermark = Decoder(watermarked_image).round().clip(0, 1)
        # target_watermark = 1 - chosen_watermark
        target_watermark = 1 - groundtruth_watermark
        target_watermark = target_watermark.squeeze(0)

    for i in range(args.iteration):
        _watermarked_image = watermarked_image.requires_grad_(True)
        min_value, max_value = torch.min(_watermarked_image), torch.max(_watermarked_image)
        decoded_watermark = get_watermark(Decoder, _watermarked_image, groundtruth_watermark, args)
        # Post-process the watermarked image.
        loss = criterion(decoded_watermark, target_watermark)
        grads = torch.autograd.grad(loss, _watermarked_image)
        with torch.no_grad():
            _watermarked_image -= lr * torch.sign(grads[0])
            _watermarked_image = torch.clamp(_watermarked_image, min_value, max_value)
        del grads, loss
        # Projection.
        perturbation_norm = torch.norm(_watermarked_image - watermarked_image_cloned, float('inf'))
        # if args.type == 'PGD':
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            _watermarked_image = project(_watermarked_image, watermarked_image_cloned, c)

        decoded_watermark = get_watermark(Decoder, _watermarked_image, groundtruth_watermark, args)
        rounded_decoded_watermark = decoded_watermark.round().clip(0, 1)
        bit_acc_target = rounded_decoded_watermark.eq(target_watermark).sum().item() / args.secret_size
        watermarked_image = _watermarked_image
        # Early Stopping.
        if bit_acc_target >= 1 - epsilon:
            success = True
            break
        del _watermarked_image, decoded_watermark, bit_acc_target

    post_processed_watermarked_image = watermarked_image
    bound = torch.norm(post_processed_watermarked_image - watermarked_image_cloned, float('inf')).item()
    # clamp the image to 0-1
    ssim_watermarked = watermarked_image_cloned.squeeze().detach().cpu()
    ssim_post_processed = post_processed_watermarked_image.squeeze().detach().cpu()
    ssim = structural_similarity(ssim_watermarked.numpy(), ssim_post_processed.numpy(),channel_axis=0,data_range=1.0).item()
    psnr = 10 * torch.log10(4 / torch.mean((ssim_watermarked-ssim_post_processed)**2)).mean().item()
    bit_acc_groundtruth = rounded_decoded_watermark.eq(groundtruth_watermark).sum().item() / args.secret_size
    decoded_logits = get_watermark(Decoder, post_processed_watermarked_image, groundtruth_watermark, args).squeeze().detach().cpu()
    return bit_acc_groundtruth, bound, ssim, psnr, success, decoded_logits


def forgery(original_image, gt_watermark, Decoder, criterion, args):
    groundtruth_watermark = gt_watermark.clone().squeeze()
    original_image_cloned = original_image.clone()
    r = args.rb
    lr = args.alpha
    epsilon = args.epsilon
    success = False

    for i in range(args.iteration):
        _original_image = original_image.requires_grad_(True)
        min_value, max_value = torch.min(_original_image), torch.max(_original_image)
        decoded_watermark = get_watermark(Decoder, _original_image, groundtruth_watermark, args)
        # Post-process the watermarked image.
        loss = criterion(decoded_watermark, groundtruth_watermark)
        grads = torch.autograd.grad(loss, _original_image)
        with torch.no_grad():
            _original_image -= lr * torch.sign(grads[0])
            _original_image = torch.clamp(_original_image, min_value, max_value)
        del grads, loss
        # Projection.
        perturbation_norm = torch.norm(_original_image - original_image_cloned, float('inf'))
        # if args.type == 'PGD':
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            _original_image = project(_original_image, original_image_cloned, c)

        decoded_watermark = get_watermark(Decoder, _original_image, groundtruth_watermark, args)
        rounded_decoded_watermark = decoded_watermark.round().clip(0, 1)
        bit_acc_target = rounded_decoded_watermark.eq(groundtruth_watermark).sum().item() / args.secret_size
        original_image = _original_image
        # Early Stopping.
        if bit_acc_target >= 1 - epsilon:
            success = True
            break
        del _original_image, decoded_watermark, bit_acc_target

    post_processed_original_image = original_image
    bound = torch.norm(post_processed_original_image - original_image_cloned, float('inf')).item()
    ssim_original = original_image_cloned.squeeze().detach().cpu()
    ssim_post_processed = post_processed_original_image.squeeze().detach().cpu()
    ssim = structural_similarity(ssim_original.numpy(), ssim_post_processed.numpy(),channel_axis=0,data_range=1.0).item()
    psnr = 10 * torch.log10(4 / torch.mean((original_image_cloned-post_processed_original_image)**2)).mean().item()
    bit_acc_groundtruth = rounded_decoded_watermark.eq(groundtruth_watermark).sum().item() / args.secret_size
    decoded_logits = get_watermark(Decoder, post_processed_original_image, groundtruth_watermark, args).squeeze().detach().cpu()
    return bit_acc_groundtruth, bound, ssim, psnr, success, decoded_logits

