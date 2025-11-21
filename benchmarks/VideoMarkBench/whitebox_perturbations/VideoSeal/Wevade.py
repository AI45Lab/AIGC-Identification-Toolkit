import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity


def project(param_data, backup, epsilon):
    # If the perturbation exceeds the upper bound, project it back.
    r = param_data - backup
    r = epsilon * r
    return backup + r


def removal(watermarked_image, gt_watermark, model, criterion, args):
    watermarked_image_cloned = watermarked_image.clone()
    groundtruth_watermark = gt_watermark.clone()
    r = args.rb
    lr = args.alpha
    epsilon = args.epsilon
    success = False

    # WEvade_W_II target watermark selection.
    if args.WEvade_type == 'WEvade-W-II':
        target_watermark = torch.randint(0, 2, (1, args.secret_size)).float() - 0.5
    # WEvade_W_I target watermark selection.
    elif args.WEvade_type == 'WEvade-W-I':
        r_target_watermark = model.detect(watermarked_image, is_video=False)["preds"][:, 1:]
        r_target_watermark = (r_target_watermark > 0).float()
        target_watermark = 0.5 - groundtruth_watermark

    for i in range(args.iteration):
        _watermarked_image = watermarked_image.requires_grad_(True)
        min_value, max_value = torch.min(_watermarked_image), torch.max(_watermarked_image)
        decoded_watermark = model.detect(_watermarked_image, is_video=False)["preds"][:, 1:]
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

        decoded_watermark = model.detect(_watermarked_image, is_video=False)["preds"][:, 1:]
        rounded_decoded_watermark = (decoded_watermark > 0).float()
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
    decoded_logits = model.detect(post_processed_watermarked_image, is_video=False)["preds"][:, 1:].squeeze().detach().cpu()
    return bit_acc_groundtruth, bound, ssim, psnr, success, decoded_logits


def forgery(original_image, gt_watermark, model, criterion, args):
    groundtruth_watermark = gt_watermark.clone()
    groundtruth_watermark -= 0.5
    original_image_cloned = original_image.clone()
    r = args.rb
    lr = args.alpha
    epsilon = args.epsilon
    success = False

    for i in range(args.iteration):
        _original_image = original_image.requires_grad_(True)
        min_value, max_value = torch.min(_original_image), torch.max(_original_image)
        decoded_watermark = model.detect(_original_image, is_video=False)["preds"][:, 1:]
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

        decoded_watermark = model.detect(_original_image, is_video=False)["preds"][:, 1:]
        rounded_decoded_watermark = (decoded_watermark > 0).float()
        bit_acc_target = rounded_decoded_watermark.eq(groundtruth_watermark>0).sum().item() / args.secret_size
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
    bit_acc_groundtruth = rounded_decoded_watermark.eq(groundtruth_watermark>0).sum().item() / args.secret_size
    decoded_logits = model.detect(post_processed_original_image, is_video=False)["preds"][:, 1:].squeeze().detach().cpu()
    return bit_acc_groundtruth, bound, ssim, psnr, success, decoded_logits

