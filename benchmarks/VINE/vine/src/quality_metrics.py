import cv2
import torchvision
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips, torch
import argparse


def compute_psnr_ssim(original_img, watermarked_img):
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)

    psnr_value = psnr(original_img, watermarked_img)
    ssim_value, _ = ssim(original_img, watermarked_img, full=True, channel_axis=2)
    print('\nPSNR:', psnr_value)
    print('SSIM:', ssim_value)
    return psnr_value, ssim_value

def compute_lpips(original_img, watermarked_img, loss_fn_alex, device):
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
    original_img = torchvision.transforms.ToTensor()(original_img) * 2 - 1
    watermarked_img = torchvision.transforms.ToTensor()(watermarked_img) * 2 - 1
    original_img = original_img.to(device)
    watermarked_img = watermarked_img.to(device)
    lpips_alex = loss_fn_alex(original_img, watermarked_img)
    print('LPIPS Alex:', lpips_alex.item())
    return lpips_alex.item()

def main(args, device):
    original_img = cv2.imread(args.input_path, cv2.IMREAD_COLOR)
    watermarked_img = cv2.imread(args.wmed_input_path, cv2.IMREAD_COLOR)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores  
    print('\n====================== Image Quality Metrics =======================')
    psnr_value, ssim_value = compute_psnr_ssim(original_img, watermarked_img)
    lpips_alex = compute_lpips(original_img, watermarked_img, loss_fn_alex, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./example/input/2.png', help='path to the input image')
    parser.add_argument('--wmed_input_path', type=str, default='./example/watermarked_img/2_wm.png', help='the directory to save the output')
    args = parser.parse_args()
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main(args, device)
