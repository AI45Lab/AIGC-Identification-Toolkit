import os, argparse, cv2
import numpy as np 
import torchvision
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
from tqdm import tqdm
import torch.nn.functional as F


def image_to_tensor(image, normalize=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    if normalize:
        image /= 255.0
    tensor = torch.from_numpy(image)
    return tensor

def computePsnr(encoded_warped, image_input):
    mse = F.mse_loss(encoded_warped, image_input, reduction='none')
    mse = mse.mean([1, 2, 3])
    psnr = 10 * torch.log10(1**2 / mse)
    average_psnr = psnr.mean().item() 
    return average_psnr

def compute_psnr_ssim(decoded, original):
    
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    decoded_tensor = image_to_tensor(decoded)
    original_tensor = image_to_tensor(original)
    # psnr_value = computePsnr(decoded_tensor.unsqueeze(0), original_tensor.unsqueeze(0))
    psnr_value = psnr(decoded, original)
    ssim_value, _ = ssim(decoded, original, full=True, channel_axis=2)
    return psnr_value, ssim_value

def compute_lpips(decoded, original, loss_fn_alex, device):
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    decoded = torchvision.transforms.ToTensor()(decoded) * 2 - 1
    original = torchvision.transforms.ToTensor()(original) * 2 - 1
    decoded = decoded.to(device)
    original = original.to(device)
    lpips_alex = loss_fn_alex(decoded, original)
    return lpips_alex.item()

def compute_avg(*l): 
    return [np.mean(i) for i in l]

def main(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.join(args.input_dir, '512') 
    category = ['INSTRUCT_1K', 'DET_INVERSION_1K', 'STO_REGENERATION_1K', 'LOCAL_EDITING_5K', 'SVD_1K', 'DISTORTION_1K']
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    root_filename = folder
    for c in category:               
        if c == 'LOCAL_EDITING_5K':
            sub_category = ['10-20', '20-30', '30-40', '40-50', '50-60']
            print(c)
            for cs in sub_category:
                image_folder = os.path.join(root_filename, c, cs)
                image_path = [os.path.join(root_filename, c, cs, i) for i in os.listdir(image_folder)]
                avg_psnr, avg_ssim, avg_alex = [], [], []
                print(cs)
                for i in tqdm(image_path, desc="Processing images"):
                    decoded = cv2.imread(i, cv2.IMREAD_COLOR)
                    original = cv2.imread(os.path.join(args.wbench_path, c, cs, 'image', os.path.split(i)[-1].replace('_wm', '')), cv2.IMREAD_COLOR)
                    assert decoded.min() >= 0 and decoded.max() <= 255
                    assert original.min() >= 0 and original.max() <= 255
                    if decoded.shape[0] != 512:
                        original = cv2.resize(original, decoded.shape[:-1])
                    assert decoded.shape == original.shape
                    psnr_value, ssim_value = compute_psnr_ssim(decoded, original)
                    lpips_alex = compute_lpips(decoded, original, loss_fn_alex, device)
                    avg_psnr.append(psnr_value)
                    avg_ssim.append(ssim_value)
                    avg_alex.append(lpips_alex)
        else:
            image_folder = os.path.join(root_filename, c)
            image_path = [os.path.join(root_filename, c, i) for i in os.listdir(image_folder)]
            avg_psnr, avg_ssim, avg_alex = [], [], []
            print(c)
            for i in tqdm(image_path, desc="Processing images"):
                decoded = cv2.imread(i, cv2.IMREAD_COLOR)
                original = cv2.imread(os.path.join(args.wbench_path, c, 'image', os.path.split(i)[-1].replace('_wm', '')), cv2.IMREAD_COLOR)
                assert decoded.min() >= 0 and decoded.max() <= 255
                assert original.min() >= 0 and original.max() <= 255
                if decoded.shape[0] != 512:
                    original = cv2.resize(original, decoded.shape[:-1])
                assert decoded.shape == original.shape
                psnr_value, ssim_value = compute_psnr_ssim(decoded, original)
                lpips_alex = compute_lpips(decoded, original, loss_fn_alex, device)
                avg_psnr.append(psnr_value)
                avg_ssim.append(ssim_value)
                avg_alex.append(lpips_alex)

    avg_psnr, avg_ssim, avg_alex = compute_avg(avg_psnr, avg_ssim, avg_alex)
    print(f'PSNR: {avg_psnr}, SSIM: {avg_ssim}, LPIPS: {avg_alex}')


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./vine_encoded_wbench')
    parser.add_argument('--wbench_path', type=str, default='./W-Bench')
    args = parser.parse_args()
    main(args)