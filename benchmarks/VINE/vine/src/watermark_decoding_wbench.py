import os, torch, argparse, json
from stega_encoder_decoder import CustomConvNeXt
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name', type=str, default='Shilin-LU/VINE-R-Dec', help='pretrained_model_name')
parser.add_argument('--load_text', type=str, default='y', help='the flag to decide to use inputed text or random bits')
parser.add_argument('--groundtruth_message', type=str, default='Hello World!', help='the secret message to be encoded')
parser.add_argument('--unwm_images_dir', type=str, default='./W-Bench', help='the path to unwatermarked images for calculating TPR, FPR, and AUROC metrics')
parser.add_argument('--wm_images_dir', type=str, default='./output/edited_wmed_wbench', 
                    help="""The path to watermarked images for decoding. 
                    For orig (the unedited/undistorted watermarked images), the default path is './vine_encoded_wbench';
                    For edit, the default path is './output/edited_wmed_wbench';
                    For distort, the default path is './output/distorted_wmed_wbench/512'; """)
parser.add_argument('--unwm_acc_dict', type=str, default=None, help='save the detection results of original images to reduce computational load')
parser.add_argument('--output_dir', type=str, default='./output/detection_results/vine_r', help='the path to save detection results of watermarked images')
parser.add_argument('--decode_wbench_raw_data', type=str, default='y', help='decode the raw data of W-Bench')
parser.add_argument("--transformation", type=str, choices=["edit", "distort", "orig"], default="distort", help='specify the transformation that the watermarked images to be decoded undergo')
args = parser.parse_args()


def process_image(image_path, decoder, GTsecret, device):
    ### ============= load image =============
    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = t_val_256(image).unsqueeze(0).to(device)

    ### ============= watermark decoding & detection =============
    pred_watermark = decoder(image)
    pred_watermark = np.array(pred_watermark[0].cpu().detach())
    pred_watermark = np.round(pred_watermark)
    pred_watermark = pred_watermark.astype(int)
    pred_watermark_list = pred_watermark.tolist()
    groundtruth_watermark_list = GTsecret[0].cpu().detach().numpy().astype(int).tolist()

    same_elements_count = sum(x == y for x, y in zip(groundtruth_watermark_list, pred_watermark_list))
    acc = same_elements_count / 100
    # print('Decoding Bit Accuracy:', acc)
    return acc
                    

def process_images_in_folder(folder_path, decoder, GTsecret, device, unwm_acc=None, wm_flag=True, transformation='edit'):
    acc_total = {}
    # Traverse all files and subfolders in the folder
    for root, dirs, files in os.walk(folder_path):
        # If the current directory has no subfolders, it means this is the deepest-level folder
        if not dirs:
            acc = []
            
            if transformation == 'distort':
                skip_keywords = [
                    'mask', 
                    'DET_INVERSION_1K',
                    'SVD_1K', 'SVD_raw', 'INSTRUCT_1K', # 'DISTORTION_1K'
                    'LOCAL_EDITING_5K', 'STO_REGENERATION_1K'
                ]
            else:
                skip_keywords = [
                    'mask', 'SVD_raw',
                    # 'SVD_1K', 'DISTORTION_1K', 'INSTRUCT_1K',
                    # 'DET_INVERSION_1K', 'LOCAL_EDITING_5K', 'STO_REGENERATION_1K'
                ]
                
            if any(keyword in root for keyword in skip_keywords):
                continue
            
            for file in tqdm(files, desc="Processing images"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(root, file)
                    acc.append(process_image(image_path, decoder, GTsecret, device))

            # wm images
            if len(acc) != 0 and wm_flag == True:
                parent_folder, last_folder = os.path.split(root)
                
                # edited images
                if transformation == 'edit':  
                    category = os.path.basename(parent_folder)
                    category_scale = os.path.join(category, last_folder)
                # distorted images
                elif transformation == 'distort':
                    sub_parent_folder, distorted_category = os.path.split(parent_folder)
                    category_scale = os.path.join(distorted_category, last_folder)
                    category = os.path.basename(sub_parent_folder)
                # original images
                elif transformation == 'orig':               
                    if 'LOCAL_EDITING_5K' in root:
                        category = os.path.basename(parent_folder)
                    else:
                        category = os.path.basename(root)
                    category_scale = category
                else:
                    raise NotImplementedError("This function is not implemented. Please set the transformation to ['edit', 'distort', 'orig'].")
                
                all_acc = np.concatenate([unwm_acc[category], acc], axis=None)
                zeros_array = np.zeros(len(unwm_acc[category]), dtype=int)
                ones_array = np.ones(len(acc), dtype=int)
                ground_truth = np.concatenate([zeros_array, ones_array], axis=None)
                auc_1, low100_1, low1000_1 = compute_auroc_fpr(ground_truth, all_acc)
                
                acc_total[category_scale] = { 
                    'bit_acc': [round(np.mean(acc), 4)], 
                    'TPR@1%FPR': [round(low100_1, 4)], 
                    'TPR@0.1%FPR': [round(low1000_1, 4)], 
                    'AUROC': [round(auc_1, 4)]
                }
                    
            # unwm images
            elif len(acc) != 0 and wm_flag == False:
                if 'LOCAL_EDITING_5K' in root:
                    parent_folder, last_folder = os.path.split(root)
                    parent_folder_1, last_folder_1 = os.path.split(parent_folder)
                    category = os.path.basename(parent_folder_1)
                else:
                    parent_folder, last_folder = os.path.split(root)
                    category = os.path.basename(parent_folder)
                    
                print(category)
                
                acc_total[category] = acc
                
                if category == 'INSTRUCT_1K':
                    acc_total['INSTRUCT_UltraEdit'] = acc
                    acc_total['INSTRUCT_Pix2Pix'] = acc
                    acc_total['INSTRUCT_MagicBrush'] = acc
                elif category == 'LOCAL_EDITING_5K':
                    acc_total['REGION_cnInpaint'] = acc
                    acc_total['REGION_UltraEdit'] = acc
                else:
                    pass

    return acc_total


def compute_auroc_fpr(original, decoded):
    fpr, tpr, thresholds = metrics.roc_curve(original, decoded, pos_label=1)
    #acc_1 = np.max(1 - (fpr + (1 - tpr)) / 2)
    auc_1 = metrics.auc(fpr, tpr)
    low100_1 = tpr[np.where(fpr < 0.01)[0][-1]]
    low1000_1 = tpr[np.where(fpr < 0.001)[0][-1]]
    return auc_1, low100_1, low1000_1


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### ============= load model =============
    decoder = CustomConvNeXt.from_pretrained(args.pretrained_model_name)
    decoder.to(device)
    
    ### ============= load groundtruth message =============
    if args.load_text == 'y': # text to bits
        if len(args.groundtruth_message) > 12:
            print('Error: Can only encode 100 bits (12 characters)')
            raise SystemExit
        data = bytearray(args.groundtruth_message + ' ' * (12 - len(args.groundtruth_message)), 'utf-8')
        packet_binary = ''.join(format(x, '08b') for x in data)
        watermark = [int(x) for x in packet_binary]
        watermark.extend([0, 0, 0, 0])
    else: # random bits
        data = torch.randint(0, 2, (100,))

    if args.decode_wbench_raw_data == 'y':
        # the groundtruth message used for all baselines in w-bench
        watermark = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    watermark = torch.tensor(watermark, dtype=torch.float).unsqueeze(0)
    watermark = watermark.to(device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    target_unwm = os.path.join(args.output_dir, 'unwm_acc_dict.json')
    target_wm = os.path.join(args.output_dir, 'wm_acc_dict.json')
    
    with torch.no_grad():
        if args.unwm_acc_dict is None:
            unwm_acc = process_images_in_folder(
                args.unwm_images_dir, decoder, watermark, device, wm_flag=False,
                transformation=args.transformation
            )
            with open(target_unwm, 'w') as json_file:
                json.dump(unwm_acc, json_file)
        else:
            with open(args.unwm_acc_dict, 'r') as json_file:
                unwm_acc = json.load(json_file)
        
        print('original images are finishied')
        
        wm_acc = process_images_in_folder(
            args.wm_images_dir, decoder, watermark, device, unwm_acc, wm_flag=True, 
            transformation=args.transformation
        )
        with open(target_wm, 'w') as json_file:
            json.dump(wm_acc, json_file)

if __name__ == "__main__":
    main()
