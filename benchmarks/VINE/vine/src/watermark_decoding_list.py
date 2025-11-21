import os, torch, argparse
from stega_encoder_decoder import CustomConvNeXt
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
import numpy as np

def main(args, device):
    ### ============= load model =============
    decoder = CustomConvNeXt.from_pretrained(args.pretrained_model_name)
    decoder.to(device)
    
    ### ============= load groundtruth message =============
    if args.load_text: # text to bits
        if len(args.groundtruth_message) > 12:
            print('Error: Can only encode 100 bits (12 characters)')
            raise SystemExit
        data = bytearray(args.groundtruth_message + ' ' * (12 - len(args.groundtruth_message)), 'utf-8')
        packet_binary = ''.join(format(x, '08b') for x in data)
        watermark = [int(x) for x in packet_binary]
        watermark.extend([0, 0, 0, 0])
        watermark = torch.tensor(watermark, dtype=torch.float).unsqueeze(0)
        watermark = watermark.to(device)
    else: # random bits
        data = torch.randint(0, 2, (100,))
        watermark = torch.tensor(data, dtype=torch.float).unsqueeze(0)
        watermark = watermark.to(device)
        
    ### ============= load image =============
    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
    ])
    
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            
            image = Image.open(input_path).convert("RGB")
            image = t_val_256(image).unsqueeze(0).to(device)

            ### ============= watermark decoding & detection =============
            pred_watermark = decoder(image)
            pred_watermark = np.array(pred_watermark[0].cpu().detach())
            pred_watermark = np.round(pred_watermark)
            pred_watermark = pred_watermark.astype(int)
            pred_watermark_list = pred_watermark.tolist()
            groundtruth_watermark_list = watermark[0].cpu().detach().numpy().astype(int).tolist()

            same_elements_count = sum(x == y for x, y in zip(groundtruth_watermark_list, pred_watermark_list))
            acc = same_elements_count / 100
            print('Decoding Bit Accuracy:', acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='./example/watermarked_img', help='path to the (edited) watermarked image')
    parser.add_argument('--pretrained_model_name', type=str, default='Shilin-LU/VINE-R-Dec', help='pretrained_model_name')
    parser.add_argument('--groundtruth_message', type=str, default='Hello World!', help='the secret message to be encoded')
    parser.add_argument('--load_text', type=bool, default=True, help='the flag to decide to use inputed text or random bits')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
    