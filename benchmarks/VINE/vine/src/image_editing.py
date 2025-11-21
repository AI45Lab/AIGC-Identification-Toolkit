import os, torch, argparse
from accelerate.utils import set_seed
from PIL import Image
from diffusers import StableDiffusion3InstructPix2PixPipeline, StableDiffusionPipeline
from vine.src.editing_pipes import edit_by_UltraEdit, ddim_inversion


def main(args, device):
    # several examples of text-driven editing
    example_prompt_list = [
        "change the player's jersey color to blue", # whatever color you want
        "Replace the UK flag with a galaxy filled with stars",
        "Remove all the furniture and replace the concrete floor with a grassy field",
        "Replace the tools with colorful balloons",
        "Change the snowy grass to a frozen lake",
    ]
    
    watermarked_img = Image.open(args.input_path)
    # extract the corresponding prompt
    filename = os.path.basename(args.input_path)
    index = filename.split('_')[0]
    prompt = example_prompt_list[int(index)]

    if args.model == 'ultraedit':
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask",
                                                                        torch_dtype=torch.float16,
                                                                        safety_checker=None,
                                                                        requires_safety_checker=False)    
    elif args.model == 'inversion':
        repo = "CompVis/stable-diffusion-v1-4" # or runwayml/stable-diffusion-v1-5
        pipe = StableDiffusionPipeline.from_pretrained(repo, safety_checker=None, torch_dtype=torch.float32)
    else:
        print('Error: Model not found')
        raise SystemExit

    pipe.to(device)
    if args.model == 'ultraedit':
        edited_image = edit_by_UltraEdit(pipe, watermarked_img, prompt, text_guidance=7.5, num_inference_steps=50)
    elif args.model == 'inversion':
        edited_image = ddim_inversion(pipe, args.input_path, repo, device, inv_type='dpm', dpm_order=2, 
                                    num_steps=25, verify=True, prompt1='', prompt2='')
    else:
        print('Error: Model not found')
        raise SystemExit

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    edited_wm_img_path = os.path.join(args.output_dir, os.path.split(args.input_path)[-1][:-4]+'_edit.png')
    edited_image.save(edited_wm_img_path)
    print(f'\nEdited watermarked image saved at: {edited_wm_img_path}\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./example/watermarked_img/2_wm.png', help='path to the watermarked image')
    parser.add_argument('--output_dir', type=str, default='./example/edited_watermarked_img', help='the directory to save the output')
    parser.add_argument('--model', type=str, default="ultraedit", help='the editing model or algorithm to use')
    args = parser.parse_args()
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
    