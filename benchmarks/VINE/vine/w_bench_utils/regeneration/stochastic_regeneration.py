import torch, argparse, os
from diffusers import DDIMScheduler
from utils_sto_regeneration import DiffWMAttacker, ReSDPipeline


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_images_folder", type=str, default="./vine_encoded_wbench/512/STO_REGENERATION_1K")
    parser.add_argument("--edited_output_folder", type=str, default="./output/edited_wmed_wbench/STO_REGENERATION_1K")
    args = parser.parse_args()
    
# TODO ---------------------------------------- DASHBOARD START ------------------------------------------------------------
    for STEPS in [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]: 
        DEVICE = 'cuda'   
        OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{STEPS}") 
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
# TODO ---------------------------------------- DASHBOARD ENDS ------------------------------------------------------------
        print(f"\n* Processing STEPS={STEPS}")
        device = DEVICE

        pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        pipe.to(device)
        print('Finished loading model')

        attackers = {'diff_attacker': DiffWMAttacker(pipe, batch_size=5, noise_step=STEPS, captions={}),}

        image_names = sorted(os.listdir(args.wm_images_folder))
        wm_img_paths = [os.path.join(args.wm_images_folder, filename) for filename in image_names]
        att_img_paths = [os.path.join(OUTPUT_PATH, filename) for filename in image_names]

        for attacker_name, attacker in attackers.items():
            attackers[attacker_name].attack(wm_img_paths, att_img_paths)

        print('Finished attacking')
        