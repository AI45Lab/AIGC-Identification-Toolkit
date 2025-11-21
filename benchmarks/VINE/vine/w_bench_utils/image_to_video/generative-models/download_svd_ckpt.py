from huggingface_hub import snapshot_download
from huggingface_hub import login
login("") # login token needed! you can get token from huggingface profile
import os
os.environ['CURL_CA_BUNDLE'] = ''

#snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid", local_dir="./w_bench_utils/image_to_video/generative-models/checkpoints/svd")
snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt", local_dir="./w_bench_utils/image_to_video/generative-models/checkpoints/svd_xt")