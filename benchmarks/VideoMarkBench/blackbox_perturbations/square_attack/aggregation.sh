### Removal
CUDA_VISIBLE_DEVICES=0 python removal_attack.py --dataset stable-video_realistic --model videoseal --aggregation logit-mean &
CUDA_VISIBLE_DEVICES=1 python removal_attack.py --dataset stable-video_realistic --model videoseal --aggregation logit-median &
CUDA_VISIBLE_DEVICES=2 python removal_attack.py --dataset stable-video_realistic --model videoseal --aggregation bit-median &
CUDA_VISIBLE_DEVICES=3 python removal_attack.py --dataset stable-video_realistic --model videoseal --aggregation ba-mean &
CUDA_VISIBLE_DEVICES=4 python removal_attack.py --dataset stable-video_realistic --model videoseal --aggregation ba-median &
CUDA_VISIBLE_DEVICES=5 python removal_attack.py --dataset stable-video_realistic --model videoseal --aggregation detection-threshold &
CUDA_VISIBLE_DEVICES=6 python removal_attack.py --dataset stable-video_realistic --model videoseal --aggregation detection-median &

### Forgery
CUDA_VISIBLE_DEVICES=0 python forgery_attack.py --aggregation logit-mean &
CUDA_VISIBLE_DEVICES=1 python forgery_attack.py --aggregation logit-median &
CUDA_VISIBLE_DEVICES=2 python forgery_attack.py --aggregation bit-median &
CUDA_VISIBLE_DEVICES=3 python forgery_attack.py --aggregation ba-mean &
CUDA_VISIBLE_DEVICES=4 python forgery_attack.py --aggregation ba-median &
CUDA_VISIBLE_DEVICES=5 python forgery_attack.py --aggregation detection-threshold &
CUDA_VISIBLE_DEVICES=6 python forgery_attack.py --aggregation detection-median &