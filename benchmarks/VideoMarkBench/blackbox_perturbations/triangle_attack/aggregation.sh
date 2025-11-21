### Removal
CUDA_VISIBLE_DEVICES=0 python removal_TA.py --type removal --aggregation logit-mean &
CUDA_VISIBLE_DEVICES=1 python removal_TA.py --type removal --aggregation logit-median &
CUDA_VISIBLE_DEVICES=2 python removal_TA.py --type removal --aggregation bit-median &
CUDA_VISIBLE_DEVICES=3 python removal_TA.py --type removal --aggregation ba-mean &
CUDA_VISIBLE_DEVICES=4 python removal_TA.py --type removal --aggregation ba-median &
CUDA_VISIBLE_DEVICES=5 python removal_TA.py --type removal --aggregation detection-threshold &
CUDA_VISIBLE_DEVICES=6 python removal_TA.py --type removal --aggregation detection-median &

### Forgery
CUDA_VISIBLE_DEVICES=0 python forgery_TA.py --type forgery --aggregation logit-mean &
CUDA_VISIBLE_DEVICES=1 python forgery_TA.py --type forgery --aggregation logit-median &
CUDA_VISIBLE_DEVICES=2 python forgery_TA.py --type forgery --aggregation bit-median &
CUDA_VISIBLE_DEVICES=3 python forgery_TA.py --type forgery --aggregation ba-mean &
CUDA_VISIBLE_DEVICES=4 python forgery_TA.py --type forgery --aggregation ba-median &
CUDA_VISIBLE_DEVICES=5 python forgery_TA.py --type forgery --aggregation detection-threshold &
CUDA_VISIBLE_DEVICES=6 python forgery_TA.py --type forgery --aggregation detection-median &