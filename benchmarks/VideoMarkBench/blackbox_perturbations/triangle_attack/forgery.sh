CUDA_VISIBLE_DEVICES=1 python forgery_TA.py --aggregation ba-mean &
CUDA_VISIBLE_DEVICES=2 python forgery_TA.py --aggregation ba-median &
CUDA_VISIBLE_DEVICES=3 python forgery_TA.py --aggregation logit-mean &
CUDA_VISIBLE_DEVICES=5 python forgery_TA.py --aggregation logit-median &
CUDA_VISIBLE_DEVICES=6 python forgery_TA.py --aggregation bit-median &
CUDA_VISIBLE_DEVICES=8 python forgery_TA.py --aggregation detection-threshold &
CUDA_VISIBLE_DEVICES=9 python forgery_TA.py --aggregation detection-median &