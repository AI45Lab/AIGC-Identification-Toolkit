# Triangle Attack

This repository contains code to reproduce results from the paper:

[Triangle Attack: A Query-efficient Decision-based Adversarial Attack](https://arxiv.org/abs/2112.06569) (ECCV 2022)

[Xiaosen Wang](http://xiaosenwang.com/), Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

## Requirements

+ python >= 3.6.5
+ pytorch == 1.7.x
+ numpy >= 1.15.4
+ imageio >= 2.6.1
+ torch_dct >= 0.1.5

## Qucik Start

### Prepare the data

Firstly, you should prepare your own benign images and victim models for attack. The pathes for the input images and model are set by ``--dataset_path`` and ``--modelpath``, respectively. You could also download our sampled 200 [images](https://drive.google.com/drive/folders/1X8cA1kpTe6cb8pGd8bMaKXJCOaSHZFUa?usp=sharing) used in the experiments and adopt the the pretrained models in pytorch. 


### Runing attack

You could run TA as follows:

```
CUDA_VISIBLE_DEVICES=gpuid python TA.py --dataset_path images --csv label.csv
```

The generated adversarial examples would be stored in directory `./output_folder`. We report the attack success rates under the thresholds of 0.1, 0.05 and 0.01 respectively.

# Citation

If you find the idea or code useful for your research, please consider citing our [paper](https://arxiv.org/abs/2112.06569):

```
@inproceedings{wang2022Triangle,
  author={Xiaosen Wang and Zeliang Zhang and Kangheng Tong and Dihong Gong and Kun He and Zhifeng Li and Wei Liu},
  booktitle = {European Conference on Computer Vision},
  title = {Triangle Attack: A Query-efficient Decision-based Adversarial Attack},
  year = {2022},
}
```

# Contact

Questions and suggestions can be sent to xswanghuster@gmail.com.