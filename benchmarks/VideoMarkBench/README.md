# VideoMarkBench

This repository is the official implementation of our paper [VideoMarkBench: Benchmarking Robustness of Video Watermarking](https://arxiv.org/abs/2505.21620), and includes both the code and dataset. It provides comprehensive resources for evaluating and benchmarking the robustness of video watermarking methods.



## Dataset

Our video dataset is generated using three state-of-the-art video generative models: Stable Video Diffusion, Sora, and Hunyuan Video. The dataset covers three distinct video styles: realistic, cartoon, and sci-fi.

[Download the dataset here](https://www.kaggle.com/datasets/zhengyuanjiang/videomarkbench/data)



## Perturbations

This repository includes two white-box, two black-box, and eight common video perturbations for evaluating watermark robustness.

- WEvade: A white-box perturbation originally designed to attack image watermarks. We extend it to the video domain for attacking video watermarks. More details are available [here](https://github.com/zhengyuan-jiang/WEvade).

- Square Attack: A score-based black-box perturbation method for images. We adapt it for each video watermarking method in our benchmark. See the [paper appendix](https://arxiv.org/abs/2505.21620) for more implementation details. The original implementation can be found [here](https://github.com/max-andr/square-attack).

- Triangle Attack: A label-based black-box perturbation method for images. The original implementation is available [here](https://github.com/xiaosen-wang/TA).



## Aggregations

There are several different strategies for aggregating watermarks (logits) decoded from individual frames in a video, as implemented in [aggregate.py](https://github.com/zhengyuan-jiang/VideoMarkBench/blob/main/aggregation.py).



## Video Quality Measurement

For more details, please refer to [VideoMetricEvaluator](https://github.com/Cookieser/VideoMetricEvaluator).



## Citation

If you find our work useful for your research, please consider citing the paper
```
@article{jiang2025videomarkbench,
  title={VideoMarkBench: Benchmarking Robustness of Video Watermarking},
  author={Jiang, Zhengyuan and Guo, Moyang and Li, Kecen and Hu, Yuepeng and Wang, Yupu and Huang, Zhicong and Hong, Cheng and Gong, Neil Zhenqiang},
  journal={arXiv preprint arXiv:2505.21620},
  year={2025}
}
```
