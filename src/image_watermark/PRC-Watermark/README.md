# Image Watermarking with PRC

[Update] The paper is accepted by ICLR 2025!

[Paper](https://arxiv.org/pdf/2410.07369)

If you find this work useful, please consider citing our paper:
```
@article{gunn2025undetectable,
  title={An undetectable watermark for generative image models},
  author={Gunn, Sam and Zhao, Xuandong and Song, Dawn},
  journal={arXiv preprint arXiv:2410.07369},
  year={2024}
}
```

PRC watermark is a method similar to Tree-Ring Watermark, where a diffusion model generates images embedded with a watermark as defined by a specified watermark key.

The output of the watermark detection is binary, indicating whether the watermark is detected or not.

We will also add how to embed and decode longer messages with PRC watermark later.

## Dependencies

The code is based on `python 3.11.9` and the packages specified in `requirements.txt`.

You can install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage

You need to specify the number of test images to generate and test on. The example uses 10. The watermark key is randomly generated and saved in the `keys` folder.

```bash
mkdir keys
python encode.py --test_num 10
```
```bash
python decode.py --test_num 10
```
Alternatively, you can specify a different folder for testing images (e.g., attacked images):
```bash
python decode.py --test_num 10 --test_path [path to test images]
```

You can also change the model and prompt in `model_id` and `dataset_id` respectively.

Additionally, you can set the targeted False Positive Rate (FPR) using the `fpr` parameter. The default value is 0.00001.

**Note**: Need to change the huggingface cache directory in `encode.py` and `decode.py`.

## Speedup

To improve decoding speed, the bottleneck lies in the `decoder_inv` function within `PRC-Watermark/src/inverse_stable_diffusion.py`.

You can reduce the number of steps to speed up processing. For example, modify:

```python
for i in self.progress_bar(range(100)):
```
to 20 steps:
```python
for i in self.progress_bar(range(20)):
```

## References

- [Treering Watermark](https://github.com/YuxinWenRick/tree-ring-watermark)
- [WAVES](https://github.com/umd-huang-lab/WAVES)
- [WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker)
- [Exact Inversion](https://github.com/smhongok/inv-dpm)
- [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading)
