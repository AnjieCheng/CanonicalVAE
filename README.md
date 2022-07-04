# CanonicalVAE
[[arxiv]](https://arxiv.org/abs/2204.01955)

This repository is the implementation of "Autoregressive 3D Shape Generation via Canonical Mapping".

Code release progress:
- [x] Training/inference code
- [ ] Installation and running instructions
- [x] Pre-trained checkpoints & generated samples
- [ ] Evaluation using Jupyter notebook
- [ ] Cleanup

## Requirements

By default, the Chamfer Loss module should work properly. If you failed to run the chamfer loss module, please see the following link and follow their instruction.
```setup
https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
```

To install EMD Loss, please follow the instruction in [here](https://github.com/AnjieCheng/CanonicalPAE/tree/main/external/emd). 
```setup
cd external/emd
python setup.py install
```
The installed `build` folder should be under `external/emd`.

# Dataset
For ShapeNet, we use the processed version provided by authors of PointFlow. Please download the dataset from this [link](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing).


# Evaluation
The pre-trained checkpoints and generated samples can be downloaded from this [link](https://drive.google.com/drive/folders/1NpSo8bBLR-vwOS5BK6pa6WRTnF1feuVl?usp=sharing). Please modify the `ckpt_path` in `configs/stage3/128/airplane_test.yaml`, `configs/stage3/128/car_test.yaml`, `configs/stage3/128/chair_test.yaml` to either your trained checkpoints or or pre-trained checkpoints. Note that the evaluation code only supports single GPU. If you use multiple GPUs, the code would still be runnable, but the calculated metrics may be incorrect.

```
python main.py --base configs/stage3/128/car_test.yaml -t False --gpus 0, -n car_test
```
The script will evaluate both auto-encoding performance and generation performance.

You can also directly evaluate the performance with our generated samples. Please download the samples from this [link](https://drive.google.com/drive/folders/1NpSo8bBLR-vwOS5BK6pa6WRTnF1feuVl?usp=sharing), and refer to `benchmark.ipynb` for details.

# Acknowledgement
This repo is built on top of [VQ-GAN](https://github.com/CompVis/taming-transformers). We thank the authors for sharing the codebase!