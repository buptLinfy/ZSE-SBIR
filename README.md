# Zero-Shot Everything Sketch-Based Image Retrieval, and in Explainable Style

In this repository, you can find the official PyTorch implementation of [Zero-Shot Everything Sketch-Based Image Retrieval, and in Explainable Style](https://bmvc2022.mpi-inf.mpg.de/0067.pdf) (CVPR2023).

Authors: [Fengyin Lin](https://github.com/MercuryMUMU) and [Yonggang Qi](https://qugank.github.io/), Beijing University of Posts and Telecommunications, Beijing, China.

> Abstract: Creative sketch is a universal way of visual expression, but translating images from an abstract sketch is very challenging. Traditionally, creating a deep learning model for sketch-to-image synthesis needs to overcome the distorted input sketch without visual details, and requires to collect large-scale sketch-image datasets. We first study this task by using diffusion models. Our model matches sketches through the cross domain constraints, and uses a classifier to guide the image synthesis more accurately. Extensive experiments confirmed that our method can not only be faithful to user’s input sketches, but also maintain the diversity and imagination of synthetic image results. Our model can beat GAN-based method in terms of generation quality and human evaluation, and does not rely on massive sketch-image datasets. Additionally, we present applications of our method in image editing and interpolation.

## Datasets

Pleace download and unzip images to `./datasets` directory.
### Sketchy
Please go to the [Sketchy official website](https://sketchy.eye.gatech.edu/) to download the _Sketches and Photos_ datasets.

### Tuberlin
Please go to the [Tuberlin official website](https://sketchy.eye.gatech.edu/) to download the _Sketches and Photos_ datasets.

### QuickDraw
Please go to the [QuickDraw official website](https://github.com/googlecreativelab/quickdraw-dataset) to download the datasets. The original data format is vector, please convert it to `.png` or `.jpg` before use it.

## Installation
The requirements of this repo can be found in [requirements.txt](https://github.com/XDUWQ/DiffSketching/blob/main/requirements.txt).
```bash
pip install -r requirements.txt
```

## Train

### Pretrained-ViT model

### Haperparameters
Here is a list of full options for the model:
```bash
lr,                   # learning rate.
log_dir,              # save log path.
dropout,              # dropout rate.
use_fp16,             # whether to use mixed precision training.
ema_rate,             # comma-separated list of EMA values
category,             # list of category name to be trained.
data_dir,             # the data sets path.
use_ddim,             # choose whether to use DDIM or DDPM
save_path,            # path to save vector results.
pen_break,            # determines the experience value of stroke break.
image_size,           # the max numbers of datasets.
model_path,           # path to save the trained model checkpoint.
class_cond,           # whether to use guidance technology.
batch_size,           # batch size of training.
emb_channels,         # Unet embedding channel numbers.
num_channels,         # the numbers of channels in Unet backbone.
out_channels,         # output channels in Unet. 
save_interval,        # saving models interval.
noise_schedule,       # the method of adding noise is linear by default.
num_res_blocks,       # numbers of resnet blocks in Unet backbone.
diffusion_steps,      # diffusion steps in the forward process.
schedule_sampler,     # the schedule of sampler.
fp16_scale_growth,    # the mixed precision scale growth.
use_scale_shift_norm, # whether to use scale shift norm. 
```

### Train ZSE-SBIR

Train the feature extraction network, please pay attention to modifying `image_root` before run.
```bash
python train.py 
```

Train model on Sketchy dataset.
```bash
python -u train.py --data_path sketchy_extend --test_class test_class_sketchy25 --cls_number 100 --batch 15 --epoch 30 -a 49 -c 0 -s ./checkpoints/sketchy_ext -r rn --split
```

Train model on Tuberlin dataset.
```bash
python scripts/image_train.py --data_dir [path/to/imagenet-datasets] \
                              --iterations 1000000 \
                              --anneal_lr True \
                              --batch_size 512 \
                              --lr 4e-4 \
                              --save_interval 10000 \
                              --weight_decay 0.05
```

Train model on Quickdraw dataset.
```bash
python scripts/classifier_train.py --data_dir [path/to/imagenet-datasets] \
                                   --iterations 1000000 \
                                   --anneal_lr True \
                                   --batch_size 512 \
                                   --lr 4e-4 \
                                   --save_interval 10000 \
                                   --weight_decay 0.05 \
                                   --image_size 256 \
                                   --classifier_width 256 \
                                   --classifier_pool attention \
                                   --classifier_resblock_updown True \
                                   --classifier_use_scale_shift_norm True
```

## Evaluation

### Our Trained Model

### Evaluate ZSE-SBIR

Evaluate on Sketchy dataset.
```bash
python test.py --model_path [/path/to/model] \
               --image_root [/path/to/reference-image] \
               --sketch_root [/path/to/reference-sketch] \
               --save_path [/path/to/save] \
               --batch_size 4 \
               --num_samples 50000 \
               --timestep_respacing ddim25 \
               --use_ddim True \
               --class_cond True \
               --image_size 256 \
               --learn_sigma True \
               --use_fp16 True \
               --use_scale_shift_norm True
```

Evaluate on Tuberlin dataset.
```bash
python test.py --model_path [/path/to/model] \
               --image_root [/path/to/reference-image] \
               --sketch_root [/path/to/reference-sketch] \
               --save_path [/path/to/save] \
               --batch_size 4 \
               --num_samples 50000 \
               --timestep_respacing ddim25 \
               --use_ddim True \
               --class_cond True \
               --image_size 256 \
               --learn_sigma True \
               --use_fp16 True \
               --use_scale_shift_norm True
```

Evaluate on Quickdraw dataset.
```bash
python test.py --model_path [/path/to/model] \
               --image_root [/path/to/reference-image] \
               --sketch_root [/path/to/reference-sketch] \
               --save_path [/path/to/save] \
               --batch_size 4 \
               --num_samples 50000 \
               --timestep_respacing ddim25 \
               --use_ddim True \
               --class_cond True \
               --image_size 256 \
               --learn_sigma True \
               --use_fp16 True \
               --use_scale_shift_norm True
```


## Bibtex
```
@inproceedings{Wang_2022_BMVC,
author    = {Qiang Wang and Di Kong and Fengyin Lin and Yonggang Qi},
title     = {DiffSketching: Sketch Control Image Synthesis with Diffusion Models},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0067.pdf}
}
```
